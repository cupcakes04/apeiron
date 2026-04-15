import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Literal
from apeiron.utils import to_raw_list
from .helper import PerceiverResampler, TextMetrics

"""
common key names used accross codebase:
1. 'gen_fn' is used to get `generated_text`, put inside forward loop to run
2. 'img_emb' and 'wrd_emb' is for similarity-search, put inside `get_model_info` to use as a utility
"""

# ==============================================================================
# 3. Contrastive Vision-Language Model
# ==============================================================================
class ContrastiveVLM(nn.Module):
    """
    Contrastive Branch: Aligns emb features with a BERT model for retrieval
    and similarity tasks using InfoNCE loss.
    """
    def __init__(self, 
        in_features: int, 
        text_model_name: Literal["emilyalsentzer/Bio_ClinicalBERT", "medicalai/ClinicalBERT"] = "emilyalsentzer/Bio_ClinicalBERT",
        num_visual_tokens=32,
        projection_dim=512,
        mode: Literal['slide', 'tile'] = 'slide'):
        super().__init__()
        
        perceiver_hidden = 512
        self.emb_pooler = PerceiverResampler(
            in_features=in_features,
            mode=mode,
            hidden_dim=perceiver_hidden, 
            num_latents=num_visual_tokens, 
            num_heads=8
        )
        
        self.con_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.con_encoder = AutoModel.from_pretrained(text_model_name)
        
        self.img_to_con_proj = nn.Linear(perceiver_hidden, projection_dim)
        self.txt_to_con_proj = nn.Linear(self.con_encoder.config.hidden_size, projection_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

        self.img_buffer = []
        self.txt_buffer = []

    def get_img_emb(self, features):
        """Returns L2-normalized image embbeding (B, H) for similarity search."""
        device = next(self.emb_pooler.parameters()).device
        pooled_emb = self.emb_pooler(features.to(device))
        img_con_embeds = pooled_emb.mean(dim=1) # (B, perceiver_hidden)
        img_con_embeds = self.img_to_con_proj(img_con_embeds) # (B, projection_dim)
        return {'img_emb': img_con_embeds / img_con_embeds.norm(dim=-1, keepdim=True)}
        
    def get_wrd_emb(self, text):
        """Returns L2-normalized word embbeding (B, H) for similarity search."""
        device = next(self.con_encoder.parameters()).device
        con_tokens = self.con_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        txt_outputs = self.con_encoder(**con_tokens)
        txt_con_embeds = self.txt_to_con_proj(txt_outputs.pooler_output)
        return {'wrd_emb': txt_con_embeds / txt_con_embeds.norm(dim=-1, keepdim=True)}

    def cache(self, text, img_emb=None):
        if img_emb is None: img_emb = self.output.get('img_emb')
        if len(self.img_buffer) == 0:
            self.img_buffer = []
            self.txt_buffer = []
        self.img_buffer.append(img_emb)
        self.txt_buffer.extend(to_raw_list(text))
        
    def clear_cache(self):
        self.img_buffer = []
        self.txt_buffer = []


    def forward(self, features, **kwargs):
        self.output = self.get_img_emb(features)
        return self.output

    def loss(self, text, img_emb=None, **kwargs):
        self.cache(text, img_emb)
            
        if len(self.img_buffer) < 2:
            device = next(self.parameters()).device
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {'text': {'con_loss': zero, 'img_loss': zero, 'txt_loss': zero}}
            
        # Accumulate representations
        img_emb = torch.cat(self.img_buffer, dim=0)
        device = img_emb.device
        
        con_tokens = self.con_tokenizer(self.txt_buffer, return_tensors="pt", padding=True, truncation=True).to(device)
        txt_outputs = self.con_encoder(**con_tokens)
        txt_pooled = txt_outputs.pooler_output # (B, Hidden)
        txt_con_embeds = self.txt_to_con_proj(txt_pooled)
        txt_con_embeds = txt_con_embeds / txt_con_embeds.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        
        logits_per_img = logit_scale * img_emb @ txt_con_embeds.t()
        logits_per_txt = logits_per_img.t()
        
        con_labels = torch.arange(img_emb.size(0), device=device)
        img_loss = F.cross_entropy(logits_per_img, con_labels)
        txt_loss = F.cross_entropy(logits_per_txt, con_labels)
        con_loss = (img_loss + txt_loss) / 2

        return {'text': {'con_loss': con_loss, 'img_loss': img_loss, 'txt_loss': txt_loss}}
        
    def predict(self, **kwargs) -> dict:
        return {}

    def metric(self, **kwargs):
        return {}
        
