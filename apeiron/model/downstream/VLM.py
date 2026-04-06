import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Literal
from apeiron.utils import to_raw_list

VLM_LOSS_TYPES = ['gen_loss', 'con_loss']

"""
common key names used accross codebase:
1. 'gen_fn' is used to get `generated_text`, put inside forward loop to run
2. 'img_emb' and 'wrd_emb' is for similarity-search, put inside `get_model_info` to use as a utility
"""

# ==============================================================================
# 1. emb Feature Aggregation (Perceiver Resampler - Best Practice)
# ==============================================================================
class PerceiverResampler(nn.Module):
    """
    Industry standard for Vision-Language Models (used in Flamingo, LLaVA-Next).
    Compresses an arbitrary number of emb tiles (N) into a fixed number of 
    visual tokens (K) using Cross-Attention. 
    
    Two modes:
        - 'slide': input (B, N, F) -> output (B, num_latents, hidden_dim). Uses full
          cross-attention to compress N tiles into num_latents learned slots.
        - 'tile': input (B, F)    -> output (B, 1, hidden_dim). Input is already a
          single global embedding; just project it. No cross-attention needed.
    """
    def __init__(self, in_features, mode: Literal['slide', 'tile'], hidden_dim=512, num_latents=32, num_heads=8):
        super().__init__()
        self.mode = mode
        self.proj_in = nn.Linear(in_features, hidden_dim) if in_features != hidden_dim else nn.Identity()

        if mode == 'slide':
            self.num_latents = num_latents
            # Learnable latent queries (These act as 'slots' that pull information from the emb)
            self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_dim))
            # Cross-Attention: Latents (Query) attend to emb Features (Key/Value)
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            self.layer_norm_latents = nn.LayerNorm(hidden_dim)
            self.layer_norm_context = nn.LayerNorm(hidden_dim)
            # FFN to process the updated latents
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            self.layer_norm_ffn = nn.LayerNorm(hidden_dim)
            
        elif mode =='tile':
            self.num_latents = 1

    def forward(self, x):
        """
        'slide': x (B, N, F) -> (B, num_latents, hidden_dim)
        'tile' : x (B, F)    -> (B, 1,           hidden_dim)
        """
        if self.mode == 'tile':
            return self.proj_in(x).unsqueeze(1)  # (B, 1, hidden_dim)
            
        elif self.mode == 'slide':
            if x.ndim == 2:
                x = x.unsqueeze(0)

            # mode == 'tiles'
            B = x.size(0)
            context = self.proj_in(x)  # (B, N, hidden_dim)

            # Expand latents for the batch
            latents = self.latents.expand(B, -1, -1)  # (B, num_latents, hidden_dim)

            # 1. Cross Attention
            q = self.layer_norm_latents(latents)
            k = v = self.layer_norm_context(context)

            # self.cross_attn returns (attn_output, attn_weights)
            attn_out, _ = self.cross_attn(q, k, v)
            latents = latents + attn_out

            # 2. Feed Forward
            latents = latents + self.ffn(self.layer_norm_ffn(latents))
            
            return latents # (B, num_latents, hidden_dim)


# ==============================================================================
# 2. Generative Vision-Language Model
# ==============================================================================
class GenerativeVLM(nn.Module):
    """
    Generative Branch: Fine-tunes an LLM to auto-regressively generate text
    based on the input emb features.
    """
    def __init__(self, 
        in_features: int, 
        lm_model_name: Literal["google/gemma-2b", "google/gemma-7b", "distilgpt2"] = "distilgpt2",
        num_visual_tokens=32,
        use_lora=True,
        mode: Literal['slide', 'tile'] = 'slide'):
        
        super().__init__()
        if mode =='tile':
            num_visual_tokens = 1

        perceiver_hidden = 512
        self.emb_pooler = PerceiverResampler(
            in_features=in_features,
            mode=mode,
            hidden_dim=perceiver_hidden, 
            num_latents=num_visual_tokens, 
            num_heads=8
        )
        
        self.gen_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        if self.gen_tokenizer.pad_token is None:
            self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token
            
        self.lm = AutoModelForCausalLM.from_pretrained(lm_model_name)
        
        if use_lora:
            target_modules = self.get_target_modules(lm_model_name)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=8, 
                lora_alpha=16, 
                lora_dropout=0.1,
                target_modules=target_modules
            )
            self.lm = get_peft_model(self.lm, peft_config)

        self.lm_hidden_size = getattr(self.lm.config, "hidden_size", getattr(self.lm.config, "n_embd", None))
        self.num_visual_tokens = num_visual_tokens
        
        self.emb_to_gen_proj = nn.Sequential(
            nn.Linear(perceiver_hidden, self.lm_hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.lm_hidden_size * 2, self.lm_hidden_size)
        )

    def forward(self, features, text=None, **kwargs):
        if text is None:
            return {
                'gen_loss': 0,
                'gen_fn': lambda f=features: self.generate_text(features=f),
            }

        pooled_emb = self.emb_pooler(features) 
        visual_embeds = self.emb_to_gen_proj(pooled_emb)
            
        # Training / Loss Path
        gen_tokens = self.gen_tokenizer(
            to_raw_list(text), return_tensors="pt", padding=True, truncation=True
        ).to(features.device)
        
        text_embeds = self.lm.get_input_embeddings()(gen_tokens.input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        visual_labels = torch.full((features.size(0), self.num_visual_tokens), -100, dtype=torch.long, device=features.device)
        text_labels = gen_tokens.input_ids.masked_fill(gen_tokens.input_ids == self.gen_tokenizer.pad_token_id, -100)
        labels = torch.cat([visual_labels, text_labels], dim=1)
        
        visual_mask = torch.ones_like(visual_labels)
        attention_mask = torch.cat([visual_mask, gen_tokens.attention_mask], dim=1)
        
        gen_outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        gen_loss = gen_outputs.loss
        return {
            'gen_loss': gen_loss,
            'gen_fn': lambda f=features: self.generate_text(features=f),
        }

        
    @torch.no_grad()
    def generate_text(self, features, max_new_tokens=125):
        """(B, H) Mean pool of projected visual tokens for basic similarity search."""
        pooled_emb = self.emb_pooler(features)
        visual_embeds = self.emb_to_gen_proj(pooled_emb)
        outputs = self.lm.generate(
            inputs_embeds=visual_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.gen_tokenizer.pad_token_id,
            eos_token_id=self.gen_tokenizer.eos_token_id,
            do_sample=False,
        )
        return {'generated_text': self.gen_tokenizer.batch_decode(outputs, skip_special_tokens=True)}

    @staticmethod
    def get_target_modules(lm_name):
        if "gemma" in lm_name.lower():
            return ["q_proj", "v_proj"]
        else:
            return ["c_attn", "c_proj"]
            
    def get_model_info(self):
        return {'modality': ['VLM']}

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

    def forward(self, features, text=None, **kwargs):
        if text is None:
            return {'con_loss': 0}

        pooled_emb = self.emb_pooler(features)
        
        img_con_embeds = pooled_emb.mean(dim=1) # (B, perceiver_hidden)
        img_con_embeds = self.img_to_con_proj(img_con_embeds) # (B, projection_dim)
        img_con_embeds = img_con_embeds / img_con_embeds.norm(dim=-1, keepdim=True)
            
        con_tokens = self.con_tokenizer(to_raw_list(text), return_tensors="pt", padding=True, truncation=True).to(features.device)
        txt_outputs = self.con_encoder(**con_tokens)
        txt_pooled = txt_outputs.pooler_output # (B, Hidden)
        txt_con_embeds = self.txt_to_con_proj(txt_pooled)
        txt_con_embeds = txt_con_embeds / txt_con_embeds.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        
        logits_per_img = logit_scale * img_con_embeds @ txt_con_embeds.t()
        logits_per_txt = logits_per_img.t()
        
        con_labels = torch.arange(img_con_embeds.size(0), device=img_con_embeds.device)
        loss_emb = F.cross_entropy(logits_per_img, con_labels)
        loss_txt = F.cross_entropy(logits_per_txt, con_labels)
        con_loss = (loss_emb + loss_txt) / 2
        
        return {'con_loss': con_loss}
        
    @torch.no_grad()
    def get_img_emb(self, features):
        """(B, H) Returns L2-normalized image embbeding for similarity search."""
        device = next(self.emb_pooler.parameters()).device
        pooled_emb = self.emb_pooler(features.to(device))
        img_con_embeds = pooled_emb.mean(dim=1)
        img_con_embeds = self.img_to_con_proj(img_con_embeds)
        return {'img_emb': img_con_embeds / img_con_embeds.norm(dim=-1, keepdim=True)}
        
    @torch.no_grad()
    def get_wrd_emb(self, text):
        """(B, H) Returns L2-normalized word embbeding for similarity search."""
        device = next(self.con_encoder.parameters()).device
        con_tokens = self.con_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        txt_outputs = self.con_encoder(**con_tokens)
        txt_con_embeds = self.txt_to_con_proj(txt_outputs.pooler_output)
        return {'wrd_emb': txt_con_embeds / txt_con_embeds.norm(dim=-1, keepdim=True)}

    def get_model_info(self):
        return {
            'modality': ['VLM'], 
            "embedding_fns": {'img_emb_fn': self.get_img_emb, 'wrd_emb_fn': self.get_wrd_emb}
        }
