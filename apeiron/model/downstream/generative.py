import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Literal
from apeiron.utils import to_raw_list
from .helper import PerceiverResampler
import numpy as np

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

        self.modality = ['text']
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
        
        self.output = {}
        self.result = {}

    @staticmethod
    def get_target_modules(lm_name):
        if "gemma" in lm_name.lower():
            return ["q_proj", "v_proj"]
        else:
            return ["c_attn", "c_proj"]

    def forward(self, features, **kwargs):
        pooled_emb = self.emb_pooler(features) 
        visual_embeds = self.emb_to_gen_proj(pooled_emb)
        self.output = {'vis_emb': visual_embeds}
        return self.output

    def loss(self, text, visual_embeds=None, **kwargs):
            
        if visual_embeds is None: visual_embeds = self.output.get('vis_emb')
        device = visual_embeds.device
        B = visual_embeds.size(0)
        
        # Training / Loss Path
        gen_tokens = self.gen_tokenizer(
            to_raw_list(text), return_tensors="pt", padding=True, truncation=True
        ).to(device)
        
        text_embeds = self.lm.get_input_embeddings()(gen_tokens.input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        visual_labels = torch.full((B, self.num_visual_tokens), -100, dtype=torch.long, device=device)
        text_labels = gen_tokens.input_ids.masked_fill(gen_tokens.input_ids == self.gen_tokenizer.pad_token_id, -100)
        labels = torch.cat([visual_labels, text_labels], dim=1)
        
        visual_mask = torch.ones_like(visual_labels)
        attention_mask = torch.cat([visual_mask, gen_tokens.attention_mask], dim=1)
        
        gen_outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        gen_loss = gen_outputs.loss
        return {'text': {'gen_loss': gen_loss}}

    @torch.no_grad()
    def predict(self, visual_embeds=None, max_new_tokens=125, **kwargs):
        """(B, H) Mean pool of projected visual tokens for basic similarity search."""
        if visual_embeds is None: visual_embeds = self.output.get('vis_emb')
            
        outputs = self.lm.generate(
            inputs_embeds=visual_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.gen_tokenizer.pad_token_id,
            eos_token_id=self.gen_tokenizer.eos_token_id,
            do_sample=False,
        )
        self.result = {'pred_txt': self.gen_tokenizer.batch_decode(outputs, skip_special_tokens=True)}
        return self.result

    def metric(self, text: list, pred_txt: list = None, **kwargs):
        """Metrics for text generation (e.g. from Vision-Language Models).

        Evaluates generated strings against target strings.
        Currently implements exact match and basic character-level accuracy
        as placeholders for more complex NLP metrics (ROUGE, BLEU).

        Input:
            pred_txt (list[str]): List of generated strings.
            text (list[str]): List of ground-truth strings.

        Output:
            dict:
                - ``exact_match`` — fraction of perfectly matching strings.
                - ``char_acc``    — average character-level overlap ratio.
        """
        if pred_txt is None: pred_txt = self.result.get('pred_txt')
        
        if pred_txt is None or text is None or len(pred_txt) != len(text):
            return {'exact_match': 0.0, 'char_acc': 0.0}

        exact_matches = 0
        char_accs = []

        for p, t in zip(pred_txt, text):
            p = str(p).strip()
            t = str(t).strip()
            
            if p == t:
                exact_matches += 1
                char_accs.append(1.0)
            else:
                # Basic character overlap (SequenceMatcher could be used for better accuracy)
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, p, t).ratio()
                char_accs.append(ratio)

        return {'text': {
            'exact_match': exact_matches / len(text),
            'char_acc': float(np.mean(char_accs)) if char_accs else 0.0
        }}