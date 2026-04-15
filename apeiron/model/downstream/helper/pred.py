import torch
import torch.nn as nn
import torch.nn.functional as F

PRED_TYPES = {
    'hard_ce': 'softmax',
    'focal': 'softmax',
    'soft_ce': 'softmax',
    'kl_div': 'softmax',
    'mse': None,
    'mae': None,
    'bce': 'sigmoid',
    'multi_fc': 'sigmoid',
}

def check_mode(loss_type):
    mode = PRED_TYPES[loss_type]
    if mode not in ('softmax', 'sigmoid'):
        raise ValueError(f"mode must be 'softmax' or 'sigmoid', got '{mode}'")
    return mode

def apply_pred(mode, logits):
    if mode == 'softmax':
        return F.softmax(logits, dim=-1)   # (B, C)
    else:
        return logits.sigmoid()            # (B, C)