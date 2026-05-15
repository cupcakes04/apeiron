import torch
import torch.nn as nn
import torch.nn.functional as F

PRED_TYPES = {
    'hard_ce': 'softmax',
    'focal': 'softmax',
    'soft_ce': 'softmax',
    'kl_div': 'softmax',
    'mse': 'regression',
    'mae': 'regression',
    'bce': 'sigmoid',
    'multi_fc': 'sigmoid',
    'margin': 'rank',
    'listnet': 'rank',
}

def check_mode(loss_type):
    mode = PRED_TYPES[loss_type]
    if mode not in ('softmax', 'sigmoid', 'regression', 'rank'):
        raise ValueError(f"mode must be 'regression', 'softmax', 'sigmoid', or 'rank', got '{mode}'")
    return mode

def apply_pred(mode, logits):
    if mode == 'softmax':
        return F.softmax(logits, dim=-1)   # (B, C)
    elif mode == 'sigmoid':
        return logits.sigmoid()            # (B, C)
    elif mode == 'regression':
        return logits
    elif mode == 'rank':
        return logits # Scores for ranking can just be the raw logits
    