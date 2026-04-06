import torch
import torch.nn as nn

class TextLoss(nn.Module):
    """
    Unifies outputs from multiple VLMs using Uncertainty Weighting.
    Includes an Auto-Bypass: if only one loss is active, weighting is skipped.
    """
    def __init__(self, loss_types: list):
        super().__init__()
        self.loss_types = loss_types
        # Initialize log_vars (uncertainty). Weight = exp(-log_var)
        self.log_vars = nn.Parameter(torch.zeros(len(loss_types)))

    def forward(self, gen_loss=None, con_loss=None, **kwargs):
        active_losses = [l for l in [gen_loss, con_loss] if l is not None]
        unified_out = {}
        
        # 1. Collect all valid 'txt_loss' values that are not None
        weighted_txt_losses = []

        # 2. Apply Dynamic Weighting or Bypass
        for i, loss in enumerate(active_losses):
            # If only one task is present, use standard weight (1.0) to prevent cheating
            if len(active_losses) <= 1:
                weighted_loss = loss
            else:
                s = self.log_vars[i]
                precision = torch.exp(-s)
                weighted_loss = precision * loss + s
            
            weighted_txt_losses.append(weighted_loss)

        # 3. Finalize Output
        if weighted_txt_losses:
            # We sum the weighted losses as per the AWL paper
            unified_out['txt_loss'] = sum(weighted_txt_losses)
            
        # 4. Pass through embeddings/metadata
        for ls_type, loss in active_losses.items():
            unified_out[ls_type] = loss

        return unified_out