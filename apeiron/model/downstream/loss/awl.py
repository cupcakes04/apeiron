import torch
import torch.nn as nn
import torch.nn.functional as F

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, task_keys: list):
        super().__init__()
        self.keys = task_keys
        # Initialize log_var at 0 (Weight = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(len(task_keys)))

    def forward(self, loss_dict: dict):
        # Initialize total_loss on the correct device
        device = next(iter(loss_dict.values())).device
        total_loss = torch.zeros(1, device=device)

        # 1. Count how many tasks are actually present in this batch
        num_tasks = len([k for k in self.keys if k in loss_dict])

        for i, key in enumerate(self.keys):
            if key not in loss_dict:
                continue
                
            loss = loss_dict[key]
            # 2. AUTO-BYPASS: If only 1 task is present, use standard weight (1.0)
            # This prevents the log_var from "cheating" when it's alone.
            if num_tasks <= 1:
                weighted_loss = loss
            else:
                s = self.log_vars[i]
                # Standard AWL formula
                weighted_loss = torch.exp(-s) * loss + s
            
            total_loss += weighted_loss
            
        return total_loss

    def get_importance(self):
        """
        - Weight = exp(-log_var). 
        - High log_var (uncertainty) = Low weight.
        """
        with torch.no_grad():
            # Correctly mapping keys to the learned weights
            return {
                key: torch.exp(-self.log_vars[i]).item() 
                for i, key in enumerate(self.keys)
            }
