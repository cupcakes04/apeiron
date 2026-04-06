import torch
import torch.nn as nn

class FmWrappers(nn.Module):
    """Wrapper for foundational models to standardize output format.
    
    Wraps various vision transformer models to provide consistent interface for
    extracting both class tokens and patch tokens regardless of model architecture.
    
    Args:
        vit_model (nn.Module): Pre-trained vision transformer model
        model_name (str): Model identifier for applying model-specific logic
    
    Attributes:
        wrappers (bool): Whether model-specific wrapping is applied
        model_name (str): Name of the wrapped model
        original_model: Original unwrapped model for saving
        model: Wrapped model or forward function
        token_count (int): Number of patch tokens (256 for most, 196 for DINOv3)
    """
    def __init__(self, vit_model, model_name):
        super().__init__()
        self.wrappers = True
        self.model_name = model_name
        self.original_model = vit_model
        
        if model_name in ['vir1', 'vir2']:
            self.model = vit_model
            self.token_count = 256
        elif model_name in ['hop0', 'hop1']:
            self.model = vit_model.forward_features
            self.token_count = 256
        elif model_name in ['dino3']:
            self.model = vit_model
            self.token_count = 196
        else:
            self.wrappers = False
            self.model = vit_model
            self.token_count = 256

    def forward(self, x):
        """Forward pass to extract class and patch tokens.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)
        
        Returns:
            tuple: (class_token, patch_tokens) where:
                - class_token: (B, F) global image features
                - patch_tokens: (B, num_tokens, F) local patch features
        """
        
        # Foward (if no wrappers specified, return)
        output = self.model(x)  # (B, T, F)
        if not self.wrappers:
            return output
        
        if self.model_name in ['dino3']:
            output = output.last_hidden_state #  1 + 4 + 16*16 = 201 (B, 201, 1280)

        class_token = output[:, 0:1].squeeze(1)  # (B,1,F) -> (B,F)
        patch_tokens = output[:, -self.token_count:]  # (B, 256, F)
        return class_token, patch_tokens
    
class FmTransform:
    """Wrapper for image preprocessing transforms with model-specific handling.
    
    Standardizes image preprocessing across different model architectures,
    handling special cases like DINOv3 which uses HuggingFace processors.
    
    Args:
        transform_func: Preprocessing function or processor
        model_name (str): Model identifier for applying model-specific logic
        device: Computation device for tensor operations
    
    Attributes:
        transform_func: The underlying transformation function
        model_name (str): Name of the model
        device: Target device for tensors
    """
    def __init__(self, transform_func, model_name, device):
        self.transform_func = transform_func
        self.model_name = model_name
        self.device = device

    def __call__(self, data):
        """Apply preprocessing transforms to input image.
        
        Args:
            data: Input image (PIL Image or numpy array)
        
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model input
        """
        if self.model_name in ['dino3']:
            inputs = self.transform_func(images=data, return_tensors="pt").to(self.device)
            return inputs["pixel_values"]
        else:
            out = self.transform_func(data)
            return out.unsqueeze(0)
        
    def __repr__(self):
        """String representation of the transform."""
        return f"{self.transform_func})"