from pathlib import Path
from typing import Literal
import torch
import timm 
import torchvision.transforms.v2 as transforms
from transformers import AutoModel, AutoImageProcessor, AutoModel
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from timm.layers import SwiGLUPacked
import pickle
from apeiron.utils import mkdir, get_device
from .wrappers import FmWrappers, FmTransform

class Backbone:
    """Manages foundational models for whole slide image feature extraction.
    
    Handles loading, caching, and switching between different vision transformer models
    optimized for histopathology. Supports multiple state-of-the-art models including
    H-optimus, Virchow, UNI, CONCH, and others.
    
    Args:
        models_save_dir (str or Path): Directory to save/load model weights and transforms
        device (str, optional): Computation device ('cuda' or 'cpu'). Auto-detected if None
        **kwargs: Additional keyword arguments
    
    Attributes:
        models_save_dir (Path): Directory for model storage
        device (torch.device): Computation device (GPU/CPU)
        model_name (str): Currently selected model name
        model (FmWrappers): Wrapped model for feature extraction
        transform (FmTransform): Image preprocessing pipeline
        model_library (dict): Cache of loaded models to avoid reloading
    """
    def __init__(self, models_save_dir, device=None, **kwargs):

        super().__init__(**kwargs)
        self.models_save_dir = Path(models_save_dir)
        self.device = get_device(device, 'backbone')
        self.model_name = None
        self.model = None
        self.transform = None
        self.model_library = {}
        self.model_feats_dim = {
            'hop0': 1536, 'hop1': 1536, 
            'vir1': 1280, 'vir2': 1280, 
            'ch15': 512, 'uni2h': 1024, 
            'mstar': 768, 'dino3': 1280,
        }
    
    def create_model(self, model_name=None, zero_shot=False):
        """Create and initialize a foundational model with its preprocessing transforms.
        
        Instantiates vision transformer models from various sources (timm, HuggingFace)
        and wraps them for consistent feature extraction interface.

        Args:
            model_name (str): Name of foundational model. Supported models:
                - 'hop0', 'hop1': H-optimus models (Bioptimus)
                - 'vir1', 'vir2': Virchow models (Paige AI)
                - 'ch15': CONCH 1.5 (Mahmood Lab)
                - 'uni2h': UNI2-h (Mahmood Lab)
                - 'mstar': mSTAR (Wangyh)
                - 'dino3': DINOv3 (Meta)
            zero_shot (bool): Return zero-shot model wrapper (for CONCH/Virchow). Default False

        Returns:
            None (sets self.model and self.transform)
        """
        self.model_name = model_name
        
        if self.model_name == "hop0":
            model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
            transform = transforms.Compose([
                transforms.ToImage(), 
                transforms.Resize(size=224, antialias=True),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617), 
                    std=(0.211883, 0.230117, 0.177517)
                ),
            ])
        
        if self.model_name == "hop1":
            model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False)
            transform = transforms.Compose([
                transforms.ToImage(), 
                transforms.Resize(size=224, antialias=True),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617), 
                    std=(0.211883, 0.230117, 0.177517)
                ),
            ])
            
        if self.model_name == "mstar":
            model = timm.create_model('hf-hub:Wangyh/mSTAR', pretrained=True, init_values=1e-5, dynamic_img_size=True)
            transform = transforms.Compose([
                transforms.ToImage(), 
                transforms.Resize(size=224, antialias=True),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                ),
            ])
            
        if self.model_name == "ch15":
            titan = AutoModel.from_pretrained('MahmoodLab/TITAN', local_files_only=True, trust_remote_code=True)
            if zero_shot: return titan
            model, transform = titan.return_conch()
            
        if self.model_name == "uni2h":
            timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
            model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            
        if self.model_name == 'vir2':
            model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        
        if self.model_name == 'vir1':
            if zero_shot: return AutoModel.from_pretrained('paige-ai/Prism', local_files_only=True, trust_remote_code=True)
            model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        
        if self.model_name == "dino3":
            pretrained_model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
            model = AutoModel.from_pretrained(
                pretrained_model_name, 
                device_map="auto", 
            )
            transform = AutoImageProcessor.from_pretrained(pretrained_model_name)
            
        # Wrap the model for class & patch tokens
        self.model = FmWrappers(model, self.model_name)
        self.transform = FmTransform(transform, self.model_name, self.device)

    def select_model(self, model_name: Literal['hop0', 'hop1', 'vir1', 'vir2', 'ch15', 'uni2h', 'mstar', 'dino3']):
        """Select and load a foundational model for feature extraction.
        
        Checks if model is already loaded or cached, otherwise loads from disk or creates new.
        Automatically moves model to appropriate device and sets to evaluation mode.

        Args:
            model_name (str): Model identifier. Options:
                - 'hop0': H-optimus-0 (1536-dim features)
                - 'hop1': H-optimus-1 (1536-dim features)
                - 'vir1': Virchow (2560-dim features)
                - 'vir2': Virchow2 (1280-dim features)
                - 'ch15': CONCH 1.5 (512-dim features)
                - 'uni2h': UNI2-h (1536-dim features)
                - 'mstar': mSTAR (768-dim features)
                - 'dino3': DINOv3 (1280-dim features)

        Returns:
            tuple: (model_name, model, transform) for the selected model
        """
        # Skip if model has already been generated
        if self.model and self.model_name == model_name:
            print('Model already match.')
            return
        elif model_name in self.model_library:
            print("Model already in cache, setting that as primary.")
            self.model_name = model_name
            self.model = self.model_library[self.model_name]['model']
            self.transform = self.model_library[self.model_name]['transform']
            return
        else:
            print("Model not loaded yet.")
            self.model_name = model_name
        
        # Check if model can be loaded
        model_path, transform_paths = self.get_model_paths()
        is_model_loaded = self.load_model(model_path, transform_paths)
        
        # Choose a model if model not available to load from (then save it)
        if is_model_loaded is False: 
            self.create_model(self.model_name)
            self.save_model(model_path, transform_paths)
        
        # The model is ready for inference
        self.model.to(self.device)
        self.model.eval()
        print(f"model used: {self.model_name}")
        print(f"transform applied: {self.transform}")
        self.model_library[self.model_name] = {'model': self.model, 'transform': self.transform}
        return self.model_name, self.model, self.transform
        
    def get_model_paths(self):
        """Get file paths for saving/loading model weights and transforms.
        
        Returns:
            tuple: (model_path, transform_path) as Path objects
        """
        _save_dir = self.models_save_dir / self.model_name
        mkdir(_save_dir)
        model_path = _save_dir / f"{self.model_name}.pth"
        transform_path = _save_dir / f"{self.model_name}_transform.pkl"
        return model_path, transform_path
        
    def load_model(self, model_path, transform_path):
        """Load a previously saved model and its transforms from disk.
        
        Args:
            model_path (Path): Path to saved model weights (.pth file)
            transform_path (Path): Path to saved transforms (.pkl file)
        
        Returns:
            bool: True if successfully loaded, False if files don't exist
        """
        print('Loading Model ...')
        if model_path.is_file() and transform_path.is_file():
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = FmWrappers(model, self.model_name)
            with open(transform_path, "rb") as f:
                transform = pickle.load(f)
                self.transform = FmTransform(transform, self.model_name, self.device)
            print(f"\nModel loaded from {model_path}")
            return True
        else:
            return False
        
    def save_model(self, model_path, transform_path):
        """Save model weights and transforms to disk for future use.
        
        Args:
            model_path (Path): Destination path for model weights
            transform_path (Path): Destination path for transforms
        
        Returns:
            bool: True if successfully saved, False otherwise
        """
        print('Saving Model ...')
        if self.model is not None and self.model_name not in ['conch-1.5']:
            if not model_path.is_file():
                torch.save(self.model.original_model, model_path)
            else:
                print(f"{model_path} already exists, delete it to replace (resave) it")
            if not transform_path.is_file():
                with open(transform_path, "wb") as f:
                    pickle.dump(self.transform.transform_func, f)
            else:  
                print(f"{transform_path} already exists, delete it to replace (resave) it")
            print(f"Model save path: {model_path}")
            return True
        else:
            return False
        