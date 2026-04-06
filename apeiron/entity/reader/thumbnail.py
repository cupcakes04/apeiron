from tiatoolbox.wsicore.wsireader import WSIReader
import matplotlib.pyplot as plt
from apeiron.utils import save_img, load_img
import numpy as np
from tiatoolbox.data import stain_norm_target
from tiatoolbox.tools.stainnorm import get_normalizer
from numpy.linalg import LinAlgError
from typing import Literal
from pathlib import Path

class Thumbnailer:
    """Generates and manages various thumbnail types for whole slide images.
    
    Creates RGB thumbnails, tissue masks, and stain-normalized thumbnails at
    specified resolutions for visualization and tissue detection.
    
    Args:
        slide_path (str or Path, optional): Path to slide file
        default_tb_mpp (float): Default microns per pixel for thumbnails. Default 8.0
        **kwargs: Additional keyword arguments passed to parent classes
    
    Attributes:
        slide_thumbnail (np.ndarray): RGB thumbnail image
        slide_thumbnail_mpp (float): Resolution of slide_thumbnail
        masked_thumbnail (np.ndarray): Binary tissue mask
        masked_thumbnail_mpp (float): Resolution of masked_thumbnail
        normalised_thumbnail (np.ndarray): Stain-normalized RGB thumbnail
        normalised_thumbnail_mpp (float): Resolution of normalised_thumbnail
        slide_obj (WSIReader): TIAToolbox slide reader
        default_tb_mpp (float): Default thumbnail resolution
    """
    def __init__(self, slide_path=None, default_tb_mpp=8.0, **kwargs):
        super().__init__(**kwargs)
        if slide_path:
            self.slide_path = slide_path
        
        self.slide_thumbnail = None
        self.slide_thumbnail_mpp = None
        self.masked_thumbnail = None
        self.masked_thumbnail_mpp = None
        self.slide_obj: WSIReader
        self.base_mpp: float
        self.default_tb_mpp = default_tb_mpp
        self.load_norm_configs()

    @staticmethod
    def _verify_tb_img_path(img_path):
        return img_path and Path(img_path).is_file()
        
    def serve_thumbnail(self, mode: Literal['slide_thumbnail', 'masked_thumbnail'], 
                        img_path=None, target_mpp=None, method=None):
        """Load or generate a thumbnail of the specified type.

        If ``img_path`` points to an existing file it is loaded from disk;
        otherwise the thumbnail is generated from the slide object.

        Args:
            mode ('slide_thumbnail' | 'masked_thumbnail'): Thumbnail type.
            img_path (str or Path, optional): Pre-existing thumbnail image to load.
            target_mpp (float, optional): Desired microns per pixel. Uses default if None.
            method (str, optional): Tissue masking method (only for masked_thumbnail).
                Options: ``'morphological'`` (default), ``'otsu'``.

        Returns:
            np.ndarray: The requested thumbnail array.
        """

        if mode == 'slide_thumbnail':
            if self._verify_tb_img_path(img_path):
                self.slide_thumbnail = load_img(img_path)
                self.slide_thumbnail_mpp = target_mpp
            else:
                self.get_slide_thumbnail(target_mpp=target_mpp)

            return self.slide_thumbnail

        elif mode == 'masked_thumbnail':
            if self._verify_tb_img_path(img_path):
                self.masked_thumbnail = load_img(img_path)
                self.masked_thumbnail_mpp = target_mpp
            else:
                self.get_masked_thumbnail(target_mpp=target_mpp, method=method)

            return self.masked_thumbnail
    
    # (1)
    def get_slide_thumbnail(self, target_mpp=None):
        """
        Generate a slide_thumbnail np.array of (H,W,3)
        Args:
            target_mpp: The desired microns per pixel (e.g., 8.0 for low res)
        """
        if target_mpp is None: 
            target_mpp = self.default_tb_mpp
            
        slide_thumbnail = self.slide_obj.slide_thumbnail(resolution=target_mpp, units='mpp')
        self.slide_thumbnail = slide_thumbnail
        self.slide_thumbnail_mpp = target_mpp
        return slide_thumbnail
    
    # (2)
    def get_masked_thumbnail(self, target_mpp=None, method="morphological"):
        """
        Generate a masked_thumbnail np.array of (H,W), binary
        Args:
            target_mpp: The desired microns per pixel (e.g., 8.0 for low res)
        """
            
        if target_mpp is None: 
            target_mpp = self.default_tb_mpp
            
        # Methods for tissue masking (binary masks)
        if method == "morphological":
            mask_reader = self.slide_obj.tissue_mask(method=method, resolution=target_mpp, units='mpp', min_region_size=0)
            masked_thumbnail = mask_reader.slide_thumbnail(resolution=target_mpp, units='mpp')
        elif method == 'otsu':
            mask_reader = self.slide_obj.tissue_mask(method=method, resolution=target_mpp, units='mpp')
            masked_thumbnail = mask_reader.slide_thumbnail(resolution=target_mpp, units='mpp')
            
        self.masked_thumbnail = masked_thumbnail
        self.masked_thumbnail_mpp = target_mpp
        return masked_thumbnail
    

    ## -------- Slide Normalisation (TBC) -------- ##
    
    def normalise_image(self, output_image=None, norm_configs=None):
        """Apply stain normalization to an image.

        Args:
            output_image (np.ndarray): RGB image array (H, W, 3) to normalize.
            norm_configs (dict, optional): Normalization settings. If provided,
                reloads the normalizer with these configs before transforming.

        Returns:
            np.ndarray: Stain-normalized image (H, W, 3).
        """
        if norm_configs:
            self.load_norm_configs(norm_configs)
        return self.stain_normalizer.transform(output_image)

    def load_norm_configs(self, norm_configs={}):
        """Initialize the stain normalizer from configuration.

        Args:
            norm_configs (dict): Normalization configuration with optional keys:
                - ``method`` (str): Normalization method — ``'macenko'`` (default),
                  ``'reinhard'``, ``'ruifrok'``, or ``'vahadane'``.
                - ``target_img_path`` (str): Direct path to the target image.
                - ``target_path`` (str): Directory containing target images.
                - ``target_name`` (str): Filename of the target image.
        """
        method = norm_configs.get('method', 'macenko')

        target_img_path = norm_configs.get('target_img_path')
        if target_img_path is None:
            target_img_path = f"{norm_configs.get('target_path', '')}/{norm_configs.get('target_name', '')}"

        target_image = load_img(target_img_path)
        if target_image is None:
            target_image = stain_norm_target()
        
        _, self.stain_normalizer = self.create_normalizer(target_image, method)

    @staticmethod
    def create_normalizer(target_image, method="macenko"):
        """
        vahadane uses more computational power & may crash if data is too noisy (data)

        Args:
            method (str, optional): "reinhard", "ruifrok", "macenko", "vahadane". Defaults to "macenko".
            target_image (ndarray, optional): a tile as the standard for normalisation. Defaults to stain_norm_target().
        Raises:
            e: when too many patches fails to normalise, it errors
        Returns:
            tuple:
                - Callable: can do in processing capabilitites
                - object: `stain_normalizer` can do `.transform`
        """
        stain_normalizer = get_normalizer(method)
        if isinstance(target_image, (str, Path)):
            target_image = load_img(target_image)
        stain_normalizer.fit(target_image)
        
        def stain_norm_func(img: np.ndarray) -> np.ndarray:
            """Helper function to perform stain normalization."""

            # Check if the input image is valid (at least 2D and non-empty)
            if img.ndim < 2 or img.size == 0:
                print(f"Skipping invalid tile.")
                return img  # Return the original image (or handle it differently if needed)
            try:
                return stain_normalizer.transform(img)
            except (ValueError, LinAlgError) as e:
                if "Empty tissue mask computed" in str(e) or isinstance(e, LinAlgError):
                    print(f"Skipping empty/invalid tile.")
                    return img  # Return the original image
                else:
                    raise e  # Reraise other errors
        
        return stain_norm_func, stain_normalizer
    
    # TODO: implement QC
    # def tissue_qc()
    #     # Quality control segmentation
    #     # Run in timm 0.4 with kwargs
    #     if units == 'mpp': read_mpp = str(resolution)
    #     output_timm04 = run_script_in_env(env_name="nntf", script_path=Path(__file__).parent/ "segmentation_models/grandqc/script.py", 
    #                                 wsi_path=str(wsi_path),
    #                                 read_mpp=read_mpp)
    #     if output_timm04:
    #         mask_thumbnail, class_img = process_grandqc_json(output_timm04=output_timm04, target_colors=[0, 3, 4, 7])
    #         mask_size = wsi.slide_dimensions(resolution=resolution, units=units)
    #         mask_thumbnail = cv2.resize(mask_thumbnail, (mask_size[0], mask_size[1]), interpolation=cv2.INTER_NEAREST)
    #     else:
    #         print("method:", method, "cannot find tissue, method changed to [morphological] for: ", Path(wsi_path).name)
    #         method = "morphological"
                
                
                
                
                