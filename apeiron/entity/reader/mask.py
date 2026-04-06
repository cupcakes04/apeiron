import numpy as np
import cv2
from matplotlib import pyplot as plt

def resolve_stride(stride, tile_size):
    """Convert stride specification to pixel step size.
    
    Args:
        stride (int, float, or None): Stride specification:
            - None: No overlap (step = tile_size)
            - float (0-1): Overlap ratio (e.g., 0.1 = 10% overlap)
            - int: Fixed pixel overlap
        tile_size (int): Size of tiles in pixels
    
    Returns:
        int: Step size in pixels between tile centers
    """
    if stride is None:
        return tile_size
    if isinstance(stride, float):  # Overlap ratio (e.g., 0.1)
        return round(tile_size * (1 - stride))
    if isinstance(stride, int):    # Fixed pixel overlap
        return round(tile_size - stride)
    return stride

def adjust_edge(val, wsi_dim, tile_size):
    """Adjust tile coordinate to prevent exceeding image boundaries.
    
    Args:
        val (int): Proposed tile start coordinate
        wsi_dim (int): Image dimension (width or height)
        tile_size (int): Tile size in pixels
    
    Returns:
        int: Adjusted coordinate ensuring tile fits within image
    """
    if val + tile_size > wsi_dim:
        return max(wsi_dim - tile_size, 0)
    return val

def mask_to_tile_coords(slide_w, slide_h, encoder, mask, tile_threshold=0.25, stride=None):
    """Generate tile coordinates from tissue mask for selective extraction.
    
    Only generates coordinates for tiles with sufficient tissue content,
    reducing computation by skipping background regions.
    
    Args:
        slide_w (int): Slide width in pixels at base resolution
        slide_h (int): Slide height in pixels at base resolution
        encoder (int): Tile size in pixels at base resolution (e.g., 224)
        mask (np.ndarray): Binary tissue mask (H, W) where 1=tissue, 0=background
        tile_threshold (float): Minimum tissue ratio (0-1) to accept tile. Default 0.25
        stride (int, float, or None): Tile overlap specification. See resolve_stride()

    Returns:
        tuple:
            - tile_coords (np.ndarray): (N, 2) top-left coordinates of valid tiles
            - scale (dict): Scaling factors {'x': ..., 'y': ...} from mask to WSI
            - step_size (dict): Tile size in mask space {'x': ..., 'y': ...}
    """
    tile_size = encoder
    stride_px = resolve_stride(stride, tile_size)
    tile_coords = []
    
    mask_h, mask_w = mask.shape
    scale = {'x': slide_w / mask_w, 'y': slide_h / mask_h}
    step_size = {'x': round(tile_size / scale['x']), 'y': round(tile_size / scale['y'])}

    for y in range(0, slide_h, stride_px):
        for x in range(0, slide_w, stride_px):
            
            # Extract tile from mask
            mask_y, mask_x = round(y / scale['y']), round(x / scale['x'])
            tile = mask[mask_y : mask_y + step_size['y'], mask_x : mask_x + step_size['x']]
            if tile.size <= 0:
                continue

            # Select tile if more than white_ratio % is in the white region
            white_ratio = np.sum(tile == 1) / tile.size
            if white_ratio >= tile_threshold:
                tile_coords.append([adjust_edge(x, slide_w, tile_size), 
                                    adjust_edge(y, slide_h, tile_size)])
                
            if x + tile_size >= slide_w: break
        if y + tile_size >= slide_h: break
        
    return np.array(tile_coords)

def full_mask_to_tile_coords(slide_w, slide_h, encoder, stride=None):
    """Generate tile coordinates covering the entire slide without masking.
    
    Creates a regular grid of tiles across the whole slide, useful when
    tissue detection is not needed or already performed.
    
    Args:
        slide_w (int): Slide width in pixels
        slide_h (int): Slide height in pixels
        encoder (int): Tile size in pixels
        stride (int, float, or None): Tile overlap specification
    
    Returns:
        np.ndarray: (N, 2) tile coordinates
    """
    tile_size = encoder
    stride_px = resolve_stride(stride, tile_size)
    tile_coords = []
    
    for y in range(0, slide_h, stride_px):
        for x in range(0, slide_w, stride_px):
            tile_coords.append([adjust_edge(x, slide_w, tile_size), 
                                adjust_edge(y, slide_h, tile_size)])
            if x + tile_size >= slide_w: break
        if y + tile_size >= slide_h: break
    return np.array(tile_coords)

def show_masked_tiles(mask, tile_coords, scale, step_size, alpha=0.5, colour=(0, 255, 255), show_img=False, tiled_mask=False):
    """Visualize tile coordinates overlaid on tissue mask for quality control.
    
    Creates a visualization showing which tiles will be extracted from the slide,
    useful for debugging tile generation and verifying tissue detection.
    
    Args:
        mask (np.ndarray): (H, W) binary tissue mask where 1=tissue, 0=background
        tile_coords (np.ndarray): (N, 2) tile top-left coordinates in WSI space
        scale (dict): Scaling factors {'x': ..., 'y': ...} from mask to WSI resolution
        step_size (dict): Tile size in mask space {'x': ..., 'y': ...}
        alpha (float): Overlay transparency (0=transparent, 1=opaque). Default 0.5
        colour (tuple): RGB color for tile rectangles. Default (0, 255, 255) yellow
        show_img (bool): If True, display image with matplotlib. Default False
        tiled_mask (bool): If True, show only tiles without background. Default False
    
    Returns:
        np.ndarray: RGB visualization with tiles overlaid on mask
    """
    if tiled_mask:
        mask_rgb = np.zeros_like(mask)
        
        for (x, y) in tile_coords:
            scaled_x = int(x / scale['x'])
            scaled_y = int(y / scale['y'])
            cv2.rectangle(mask_rgb, (scaled_x, scaled_y), (scaled_x + step_size['x'], scaled_y + step_size['y']), 255, -1)
    
    else:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        overlay = mask_rgb.copy()
        
        for (x, y) in tile_coords:
            scaled_x = int(x / scale['x'])
            scaled_y = int(y / scale['y'])
            cv2.rectangle(overlay, (scaled_x, scaled_y), (scaled_x + step_size['x'], scaled_y + step_size['y']), colour, -1)
        
        # Blend the colour (with alpha)
        cv2.addWeighted(overlay, alpha, mask_rgb, 1 - alpha, 0, mask_rgb)

    if show_img:
        plt.figure(figsize=(5, 5))
        plt.imshow(mask_rgb, cmap='gray' if tiled_mask else None)
        plt.show()

    return mask_rgb


def clean_tissue_mask(mask, min_area_threshold=50):
    """
    Locates all blobs in a binary mask and removes those smaller than threshold.
    
    Args:
        mask (np.array): Binary 16x mask (0=bg, 255=tissue)
        min_area_threshold (int): Min pixels for a blob to be kept.
    """
    # 1. Label every disconnected 'island' of tissue
    nb_blobs, im_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    # 2. Identify blobs to keep
    sizes = stats[:, cv2.CC_STAT_AREA]
    
    # 3. Create a new clean mask
    new_mask = np.zeros_like(mask)
    for i in range(1, nb_blobs):
        if sizes[i] >= min_area_threshold:
            # Re-fill only the blobs that passed the size filter
            new_mask[im_labels == i] = 255
            
    return new_mask