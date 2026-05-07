# The Visualiser class is strictly dependent on the current analyzer structure
# NOT STANDALONE USABLE
# Uses self instances for visualising tools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .helper.features import *
from apeiron.model.inference import ModelData
from typing import Literal, Callable
import seaborn as sns
from dataclasses import dataclass

@dataclass
class VizData:
    '''
    - draw_coords is L,N,2 (x,y)
    - draw_colors is L,N,3 (RGB)
    - color_map is (C, 3) or None when using heatmap mode.
    '''
    # L = Length, N = Number of points
    draw_coords: np.ndarray
    draw_colors: np.ndarray
    color_map: tuple | list | None = None
    show: Callable | None = None
    hist: Callable | None = None
    

class Visualiser:
    """Creates feature-overlay visualizations on slide thumbnails.

    Paints per-tile colors (from PCA/UMAP reduction, clustering, similarity
    scoring, or annotation labels) onto a canvas that matches the slide
    thumbnail dimensions, then alpha-blends the result for display.

    This class is mixed into the Processor via multiple inheritance and
    expects the following attributes to be set by sibling classes:

    - ``slide_thumbnail`` (np.ndarray): RGB thumbnail from Thumbnailer.
    - ``slide_thumbnail_mpp`` (float): Resolution of the thumbnail.
    - ``base_mpp`` (float): Slide base resolution.
    - ``feats_size`` (int): Spatial size of each feature window.
    - ``feats_color``, ``feats_score``, ``feats_clusters``, ``feats_color_map``:
      Feature arrays from Processor.
    - ``coords`` (np.ndarray): (N, 2+) feature coordinates.

    Args:
        **kwargs: Forwarded to parent classes.

    Attributes:
        overlay (np.ndarray): (H, W, 3) RGB overlay image in [0, 255] range.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.draw_coords: np.ndarray = None
        self.draw_colors: np.ndarray = None
        self.color_map: dict = None
        self.overlay: np.ndarray = None

        # Attached dataclasses
        self.proc_ext: ProcessedExt = None
        self.mdata: ModelData = None

    @property
    def overlay_tools(self):
        return {
            'draw_colors': self.draw_colors, 'draw_coords': self.draw_coords, 'color_map': self.color_map,
            'slide_thumbnail': self.slide_thumbnail, 'slide_thumbnail_mpp': self.slide_thumbnail_mpp, 
            'base_mpp': self.base_mpp, 'feats_size': self.proc_ext.feats_size
        }

    # |-----------------------------------------------|
    # |------------ Create Viz Dataclass -------------|
    # |-----------------------------------------------|


    def create_feature_viz(self, mode: Literal['color', 'clusters', 'score'] = 'color'):
        """Create RGB overlay image from features for visualization.
        
        Paints feature colors onto a canvas matching thumbnail dimensions,
        with each feature window colored according to the specified mode.
        
        Args:
            mode (str): Visualization mode:
                - 'color': Use PCA/UMAP reduced RGB features
                - 'clusters': Use cluster assignment colors
                - 'score': Use similarity score heatmap (coolwarm)
        
        Returns:
            np.ndarray: (H, W, 3) RGB overlay image in [0, 255] range
        """
        self.draw_coords = self.proc_ext.coords

        # We use feats_color_map (RGB 0-1) and feats_clusters (indices)
        if mode == 'color': 
            colors = self.proc_ext.feats_color
        elif mode == 'clusters': 
            colors = self.proc_ext.feats_color_map[self.proc_ext.feats_clusters]
        elif mode == 'score':
            colors = feats_score_to_coolwarm(normalise(self.proc_ext.feats_score, mode='minmax'))
        
        self.draw_colors = np.expand_dims(colors, axis=0)
        self.draw_coords = get_grid_coords(self.draw_coords)
        viz_overlay = lambda **k: self.visualise_overlay(overlay_tools=self.overlay_tools, **k)
        viz_histogram = lambda **k: self.visualise_histogram(hist=hist, **k)

        return VizData(
            draw_coords=self.draw_coords, draw_colors=self.draw_colors,
            show=viz_overlay, hist=viz_histogram)

    def create_annotation_viz(self,
        mode: Literal['annotation', 'pred_lbl', 'pred_ann', 'pred_atn', 'pred_obj'], 
        heatmap_target: int|list = None, n_obj: int = None,
        ):

        """Create an RGB overlay from annotation class fractions.

        Args:
            - coord (np.ndarray): (N, 2+) tile coordinates.
            - annotation (np.ndarray): (N, C) class fraction matrix.
            - heatmap_target (int, list, np.ndarray, 'all', optional): Target classes.
            - n_obj (int, optional): Number of top objects to visualize (for pred_obj).

        Returns:
            - draw_coords is L,N,2 (x,y)
            - draw_colors is L,N,3 (RGB)
            - color_map is (C, 3) or None when using heatmap mode.
        """
        # 1. Prepare `ann` which is (N, C)
        if mode in ['annotation', 'pred_lbl', 'pred_ann', 'pred_atn']:
            if mode == 'annotation': 
                ann = self.annotation
                self.draw_coords = self.proc_ext.coords
            elif mode == 'pred_lbl' and self.mdata.pred.pred_data_type == 'group':
                ann = self.mdata.pred.pred_lbl
                self.draw_coords = get_grid_coords(self.mdata.pred.pred_crd)
            elif mode == 'pred_ann':
                ann = self.mdata.pred.pred_ann
                self.draw_coords = self.mdata.pred.pred_crd
            elif mode == 'pred_atn':
                ann = normalise(self.mdata.pred.pred_atn, mode='minmax')
                self.draw_coords = self.mdata.pred.pred_crd
                heatmap_target = 0  # no multi-class attention
            
            if ann is None: 
                return
            
            # Segment classes or focus on a class
            if heatmap_target is not None:
                if isinstance(heatmap_target, str) and heatmap_target == 'all':
                    # Average over all classes or max
                    heat = np.max(ann, axis=1)
                elif isinstance(heatmap_target, (list, tuple, np.ndarray)):
                    # Max over specified classes
                    heat = np.max(ann[:, heatmap_target], axis=1)
                else:
                    # Single integer class
                    heat = ann[:, heatmap_target]
                
                colors = feats_score_to_coolwarm(heat)
                self.draw_colors = np.expand_dims(colors, axis=0) # (1, N, 3)
                self.color_map = None
            else:
                colors, self.color_map = ann_percentages_to_rgb(ann)
                self.draw_colors = np.expand_dims(colors, axis=0) # (1, N, 3)
            
            hist = ann

        # 2. Prepare `obj` which is (N, C)
        elif mode in ['pred_obj']:

            # Sort objects by score descending
            obj_list = sorted(self.mdata.pred.pred_obj, key=lambda x: x['scores'], reverse=True)

            # default top 1
            if n_obj:
                obj_list = obj_list[:n_obj]
            else:
                obj_list = [obj_list[0]]

            self.draw_colors, self.color_map = obj_to_rgb(obj_list, self.draw_coords)
            hist = obj_to_NC(obj_list)
        
        viz_overlay = lambda **k: self.visualise_overlay(overlay_tools=self.overlay_tools, **k)
        viz_histogram = lambda **k: self.visualise_histogram(hist=hist, **k)
        
        return VizData(
            draw_coords=self.draw_coords, draw_colors=self.draw_colors, color_map=self.color_map, 
            show=viz_overlay, hist=viz_histogram)


    # |-----------------------------------------------|
    # |--------------- Create Overlays ---------------|
    # |-----------------------------------------------|


    @staticmethod
    def draw_on_thumbnail(
        draw_colors, draw_coords, color_map,
        slide_thumbnail, slide_thumbnail_mpp, 
        base_mpp, feats_size):
        """Paint per-tile colors onto a blank overlay matching thumbnail size.

        Each tile is drawn as a filled rectangle at its scaled position.
        The result is stored in ``self.overlay``.
        """
        # 1. Setup dimensions and scale
        thumb_h, thumb_w = slide_thumbnail.shape[:2]
        down_scale = base_mpp / slide_thumbnail_mpp
        tile_size_scaled = int(round(feats_size * down_scale))
        
        # 2. Initialize an empty overlay (zeros)
        # We use 3 channels for RGB
        overlay = np.zeros((thumb_h, thumb_w, 3), dtype=np.float32)
        
        # Determine effective coords and colors
        # Flatten batch dimension, taking max color
        # Because objects might overlap, maxing the RGB ensures we see overlapping objects
        colors_flat = np.max(draw_colors, axis=0)
            
        for i, (x, y) in enumerate(draw_coords[:, :2]):
            # Scale coordinates to thumbnail space
            y_s, x_s = int(round(y * down_scale)), int(round(x * down_scale))
            
            # Determine bounds (with clipping to prevent index errors)
            y_end = min(y_s + tile_size_scaled, thumb_h)
            x_end = min(x_s + tile_size_scaled, thumb_w)
            
            # Paint the patch
            overlay[y_s:y_end, x_s:x_end] = colors_flat[i] * 255

        # 3. Create Legend Handles
        # We iterate through your color map to build the labels
        if color_map is None:
            return overlay, []

        legend_handles = []
        for i, color in enumerate(color_map['color']):
            # Skip background (usually index 0) if you don't want it in the legend
            if i == 0 and np.all(color == 0):
                continue
            patch = mpatches.Patch(color=color, label=color_map['class'][i])
            legend_handles.append(patch)
        return overlay, legend_handles

    def visualise_overlay(self, overlay_tools=None, alpha=0.5):
        """Blend feature overlay with slide thumbnail and display.
        
        Args:
            alpha (float): Overlay opacity (0=transparent, 1=opaque). Default 0.5
        
        Returns:
            np.ndarray: (H, W, 3) blended image
        """
        tools = overlay_tools if overlay_tools else self.overlay_tools
        self.overlay, legend_handles = self.draw_on_thumbnail(**tools)

        # 1. Blend overlay with thumbnail using alpha compositing
        # Only blend regions where features exist (non-zero overlay)
        mask = np.sum(self.overlay, axis=2) > 0
        blended = self.slide_thumbnail.copy()
        blended[mask] = (1 - alpha) * self.slide_thumbnail[mask] + alpha * self.overlay[mask]

        # 2. Display
        plt.figure(figsize=(15, 12))
        plt.imshow(blended)
        plt.axis('off')

        # 3. Add Legend to plot
        if len(legend_handles) > 0:
            plt.legend(
                handles=legend_handles, 
                bbox_to_anchor=(1.05, 1), 
                loc='upper left', 
                borderaxespad=0.,
                fontsize=12,
                title="Gleason Classes"
            )
            plt.tight_layout() # Ensures legend isn't cut off
        plt.show()
        return blended

    @staticmethod
    def visualise_histogram(hist, bins=30, cols_per_row=4):
        """
        Visualizes the distribution of each feature C across N samples.
        
        Args:
            hist: numpy array or torch tensor of shape (N, C)
            bins: number of bins for the histogram
            cols_per_row: how many subplots to show per row
        """
        # Convert to numpy if it's a torch tensor
        if hasattr(hist, 'detach'):
            hist = hist.detach().cpu().numpy()
        
        N, C = hist.shape
        rows = (C + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 4, rows * 3))
        axes = axes.flatten()
        
        for i in range(C):
            sns.histplot(hist[:, i], bins=bins, kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Feature C_{i}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        fig.suptitle(f"Distribution of N: {N}", fontsize=16)
        plt.tight_layout()
        plt.show()

