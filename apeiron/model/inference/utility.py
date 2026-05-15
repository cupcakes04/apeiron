import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, r2_score, mean_squared_error, mean_absolute_error
from typing import Literal, List
from ..downstream import *
from dataclasses import dataclass, field
from apeiron.utils import save_and_show_plot


MODALITIES = ['label', 'annotation' , 'objects', 'text']
LOSS_MODALITIES = ['lbl_loss', 'ann_loss', 'obj_loss', 'gen_loss', 'con_loss']
INF_MODELS = ['abmil', 'gatmil', 'roimil', 'mask2f', 'classifier', 'unet', 'spconv', 'detr', 'generative', 'contrastive']
            
@dataclass
class HeadData:
    attention: np.ndarray | None = None      # (B, 1, N) or (B, C, N) attention scores
    lbl_logits: np.ndarray | None = None     # (B, C) slide-level class logits
    ann_logits: np.ndarray | None = None     # (B, N, C) tile-level annotation logits
    obj_classes: np.ndarray | None = None    # (B, Q, C+1) per-query class logits (includes 'no-object' class)
    obj_masks: np.ndarray | None = None      # (B, Q, N) per-query binary mask logits
    vis_emb: np.ndarray | None = None        # (B, H) pooled visual embeddings for text generation
    img_emb: np.ndarray | None = None        # (B, H) image embeddings for contrastive learning

@dataclass
class PredData:
    pred_atn: np.ndarray | None = None       # (B, 1, N) normalized attention weights
    pred_lbl: np.ndarray | None = None       # (B, C) probabilities/predictions for slide
    pred_ann: np.ndarray | None = None       # (B, N, C) probabilities/predictions for tiles
    pred_obj: list | None = None             # list (length B) of lists containing dicts: {"ids": list[int], "labels": np.ndarray(C,), "scores": float}
    pred_txt: list | None = None             # list[str] (length B) of generated text strings
    
    # Post-processed predictions
        # label (C)
        # annotation (N, C)
        # attention (N, C)
        # objects (Q, N, C)
        # scores(Q)
    post_processed: bool = False
    pred_crd: np.ndarray | None = None
    pred_scr: np.ndarray | None = None
    pred_data_type: str | None = None   # 'single' or 'group'

@dataclass
class Objectives:
    label: dict = field(default_factory=dict)
    annotation: dict = field(default_factory=dict)
    objects: dict = field(default_factory=dict)
    text: dict = field(default_factory=dict)

    def get_dict(self, **kwargs):
        return {
            'label': self.label,
            'annotation': self.annotation,
            'objects': self.objects,
            'text': self.text,
            **kwargs
        }

@dataclass
class ModelData:

    model_names: dict | None = None
    composite: dict = field(default_factory=lambda: {'final_loss': 0})

    head: HeadData = field(default_factory=HeadData)
    pred: PredData = field(default_factory=PredData)
    loss: Objectives = field(default_factory=Objectives)
    metric: Objectives = field(default_factory=Objectives)

    def assign(self, mode: Literal['head', 'pred', 'loss', 'metric'], **kwargs):
        mode_map = {
            'head': self.head,
            'pred': self.pred,
            'loss': self.loss,
            'metric': self.metric
        }
        obj = mode_map.get(mode)

        for key, value in kwargs.items():
            if not hasattr(obj, key):
                continue

            if isinstance(value, dict):
                target_obj_dict = getattr(obj, key)
                if len(target_obj_dict) == 0:
                    setattr(obj, key, value)
                else:
                    target_obj_dict.update(value)
            else:
                setattr(obj, key, value)

    def composite_loss(self, final_loss, **importance):
        self.composite = {
            'final_loss': final_loss,
            **importance,
        }


def choose_inferencer(
    in_features, mode=None, inf_model='abmil',
    lbl_n_classes=0, lbl_loss_type: str = 'hard_ce', lbl_cls_weights: dict = None, 
    ann_n_classes=0, ann_loss_type: str = 'bce', ann_cls_weights: dict = None):

    if inf_model == 'abmil':
        inferencer = ABMIL(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            embed_dim = 256,
            attn_dim = 128,
            dropout = 0.25,
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights, 
        )

    elif inf_model == 'gatmil':
        inferencer = GATMIL(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            embed_dim = 256,
            attn_dim = 128,
            k_neighbors = 32,
            num_heads = 4,
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights, 
        )

    elif inf_model == 'roimil':
        inferencer = ROIMIL(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            embed_dim = 256,
            dropout = 0.25,
            attn_dim = 128,
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights, 
            ann_loss_type = ann_loss_type, 
            ann_cls_weights = ann_cls_weights, 
        )
        
    elif inf_model == 'mask2f':
        inferencer = Mask2FormerMIL(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            embed_dim = 256,
            dropout = 0.25,
            attn_dim = 128,
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights, 
            ann_loss_type = ann_loss_type, 
            ann_cls_weights = ann_cls_weights, 
            num_heads = 4,
            num_layers = 2,
        )

    elif inf_model == 'classifier':
        inferencer = MLPClassifier(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            hidden_dim = 256,
            n_layers = 2,
            dropout = 0.25,
            mode = mode,
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights, 
            ann_loss_type = ann_loss_type, 
            ann_cls_weights = ann_cls_weights, 
        )

    elif inf_model == 'unet':
        inferencer = UNet(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            base_channels = 256,
            depth = 3,
            ann_loss_type = ann_loss_type, 
            ann_cls_weights = ann_cls_weights, 
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights,
        )

    elif inf_model == 'spconv':
        inferencer = SparseUNet(
            in_features = in_features,
            lbl_n_classes = lbl_n_classes,
            ann_n_classes = ann_n_classes,
            base_channels = 256,
            depth = 3,
            ann_loss_type = ann_loss_type, 
            ann_cls_weights = ann_cls_weights, 
            lbl_loss_type = lbl_loss_type, 
            lbl_cls_weights = lbl_cls_weights,
        )

    elif inf_model == 'detr':
        inferencer = SparseDETR(
            in_features = in_features,
            ann_n_classes = ann_n_classes,
            embed_dim = 256,
            n_heads = 8,
            n_encoder_layers = 3,
            n_decoder_layers = 3,
            n_queries = 20,
            dropout = 0.1,
            ann_cls_weights = ann_cls_weights,
            ann_loss_type = ann_loss_type,
        )

    elif inf_model == 'generative':
        inferencer = GenerativeVLM(
            in_features = in_features,
            lm_model_name = "distilgpt2",
            num_visual_tokens=32,
            use_lora=True,
            mode = mode,
        )

    elif inf_model == 'contrastive':
        inferencer = ContrastiveVLM(
            in_features=in_features,
            text_model_name = "emilyalsentzer/Bio_ClinicalBERT",
            num_visual_tokens=32,
            projection_dim=512,
            mode = mode,
        )

    return inferencer


def plot_analysis(y_true, y_pred, mode, title_prefix="", save_base=None, show=True):
    res = {}
    is_soft = np.any((y_true > 0) & (y_true < 1))
    
    # 1. Sanity Check Hard Confusion Matrix for all non-regression modes
    if mode != 'regression':
        # Usually true is one-hot (B, C) or indices (B,)
        y_true_cls = np.argmax(y_true, axis=1) if y_true.ndim > 1 and y_true.shape[1] > 1 else y_true.flatten()
        y_pred_cls = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 and y_pred.shape[1] > 1 else y_pred.flatten()
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_cls, y_pred_cls)
        res['confusion_matrix'] = cm
        
        # Normalize for coloring correctly by percentages
        cm_norm = confusion_matrix(y_true_cls, y_pred_cls, normalize='true')
        total_count = np.sum(cm)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        
        # Overwrite text to show Row Percentage \n Global Percentage
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_norm[i, j]
                pct_str = f"{pct:.1%}" if not np.isnan(pct) else "0.0%"
                global_pct = (count / total_count) * 100 if total_count > 0 else 0
                disp.text_[i, j].set_text(f"{pct_str}\n({global_pct:.2f}%)")
                
        title_text = f"{title_prefix} Hard Confusion Matrix\nTotal = {total_count}"
        ax.set_title(title_text)
        
        cfm_path = f"{save_base}_hard_cfm.png" if save_base else None
        save_and_show_plot(fig, cfm_path, show)
        
    # 2. Mode-specific graphs
    if mode == 'softmax':
        if is_soft:
            # Soft Confusion Matrix (fractional mass distribution)
            cm = y_true.T @ y_pred
            res['soft_confusion_matrix'] = cm
            
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums!=0)
            total_count = np.sum(cm)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
            disp.plot(ax=ax, cmap='Blues', colorbar=True)
            
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    count = cm[i, j]
                    pct = cm_norm[i, j]
                    pct_str = f"{pct:.1%}" if not np.isnan(pct) else "0.0%"
                    disp.text_[i, j].set_text(f"{pct_str}\n({count:.1f})")
                    
            title_text = f"{title_prefix} Soft Confusion Matrix\nTotal Mass = {total_count:.1f}"
            ax.set_title(title_text)
            
            cfm_path = f"{save_base}_soft_cfm.png" if save_base else None
            save_and_show_plot(fig, cfm_path, show)
            
            # Plot calibration scatter for soft labels in softmax
            num_classes = y_true.shape[1] if y_true.ndim > 1 else 1
            if num_classes > 1:
                fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))
                if num_classes == 1: axes = [axes]
                for c in range(num_classes):
                    ax = axes[c]
                    hb = ax.hexbin(y_true[:, c], y_pred[:, c], gridsize=40, cmap='viridis', bins='log', mincnt=1)
                    fig.colorbar(hb, ax=ax, label='log10(N)')
                    ax.plot([0, 1], [0, 1], 'r--', lw=2)
                    ax.set_xlabel('True Prob')
                    ax.set_ylabel('Pred Prob')
                    ax.set_title(f"Class {c}")
                    ax.grid(True, linestyle=':', alpha=0.6)
                fig.suptitle(f"{title_prefix} Softmax Calibration", y=1.05)
                scatter_path = f"{save_base}_scatter.png" if save_base else None
                save_and_show_plot(fig, scatter_path, show)

    elif mode == 'sigmoid':
        # y_true (B, C) binary, y_pred (B, C) probabilities
        num_classes = y_true.shape[1] if y_true.ndim > 1 else 1
        if num_classes == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
            
        # If not strictly soft, we can still compute PR curves
        if not is_soft:
            res['ap_scores'] = []
            fig, ax = plt.subplots(figsize=(8, 6))
            for c in range(num_classes):
                precision, recall, _ = precision_recall_curve(y_true[:, c], y_pred[:, c])
                ap = average_precision_score(y_true[:, c], y_pred[:, c])
                res['ap_scores'].append(ap)
                ax.plot(recall, precision, label=f'Class {c} (AP={ap:.2f})')
                
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f"{title_prefix} Precision-Recall Curve")
            ax.legend(loc='best', fontsize='small')
            ax.grid(True, linestyle=':', alpha=0.6)
            pr_path = f"{save_base}_pr.png" if save_base else None
            save_and_show_plot(fig, pr_path, show)
        
        # Plot calibration scatter for sigmoid
        fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))
        if num_classes == 1: axes = [axes]
        for c in range(num_classes):
            ax = axes[c]
            hb = ax.hexbin(y_true[:, c], y_pred[:, c], gridsize=40, cmap='viridis', bins='log', mincnt=1)
            fig.colorbar(hb, ax=ax, label='log10(N)')
            ax.plot([0, 1], [0, 1], 'r--', lw=2)
            ax.set_xlabel('True Prob')
            ax.set_ylabel('Pred Prob')
            ax.set_title(f"Class {c}")
            ax.grid(True, linestyle=':', alpha=0.6)
        fig.suptitle(f"{title_prefix} Sigmoid Calibration", y=1.05)
        scatter_path = f"{save_base}_scatter.png" if save_base else None
        save_and_show_plot(fig, scatter_path, show)
        
    elif mode == 'regression':
        # y_true (B, C), y_pred (B, C)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        res['mse'] = mse
        res['mae'] = mae
        res['r2'] = r2
        if show: print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        
        fig, ax = plt.subplots(figsize=(7, 5))
        hb = ax.hexbin(y_true.flatten(), y_pred.flatten(), gridsize=50, cmap='viridis', bins='log', mincnt=1)
        fig.colorbar(hb, ax=ax, label='log10(N)')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f"{title_prefix} Regression Scatter")
        ax.grid(True, linestyle=':', alpha=0.6)
        scatter_path = f"{save_base}_scatter.png" if save_base else None
        save_and_show_plot(fig, scatter_path, show)
        
    elif mode == 'rank':
        # y_true (B, C), y_pred (B, C)
        num_classes = y_true.shape[1] if y_true.ndim > 1 else 1
        if num_classes == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

        res['spearman'] = []
        
        # Rank Correlation Scatter
        fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 4))
        if num_classes == 1: axes = [axes]
        
        for c in range(num_classes):
            ax = axes[c]
            true_ranks = scipy.stats.rankdata(y_true[:, c])
            pred_ranks = scipy.stats.rankdata(y_pred[:, c])
            rho, _ = scipy.stats.spearmanr(y_true[:, c], y_pred[:, c])
            res['spearman'].append(rho)
            
            hb = ax.hexbin(true_ranks, pred_ranks, gridsize=40, cmap='viridis', bins='log', mincnt=1)
            fig.colorbar(hb, ax=ax, label='log10(N)')
            ax.plot([0, len(true_ranks)], [0, len(true_ranks)], 'r--', lw=2)
            ax.set_xlabel('True Rank')
            ax.set_ylabel('Predicted Rank')
            ax.set_title(f"Class {c} (rho={rho:.3f})")
            ax.grid(True, linestyle=':', alpha=0.6)
            
        fig.suptitle(f"{title_prefix} Rank Correlation", y=1.05)
        rank_path = f"{save_base}_rank.png" if save_base else None
        save_and_show_plot(fig, rank_path, show)
        
        # True-Sorted Monotonicity Plot
        fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 4))
        if num_classes == 1: axes = [axes]
        
        for c in range(num_classes):
            ax = axes[c]
            sort_idx = np.argsort(y_true[:, c])
            
            # Because Monotonicity plots can also be very dense:
            x_idx = np.arange(len(sort_idx))
            hb = ax.hexbin(x_idx, y_pred[sort_idx, c], gridsize=40, cmap='viridis', bins='log', mincnt=1)
            fig.colorbar(hb, ax=ax, label='log10(N)')
            
            ax.set_xlabel('Instances sorted by True Label')
            ax.set_ylabel('Predicted Score')
            ax.set_title(f"Class {c} Monotonicity")
            ax.grid(True, linestyle=':', alpha=0.6)
            
        fig.suptitle(f"{title_prefix} Monotonicity", y=1.05)
        mono_path = f"{save_base}_mono.png" if save_base else None
        save_and_show_plot(fig, mono_path, show)
        
    return res
