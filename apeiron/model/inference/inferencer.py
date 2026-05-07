from ..downstream import *
from .composite import *
from .utility import ModelData
from .optimizer import Optimizer
import torch
from typing import Literal, List
import numpy as np
from pathlib import Path
from apeiron.utils import get_device, extend_dict
from tqdm import tqdm
from apeiron.utils import to_cpu
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, r2_score, mean_squared_error, mean_absolute_error

class Inferencer:
    def __init__(self, device: str = None, **kwargs):
        super().__init__(**kwargs)
        self.device = get_device(device, 'inferencer')
        self.inferencer: MultiHeadModel = None

        self.optim_cfgs: dict = {}
        self.optimizer: Optimizer = None
        self.optimizer_state_dict: dict = None
        
        self.epoch_losses: dict = {}
        self.epoch_metrics: dict = {}
        self.epoch_xy: dict = {'true_ann': [], 'true_lbl': [], 'pred_ann': [], 'pred_lbl': []}


    # |-----------------------------------------------|
    # |------------ Set Up & Initialise --------------|
    # |-----------------------------------------------|


    def setup_inferencer(self, 
        in_features, mode=None, inf_models = 'abmil', 
        lbl_n_classes=None, lbl_loss_type='bce', lbl_cls_weights: dict = None,
        ann_n_classes=None, ann_loss_type='bce', ann_cls_weights: dict = None,
        lr=1e-4, optimizer='adam', weight_decay: float = 0.0, scheduler: str = None,
        chkp_path=None, **kwargs
    ):
        """
        Initializes the downstream model architectures, loss functions, and predictors.
        
        Args:
            in_features (int): Dimensionality of the input features.
            mode (str): Execution mode ('slide' or 'tile').
            lbl_n_classes (int): Number of label classes.
            ann_n_classes (int): Number of annotation classes.
            inf_models (str or list): Name of the downstream model(s) to use (e.g., 'abmil', 'clam').
            lbl_loss_type (str): Type of loss for label prediction.
            ann_loss_type (str): Type of loss for annotation segmentation.
            lbl_cls_weights (dict): Class weights for label loss.
            ann_cls_weights (dict): Class weights for annotation loss.
            lr (float): Learning rate.
            optimizer (str): Optimizer algorithm.
            weight_decay (float): Weight decay for optimizer.
            scheduler (str): Learning rate scheduler type.
            chkp_path (str): Path to a checkpoint to load model weights from.
        """

        # 1. Prepare multimodal heads
        self.inferencer = MultiHeadModel(
            in_features, mode=mode, inf_models=inf_models,
            lbl_n_classes=lbl_n_classes, lbl_loss_type=lbl_loss_type, lbl_cls_weights=lbl_cls_weights,
            ann_n_classes=ann_n_classes, ann_loss_type=ann_loss_type, ann_cls_weights=ann_cls_weights,
        ).to(self.device)

        # 3. Prepare Optimizers
        self.optim_cfgs = {'lr': lr, 'optimizer': optimizer, 'weight_decay': weight_decay, 'scheduler': scheduler}
        self.optimizer_state_dict = None

        # Load model if available
        if chkp_path and Path(chkp_path).is_file():
            self.load_inferencer(chkp_path)


    def setup_optimizer(self, **kwargs):
        if len(self.optim_cfgs) > 0:
            kwargs = extend_dict(kwargs, **self.optim_cfgs)

        # Get IDs of parameters in the losser
        losser_params = set(self.inferencer.losser.parameters())

        # Filter the main parameters to exclude those in losser
        main_params = [p for p in self.inferencer.parameters() if p not in losser_params]

        param_groups = [
            {'params': main_params}, # Main model minus losser
            {'params': list(losser_params), 'lr': 1e-3} # Just losser
        ]

        self.optimizer = Optimizer(params=param_groups, **kwargs)
        if self.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(self.optimizer_state_dict)

    def get_contrastive_embeddings(self, *args, **kwargs):
        return self.inferencer.get_contrastive_embeddings(*args, **kwargs)


    # |-----------------------------------------------|
    # |-------------- Train & Evaluate ---------------|
    # |-----------------------------------------------|


    def prep_data(self, data: dict, mode: Literal['tocpu', 'torch']):
        for k, v in data.items():
            if v is None: 
                continue
            elif k in ['coords', 'features', 'annotation', 'label'] and len(v) > 0:
                if mode == 'torch':
                    data[k] = torch.as_tensor(v).to(self.device)
                elif mode == 'tocpu':
                    data[k] = to_cpu(v)

    @torch.no_grad()
    def predict_data(self, data: dict, threshold=0.5, run_metric=True):
        """Forward pass + prediction (single sample, no grad)."""

        self.prep_data(data, mode='torch')
        mdata = self.inferencer(**data)
        self.inferencer.predict(mdata)
        if run_metric: self.inferencer.metric(mdata, threshold=threshold, **data)

        data.pop('features', None)
        self.prep_data(data, mode='tocpu')
        return {'mdata': mdata, 'data': data}


    def train_epoch(self, data_collector):
        """Run one training epoch over a data generator.

        Since each slide may have a different N (number of tiles), we process
        one sample at a time but accumulate gradients over ``batch_size``
        samples before performing an optimizer step.

        Args:
            data_collector: Iterable yielding data dicts (from Collector).
            batch_size (int): Number of samples to accumulate gradients over
                before stepping. Default ``1``.

        Returns:
            dict: ``epoch_loss`` (float) — mean loss over all samples,
                plus per-component mean losses.
        """
        self.inferencer.train()
        self.epoch_losses = {}
        self.epoch_metrics = {}

        n_samples = 0
        self.optimizer.zero_grad()
        accumulated_loss = 0.0

        for data in tqdm(data_collector):
            # 1. Compute loss (this now processes a whole batch)
            self.prep_data(data, mode='torch')
            mdata: ModelData = self.inferencer(**data)
            self.inferencer.loss(mdata, **data)
            final_loss = mdata.composite['final_loss']

            accumulated_loss = accumulated_loss + final_loss

            # 2. Backprop and Step immediately
            if data.get('propagate_loss', True):
                accumulated_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.inferencer.clear_cache()
                accumulated_loss = 0.0

            # 3. Accumulate loss tracking (using batch averages)
            self.cum_epoch(mdata.loss.get_dict(composite=mdata.composite), self.epoch_losses)

            # 4. Compute metrics (no grad — detached from the graph)
            with torch.no_grad():
                self.inferencer.predict(mdata)
                self.inferencer.metric(mdata, threshold=0.5, **data)
                self.cum_epoch(mdata.metric.get_dict(), self.epoch_metrics)
            n_samples += 1

        # Average losses and metrics across all batches seen
        self.avg_epoch(n_samples, self.epoch_losses)
        self.avg_epoch(n_samples, self.epoch_metrics)
        return {'loss': self.epoch_losses, 'metric': self.epoch_metrics}

    @torch.no_grad()
    def eval_epoch(self, data_collector, threshold=0.5):
        """Run one evaluation epoch (no gradient, no optimizer step).

        Args:
            data_collector: Iterable yielding data dicts.

        Returns:
            tuple: (epoch_losses, epoch_metrics, predictions)
                - ``epoch_losses``  (dict): Mean losses over all samples.
                - ``epoch_metrics`` (dict): Mean metrics over all samples.
                - ``predictions``   (list[dict]): Per-sample prediction dicts.
        """
        self.inferencer.eval()

        self.epoch_losses = {}
        self.epoch_metrics = {}

        n_samples = 0
        for data in tqdm(data_collector):
            self.prep_data(data, mode='torch')
            mdata: ModelData = self.inferencer(**data)
            self.inferencer.loss(mdata, **data)
            self.inferencer.predict(mdata)
            self.inferencer.metric(mdata, threshold=threshold, **data)
            
            # For `val_graphs` to use without re-running
            record_lbl, record_ann = False, False
            if 'label' in data: 
                self.epoch_xy['true_lbl'].append(to_cpu(data['label']))
                record_lbl = True
            if 'annotation' in data:
                self.epoch_xy['true_ann'].append(to_cpu(data['annotation']))
                record_ann = True
                
            for _, head_mod in self.inferencer.heads.items():
                res = head_mod.result
                if 'pred_lbl' in res and record_lbl: self.epoch_xy['pred_lbl'].append(res['pred_lbl'])
                if 'pred_ann' in res and record_ann: self.epoch_xy['pred_ann'].append(res['pred_ann'])
            
            self.cum_epoch(mdata.loss.get_dict(composite=mdata.composite), self.epoch_losses)
            self.cum_epoch(mdata.metric.get_dict(), self.epoch_metrics)
            n_samples += 1

        self.avg_epoch(n_samples, self.epoch_losses)
        self.avg_epoch(n_samples, self.epoch_metrics)
        return {'loss': self.epoch_losses, 'metric': self.epoch_metrics}


    def cum_epoch(self, loss_dict, epoch_dict):
        for modality, res in loss_dict.items():
            epoch_dict.setdefault(modality, {})
            for ls_type, val in res.items():
                if hasattr(val, "item"): val = val.item()
                epoch_dict[modality][ls_type] = epoch_dict[modality].get(ls_type, 0.0) + val

    def avg_epoch(self, n_samples, epoch_dict):
        if n_samples == 0:
            return
        for name, res in epoch_dict.items():
            for key, loss in res.items():
                epoch_dict[name][key] = loss / n_samples

    @torch.no_grad()
    def val_graphs(self):
        """Compute confusion matrix, PR curves, etc. over validation data.
        
        Uses cached valid_history if evaluation has already been run.
        Otherwise, runs evaluation over data_collector.

        Returns:
            dict: Numerical matrices and curves data.
        """
        # 1. Reconstruct from the last evaluated epoch in valid_history if we stored predictions
        all_true, all_pred = {}, {}
        all_true['label'] = self.epoch_xy.get('true_lbl', [])
        all_pred['label'] = self.epoch_xy.get('pred_lbl', [])
        all_true['annotation'] = self.epoch_xy.get('true_ann', [])
        all_pred['annotation'] = self.epoch_xy.get('pred_ann', [])

        # 2. Analyze Label Modality
        analysis_results = {}
        if all_true['label'] and all_pred['label']:
            y_true = np.concatenate(all_true['label'], axis=0) # (B, C) or (B,)
            y_pred = np.concatenate(all_pred['label'], axis=0) # (B, C) or (B,)
            
            # Determine mode from the first head that handles labels
            mode = 'softmax' # Default fallback
            for head_mod in self.inferencer.heads.values():
                if hasattr(head_mod, 'lbl_mode'):
                    mode = head_mod.lbl_mode
                    break
                    
            print(f"\n--- Label Analysis ({mode}) ---")
            analysis_results['label'] = self._plot_analysis(y_true, y_pred, mode, title_prefix="Label")

        # 3. Analyze Annotation Modality
        if all_true['annotation'] and all_pred['annotation']:
            # Flatten spatial/batch dimensions for each array to (-1, C) before concat to handle varying N
            # (B,N,C) -> (B*N,C)
            y_true = np.concatenate([y.reshape(-1, y.shape[-1]) for y in all_true['annotation']], axis=0)
            y_pred = np.concatenate([y.reshape(-1, y.shape[-1]) for y in all_pred['annotation']], axis=0)
            
            mode = 'softmax' # Default fallback
            for head_mod in self.inferencer.heads.values():
                if hasattr(head_mod, 'ann_mode'):
                    mode = head_mod.ann_mode
                    break
                    
            print(f"\n--- Annotation Analysis ({mode}) ---")
            analysis_results['annotation'] = self._plot_analysis(y_true, y_pred, mode, title_prefix="Annotation")

        return analysis_results

    def _plot_analysis(self, y_true, y_pred, mode, title_prefix=""):
        res = {}
        
        # Handle formats based on pred.py modes
        if mode == 'softmax':
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
            
            # Overwrite text to show Row Percentage \n (Global Percentage)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    count = cm[i, j]
                    pct = cm_norm[i, j]
                    pct_str = f"{pct:.1%}" if not np.isnan(pct) else "0.0%"
                    global_pct = (count / total_count) * 100 if total_count > 0 else 0
                    disp.text_[i, j].set_text(f"{pct_str}\n({global_pct:.2f}%)")
                    
            ax.set_title(f"{title_prefix} Confusion Matrix\nTotal = {total_count}")
            plt.tight_layout()
            plt.show()

        elif mode == 'sigmoid':
            # y_true (B, C) binary, y_pred (B, C) probabilities
            num_classes = y_true.shape[1] if y_true.ndim > 1 else 1
            if num_classes == 1:
                y_true = y_true.reshape(-1, 1)
                y_pred = y_pred.reshape(-1, 1)
                
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
            plt.tight_layout()
            plt.show()
            
        elif mode == 'regression':
            # y_true (B, C), y_pred (B, C)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            res['mse'] = mse
            res['mae'] = mae
            res['r2'] = r2
            print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
            
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5)
            ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.set_title(f"{title_prefix} Regression Scatter")
            ax.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.show()
            
        return res

    # |-----------------------------------------------|
    # |---------------- Save & Load ------------------|
    # |-----------------------------------------------|


    def save_inferencer(self, chkp_path, epoch=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.inferencer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, chkp_path)

    def load_inferencer(self, chkp_path):

        # 1. Load the checkpoint file from disk
        checkpoint = torch.load(chkp_path, map_location=self.device)

        # 2. Load the weights into the objects
        self.inferencer.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer_state_dict = checkpoint['optimizer_state_dict']
        print('Loaded epoch:', checkpoint['epoch'])

        # 3. Move to device
        self.inferencer.to(self.device)