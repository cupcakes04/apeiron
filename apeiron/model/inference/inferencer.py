from ..downstream import *
from .composite import *
from .optimizer import Optimizer
import torch
from typing import Literal, List
import numpy as np
from pathlib import Path
from apeiron.utils import get_device, extend_dict
from tqdm import tqdm
from apeiron.utils import to_cpu

class Inferencer:
    def __init__(self, device: str = None, **kwargs):
        super().__init__(**kwargs)
        # Check if a GPU (CUDA) is available
        if device:
            self.device = device
            print(f"Device: {device}")
        else:
            self.device = get_device()
            
        self.inferencer: MultiHeadModel
        self.losser: MultiModalLoss
        self.predictor: MultiModalPredict

        self.optim_cfgs: dict
        self.optimizer: Optimizer
        self.optimizer_state_dict: dict = None
        
        self.epoch_losses: dict = {}
        self.epoch_metrics: dict = {}


    # |-----------------------------------------------|
    # |------------ Set Up & Initialise --------------|
    # |-----------------------------------------------|


    def setup_inferencer(self, 
        in_features, mode=None, lbl_n_classes=None, ann_n_classes=None,
        inf_models = 'abmil', lbl_loss_type='bce', ann_loss_type='bce',
        lbl_cls_weights: dict = None, ann_cls_weights: dict = None,
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

        # 1. Prepare multimodal heads and lossers
        self.inferencer = MultiHeadModel(
            in_features, mode=mode, inf_models=inf_models,
            lbl_n_classes=lbl_n_classes, ann_n_classes=ann_n_classes, 
        )
        self.losser = MultiModalLoss(
            lbl_loss_type=lbl_loss_type, ann_loss_type=ann_loss_type, 
            lbl_cls_weights=lbl_cls_weights, ann_cls_weights=ann_cls_weights,
        )
        self.inferencer.to(self.device)
        self.losser.to(self.device)

        # 2. Prepare predictor with metricator
        self.predictor = MultiModalPredict(lbl_loss_type=lbl_loss_type, ann_loss_type=ann_loss_type)

        # 3. Prepare Optimizers
        self.optim_cfgs = {'lr': lr, 'optimizer': optimizer, 'weight_decay': weight_decay, 'scheduler': scheduler}
        self.optimizer_state_dict = None

        # Load model if available
        if chkp_path and Path(chkp_path).is_file():
            self.load_inferencer(chkp_path)


    def setup_optimizer(self, **kwargs):
        if len(self.optim_cfgs) > 0:
            kwargs = extend_dict(kwargs, **self.optim_cfgs)
        param_groups = [
            {'params': self.inferencer.parameters()},
            {'params': self.losser.parameters(), 'lr': 1e-3}
        ]
        self.optimizer = Optimizer(params=param_groups, **kwargs)
        if self.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(self.optimizer_state_dict)


    # |-----------------------------------------------|
    # |-------------- Train & Evaluate ---------------|
    # |-----------------------------------------------|


    def prep_data(self, data: dict, mode: Literal['cpu', 'cuda']):
        for k, v in data.items():
            if v is None: 
                continue
            elif k in ['coords', 'features', 'annotation', 'label'] and len(v) > 0:
                if mode == 'cuda': 
                    data[k] = torch.as_tensor(v).to(self.device)
                elif mode == 'cpu':
                    data[k] = to_cpu(v)

    @torch.no_grad()
    def predict_data(self, data: dict, threshold=0.5):
        """Forward pass + prediction (single sample, no grad)."""

        self.prep_data(data, mode='cuda')
        mdata = self.inferencer(**data)
        self.predictor(mdata, target=data, threshold=threshold)

        data.pop('features', None)
        self.prep_data(data, mode='cpu')
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

        for data in tqdm(data_collector):
            self.optimizer.zero_grad()

            # 1. Compute loss (this now processes a whole batch)
            self.prep_data(data, mode='cuda')
            mdata: Modeldata = self.inferencer(**data)
            self.losser(mdata, **data)
            final_loss = mdata.composite['final_loss']

            # 2. Backprop and Step immediately
            final_loss.backward()
            self.optimizer.step()

            # 3. Accumulate loss tracking (using batch averages)
            self.cum_epoch(mdata.loss.get_dict(composite=mdata.composite), self.epoch_losses)

            # 4. Compute metrics (no grad — detached from the graph)
            with torch.no_grad():
                self.predictor(mdata, target=data)
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
            self.prep_data(data, mode='cuda')
            mdata: Modeldata = self.inferencer(**data)
            self.losser(mdata, **data)
            self.predictor(mdata, target=data, threshold=threshold)
            
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
                if not isinstance(val, (float, int)): val = val.item()
                epoch_dict[modality][ls_type] = epoch_dict[modality].get(ls_type, 0.0) + val

    def avg_epoch(self, n_samples, epoch_dict):
        if n_samples == 0:
            return
        for name, res in epoch_dict.items():
            for key, loss in res.items():
                epoch_dict[name][key] = loss / n_samples


    # |-----------------------------------------------|
    # |---------------- Save & Load ------------------|
    # |-----------------------------------------------|


    def save_inferencer(self, chkp_path, epoch=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.inferencer.state_dict(),
            'losser_state_dict': self.losser.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, chkp_path)

    def load_inferencer(self, chkp_path):

        # 1. Load the checkpoint file from disk
        checkpoint = torch.load(chkp_path, map_location=self.device)

        # 2. Load the weights into the objects
        self.inferencer.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.losser.load_state_dict(checkpoint['losser_state_dict'], strict=False)
        self.optimizer_state_dict = checkpoint['optimizer_state_dict']
        print('Loaded epoch:', checkpoint['epoch'])

        # 3. Move to device
        self.inferencer.to(self.device)
        self.losser.to(self.device)


    # |-----------------------------------------------|
    # |-------------------- Extra --------------------|
    # |-----------------------------------------------|

    def run_emb_fns(self, features=None, text=None):
        """
        Args:
        - features -> (B, N, F)
        - text -> list[str]
        
        output: (dict)
        - img_emb -> (B,H)
        - wrd_emb -> (B,H)
        
        usage:
        ```python
        # Calculate scores (matching score for each text)
        # (4, H) @ (H, 1) -> (4, 1)
        scores = wrd_emb @ img_emb.t()

        # Calculate scores (matching score for each emb)
        # (10, H) @ (H, 1) -> (10, 1)
        scores = img_emb @ wrd_emb.t()
        ```

        """
        vectors = {}

        img_emb_fn = self.inferencer.embedding_fns.get('img_emb_fn')
        if all([x is not None for x in [features, img_emb_fn]]):
            vectors.update(img_emb_fn(features=torch.as_tensor(features)))

        wrd_emb_fn = self.inferencer.embedding_fns.get('wrd_emb_fn')
        if all([x is not None for x in [text, wrd_emb_fn]]):
            vectors.update(wrd_emb_fn(text=text))

        return vectors
        