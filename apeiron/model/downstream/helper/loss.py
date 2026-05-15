import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# (1) Primitive losses — slide labels: logits (B, C), labels (B, C)
# ============================================================================

def weights_tensor(cls_weights: dict | None, device=None) -> torch.Tensor | None:
    """Convert ``{class_id: weight}`` dict to a float tensor, or ``None``."""
    if cls_weights is None:
        return None
    keys = sorted(cls_weights.keys())
    t = torch.tensor([cls_weights[k] for k in keys], dtype=torch.float32)
    return t.to(device) if device is not None else t


class HardCELoss(nn.Module):
    """Cross-entropy for hard (one-hot) labels.

    Converts one-hot ``(B, C)`` labels to class indices and applies
    standard cross-entropy.

    Args:
        cls_weights (dict | None): ``{class_id: weight}`` per-class weights.
            Weight 0 excludes a class from the loss. Default ``None`` (equal).

    Input:  logits ``(B, C)``, labels ``(B, C)`` one-hot
    Output: scalar loss
    """

    def __init__(self, cls_weights: dict = None):
        super().__init__()
        w = weights_tensor(cls_weights)
        self.register_buffer('weight', w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = labels.argmax(dim=-1)  # (B,)
        return F.cross_entropy(logits, targets, weight=self.weight)


class FocalLoss(nn.Module):
    """Focal loss for hard (one-hot) labels.

    Down-weights well-classified examples to focus on hard negatives.

    Args:
        gamma (float): Focusing parameter. Default 2.0.
        cls_weights (dict | None): ``{class_id: weight}`` per-class weights
            applied to the CE term. Weight 0 excludes a class. Default ``None``.

    Input:  logits ``(B, C)``, labels ``(B, C)`` one-hot
    Output: scalar loss
    """

    def __init__(self, gamma: float = 2.0, cls_weights: dict = None):
        super().__init__()
        self.gamma = gamma
        w = weights_tensor(cls_weights)
        self.register_buffer('weight', w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = labels.argmax(dim=-1)  # (B,)
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')  # (B,)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class SoftCELoss(nn.Module):
    """Cross-entropy for soft (distribution) labels.

    Computes ``-sum(w * labels * log_softmax(logits))`` per sample.

    Args:
        cls_weights (dict | None): ``{class_id: weight}`` per-class weights.
            Weight 0 excludes a class. Default ``None`` (equal).

    Input:  logits ``(B, C)``, labels ``(B, C)`` soft distribution (sums to 1)
    Output: scalar loss
    """

    def __init__(self, cls_weights: dict = None):
        super().__init__()
        w = weights_tensor(cls_weights)
        self.register_buffer('weight', w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
        loss = -(labels * log_probs * self.weight).sum(dim=-1)    # (B,)
        return loss.mean()


class KLDivLoss(nn.Module):
    """KL divergence for soft (distribution) labels.

    Measures how the predicted distribution diverges from the target.

    Args:
        cls_weights (dict | None): ``{class_id: weight}`` per-class weights
            applied before reduction. Weight 0 excludes a class. Default ``None``.

    Input:  logits ``(B, C)``, labels ``(B, C)`` soft distribution (sums to 1)
    Output: scalar loss
    """

    def __init__(self, cls_weights: dict = None):
        super().__init__()
        w = weights_tensor(cls_weights)
        self.register_buffer('weight', w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
        kl = F.kl_div(log_probs, labels, reduction='none')  # (B, C)
        if self.weight is not None:
            kl = kl * self.weight
        return kl.sum(dim=-1).mean()


class MSELoss(nn.Module):
    """Mean squared error for score (regression) labels.

    Args:
        cls_weights (dict | None): ``{class_id: weight}`` per-class weights.
            Weight 0 excludes a class. Default ``None`` (equal).

    Input:  logits ``(B, C)``, labels ``(B, C)`` continuous scores
    Output: scalar loss
    """

    def __init__(self, cls_weights: dict = None):
        super().__init__()
        w = weights_tensor(cls_weights)
        self.register_buffer('weight', w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        err = (logits - labels) ** 2  # (B, C)
        if self.weight is not None:
            err = err * self.weight
        return err.mean()


class MAELoss(nn.Module):
    """Mean absolute error for score (regression) labels.

    Args:
        cls_weights (dict | None): ``{class_id: weight}`` per-class weights.
            Weight 0 excludes a class. Default ``None`` (equal).

    Input:  logits ``(B, C)``, labels ``(B, C)`` continuous scores
    Output: scalar loss
    """

    def __init__(self, cls_weights: dict = None):
        super().__init__()
        w = weights_tensor(cls_weights)
        self.register_buffer('weight', w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        err = (logits - labels).abs()  # (B, C)
        if self.weight is not None:
            err = err * self.weight
        return err.mean()


class BCEWithLogitsLoss(nn.Module):
    """Binary cross-entropy for multi-label classification.

    Each class is treated as an independent binary classification.

    Args:
        cls_weights (dict | None): ``{class_id: weight}`` per-class weights.
            Weight 0 excludes a class. Default ``None`` (equal).

    Input:  logits ``(B, C)``, labels ``(B, C)`` multi-label (each class in [0,1])
    Output: scalar loss
    """

    def __init__(self, cls_weights: dict = None):
        super().__init__()
        w = weights_tensor(cls_weights)
        self.register_buffer('weight', w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, labels.float(), weight=self.weight)


class MultiLabelFocalLoss(nn.Module):
    """Focal loss for multi-label classification.

    Applies focal modulation to per-class BCE independently.

    Args:
        gamma (float): Focusing parameter. Default 2.0.
        alpha (float): Positive class weight. Default 0.25.
        cls_weights (dict | None): ``{class_id: weight}`` per-class weights.
            Weight 0 excludes a class. Default ``None`` (equal).

    Input:  logits ``(B, C)``, labels ``(B, C)`` multi-label (each class in [0,1])
    Output: scalar loss
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, cls_weights: dict = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        w = weights_tensor(cls_weights)
        self.register_buffer('weight', w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float()
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')  # (B, C)
        pt = torch.exp(-bce)
        focal = self.alpha * ((1 - pt) ** self.gamma) * bce  # (B, C)
        if self.weight is not None:
            focal = focal * self.weight
        return focal.mean()


# ============================================================================
# (1.5) Ranking losses
# ============================================================================

class PairwiseMarginLoss(nn.Module):
    """Pairwise margin ranking loss for soft float labels.
    
    Ranks the CLASSES within each sample.
    Penalizes when the predicted score for a higher true-label class 
    is not greater than a lower true-label class by at least `margin`.
    
    `Best for equally important ranks across classes`
    Args:
        margin (float): The required margin between predictions. Default 0.1.
        cls_weights (dict | None): ``{class_id: weight}`` per-class weights.
    """
    def __init__(self, margin: float = 0.1, cls_weights: dict = None):
        super().__init__()
        self.margin = margin
        w = weights_tensor(cls_weights)
        self.register_buffer('weight', w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), labels: (B, C)
        # We want to rank classes against each other for each item in the batch
        
        # diff_true[b, i, j] = labels[b, i] - labels[b, j]
        diff_true = labels.unsqueeze(2) - labels.unsqueeze(1)  # (B, C, C)
        diff_pred = logits.unsqueeze(2) - logits.unsqueeze(1)  # (B, C, C)
        
        mask = diff_true > 0  # Only pairs where true class i > true class j
        
        if mask.sum() > 0:
            pair_loss = F.relu(self.margin - diff_pred) # (B, C, C)
            
            if self.weight is not None:
                # Apply weight of the "higher" class (i) to the loss
                w = self.weight.unsqueeze(0).unsqueeze(2) # (1, C, 1)
                pair_loss = pair_loss * w
                
            loss = (pair_loss * mask).sum() / mask.sum()
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        return loss

class ListNetLoss(nn.Module):
    """ListNet loss for ranking soft float labels.
    
    Applies softmax over the CLASS dimension to treat the classes as a list 
    to be ranked, then computes cross-entropy against the softmax of true labels.

    `Best for securing top rank only`
    Args:
        cls_weights (dict | None): ``{class_id: weight}`` per-class weights.
    """
    def __init__(self, cls_weights: dict = None):
        super().__init__()
        w = weights_tensor(cls_weights)
        self.register_buffer('weight', w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Rank classes within each sample
        pred_probs = F.softmax(logits, dim=-1) # (B, C)
        true_probs = F.softmax(labels, dim=-1) # (B, C)
        
        loss = -(true_probs * torch.log(pred_probs + 1e-7)) # (B, C)
        
        if self.weight is not None:
            loss = loss * self.weight
            
        return loss.sum(dim=-1).mean()

# ============================================================================
# (2) Structural losses — segmentation
# ============================================================================

class DiceLoss(nn.Module):
    """Dice loss for segmentation masks.

    Operates per-class and averages. Accepts logits (applies sigmoid internally).

    Args:
        smooth (float): Laplace smoothing. Default 1e-6.
        cls_weights (dict | None): ``{class_id: weight}`` per-class weights.
            Weight 0 excludes a class. Default ``None`` (equal).

    Input:  logits ``(C, N)``, labels ``(C, N)`` binary masks
    Output: scalar loss
    """

    def __init__(self, smooth: float = 1e-6, cls_weights: dict = None):
        super().__init__()
        self.smooth = smooth
        w = weights_tensor(cls_weights)
        self.register_buffer('weight', w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = logits.sigmoid()  # (C, N)
        num = 2 * (probs * labels).sum(dim=-1) + self.smooth   # (C,)
        den = probs.sum(dim=-1) + labels.sum(dim=-1) + self.smooth  # (C,)
        dice = 1 - num / den  # (C,)
        if self.weight is not None:
            dice = dice * self.weight
        return dice.mean()

        
# ============================================================================
# Helper: build a label loss by name
# ============================================================================

_LABEL_LOSS_REGISTRY = {
    'hard_ce':     HardCELoss,
    'focal':       FocalLoss,
    'soft_ce':     SoftCELoss,
    'kl_div':      KLDivLoss,
    'mse':         MSELoss,
    'mae':         MAELoss,
    'bce':         BCEWithLogitsLoss,
    'multi_fc':    MultiLabelFocalLoss,
    'margin':      PairwiseMarginLoss,
    'listnet':     ListNetLoss,
    'dice':        DiceLoss,
}

def build_loss(loss_type: str, cls_weights: dict = None, **kwargs) -> nn.Module:
    """Instantiate a label loss by name.

    Args:
        loss_type (str): Key from ``_LABEL_LOSS_REGISTRY``.
        cls_weights (dict | None): ``{class_id: weight}`` forwarded to the
            loss constructor as ``cls_weights``. Default ``None``.
        **kwargs: Forwarded to the loss constructor.

    Returns:
        nn.Module: Instantiated loss function.

    Raises:
        ValueError: If ``loss_type`` is not recognised.
    """
    if loss_type not in _LABEL_LOSS_REGISTRY:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Choose from: {list(_LABEL_LOSS_REGISTRY.keys())}")
    cls = _LABEL_LOSS_REGISTRY[loss_type]
    try:
        return cls(cls_weights=cls_weights, **kwargs)
    except TypeError:
        return cls(**kwargs)