"""Optimizer and learning rate scheduler configuration for downstream training.

Provides a simple factory that wraps ``torch.optim`` optimizers and
``torch.optim.lr_scheduler`` schedulers behind a unified interface.

Usage::

    opt = Optimizer(model.parameters(), optimizer='adam', lr=1e-4, scheduler='cosine', n_epochs=50)
    for epoch in range(n_epochs):
        ...
        opt.step()
        opt.zero_grad()
        opt.step_scheduler()
"""

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR,
)


# ============================================================================
# Registry
# ============================================================================

_OPTIMIZER_REGISTRY = {
    'adam':    optim.Adam,
    'adamw':  optim.AdamW,
    'sgd':    optim.SGD,
    'rmsprop': optim.RMSprop,
}

_SCHEDULER_REGISTRY = {
    'cosine':   CosineAnnealingLR,
    'step':     StepLR,
    'plateau':  ReduceLROnPlateau,
    'onecycle': OneCycleLR,
}


# ============================================================================
# Optimizer wrapper
# ============================================================================

class Optimizer:
    """Thin wrapper around a ``torch.optim`` optimizer + optional LR scheduler.

    Args:
        params: Model parameters (from ``model.parameters()``).
        optimizer (str): Optimizer name. One of ``'adam'``, ``'adamw'``,
            ``'sgd'``, ``'rmsprop'``. Default ``'adam'``.
        lr (float): Learning rate. Default ``1e-4``.
        weight_decay (float): Weight decay. Default ``0.0``.
        scheduler (str or None): Scheduler name. One of ``'cosine'``,
            ``'step'``, ``'plateau'``, ``'onecycle'``, or ``None``.
            Default ``None``.
        n_epochs (int): Total epochs — used by cosine / onecycle schedulers.
            Default ``100``.
        steps_per_epoch (int): Steps per epoch — used by onecycle scheduler.
            Default ``1``.
        **kwargs: Extra keyword arguments forwarded to the scheduler
            constructor (e.g. ``step_size`` for StepLR).

    Attributes:
        optim: The underlying ``torch.optim.Optimizer``.
        sched: The underlying scheduler, or ``None``.
    """

    def __init__(
        self,
        params,
        optimizer: str = 'adam',
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler: str = None,
        n_epochs: int = 100,
        steps_per_epoch: int = 1,
        **kwargs,
    ):
        if optimizer not in _OPTIMIZER_REGISTRY:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from: {list(_OPTIMIZER_REGISTRY.keys())}")

        opt_cls = _OPTIMIZER_REGISTRY[optimizer]
        opt_kwargs = dict(lr=lr, weight_decay=weight_decay)
        if optimizer == 'sgd':
            opt_kwargs['momentum'] = kwargs.pop('momentum', 0.9)
        self.optim = opt_cls(params, **opt_kwargs)

        # Scheduler
        self.sched = None
        if scheduler is not None:
            if scheduler not in _SCHEDULER_REGISTRY:
                raise ValueError(f"Unknown scheduler '{scheduler}'. Choose from: {list(_SCHEDULER_REGISTRY.keys())}")
            sched_cls = _SCHEDULER_REGISTRY[scheduler]
            if scheduler == 'cosine':
                self.sched = sched_cls(self.optim, T_max=n_epochs, **kwargs)
            elif scheduler == 'step':
                self.sched = sched_cls(self.optim, step_size=kwargs.pop('step_size', 30), **kwargs)
            elif scheduler == 'plateau':
                self.sched = sched_cls(self.optim, mode='min', patience=kwargs.pop('patience', 10), **kwargs)
            elif scheduler == 'onecycle':
                self.sched = sched_cls(self.optim, max_lr=lr, epochs=n_epochs,
                                       steps_per_epoch=steps_per_epoch, **kwargs)

    def zero_grad(self):
        """Zero all parameter gradients."""
        self.optim.zero_grad()

    def step(self):
        """Perform a single optimisation step."""
        self.optim.step()

    def step_scheduler(self, metric=None):
        """Advance the LR scheduler by one step.

        Args:
            metric (float, optional): Validation metric for
                ``ReduceLROnPlateau``. Ignored by other schedulers.
        """
        if self.sched is None:
            return
        if isinstance(self.sched, ReduceLROnPlateau):
            if metric is not None:
                self.sched.step(metric)
        else:
            self.sched.step()

    @property
    def lr(self) -> float:
        """Current learning rate of the first param group."""
        return self.optim.param_groups[0]['lr']
        
    def state_dict(self):
        """Returns the state of the optimizer and scheduler."""
        state = {'optim': self.optim.state_dict()}
        if self.sched is not None:
            state['sched'] = self.sched.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the optimizer and scheduler state."""
        self.optim.load_state_dict(state_dict['optim'])
        if self.sched is not None and 'sched' in state_dict:
            self.sched.load_state_dict(state_dict['sched'])
