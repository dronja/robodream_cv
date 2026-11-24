"""Exponential moving average (EMA) of model weights."""

import copy

import torch
from torch import nn


class ModelEMA:
    """A copy of ``model`` whose weights follow ``ema = decay * ema + (1 - decay) * weights``.

    Call ``update`` after every optimizer step. Floating-point buffers (BatchNorm running
    statistics) are averaged like the weights; other buffers (``num_batches_tracked``) are
    copied. The copy stays in eval mode and is used for validation and the ``best`` checkpoint.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.is_floating_point():
                value.lerp_(current[name], 1.0 - self.decay)
            else:
                value.copy_(current[name])
