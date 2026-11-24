"""Layers with Keras default settings, for the 1:1 ports of the legacy Keras models.

The legacy models (``legacy/FH.ipynb``) used Keras defaults, which differ from PyTorch's:

- Conv2D: Glorot-uniform weights and zero bias (PyTorch: Kaiming-uniform weights and bias);
- BatchNormalization: epsilon 1e-3, moving-average momentum 0.99, which is PyTorch momentum 0.01
  (PyTorch default: eps 1e-5, momentum 0.1);
- PReLU: alpha initialised to 0 (PyTorch: 0.25).

Keras ``padding="same"`` with odd kernels and stride 1 is symmetric padding of ``k // 2``.
"""

import torch
from torch import nn


def conv(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv2d:
    """Keras ``Conv2D(out_channels, kernel_size, padding="same")``."""
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    assert layer.bias is not None
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def batch_norm(channels: int) -> nn.BatchNorm2d:
    """Keras ``BatchNormalization()``: epsilon 1e-3, momentum 0.99 (PyTorch momentum 0.01)."""
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)


def prelu(channels: int) -> nn.PReLU:
    """Keras ``PReLU(shared_axes=[1, 2])``: one alpha per channel, initialised to 0."""
    return nn.PReLU(num_parameters=channels, init=0.0)


class ElementwisePReLU(nn.Module):
    """Keras ``PReLU()`` without ``shared_axes``: one alpha per channel *and* position.

    The parameter has shape (C, H, W), so the layer only accepts inputs of that size.
    """

    def __init__(self, channels: int, height: int, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, height, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1:] != self.weight.shape:
            raise ValueError(
                f"input of shape {tuple(x.shape[1:])} does not match the PReLU parameter "
                f"shape {tuple(self.weight.shape)}"
            )
        return x.clamp(min=0) + self.weight * x.clamp(max=0)
