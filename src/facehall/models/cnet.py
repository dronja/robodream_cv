"""CNet: the legacy "simplified C-Face", ported 1:1 from ``legacy/FH.ipynb``.

Same layers as ``build_cface`` in the notebook: Conv 5×5, then per ×2 stage nearest upsampling and
Conv 3×3, each conv followed by PReLU and BatchNorm (in that order), and a 3×3 conv with sigmoid
output. The only generalisation is the scale (one stage per factor of 2; the notebook had ×4).

The notebook's ``PReLU()`` has no ``shared_axes``, so it learns one alpha per channel *and* pixel.
The port keeps this: the model is built for one LR size (``lr_size``) and accepts only that size.
At 24 px and ×4 the PReLU alphas are 774,144 of the 854,979 trainable parameters.
"""

import torch
from torch import nn

from facehall.models.fnet import upsampling_stages
from facehall.models.layers import ElementwisePReLU, batch_norm, conv


class CNet(nn.Module):
    def __init__(self, scale: int, lr_size: int, channels: int = 64) -> None:
        super().__init__()
        stages = upsampling_stages(scale)
        self.scale = scale
        self.lr_size = lr_size
        layers: list[nn.Module] = [
            conv(3, channels, 5),
            ElementwisePReLU(channels, lr_size, lr_size),
            batch_norm(channels),
        ]
        for stage in range(1, stages + 1):
            size = lr_size * 2**stage
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                conv(channels, channels, 3),
                ElementwisePReLU(channels, size, size),
                batch_norm(channels),
            ]
        layers.append(conv(channels, 3, 3))
        self.layers = nn.Sequential(*layers)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layers(lr))
