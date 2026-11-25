"""Relativistic average GAN losses (ESRGAN, arXiv:1809.00219, Sec. 3.2, Eqs. 1–2).

With raw discriminator outputs C(x), D_Ra(x_r, x_f) = σ(C(x_r) − E[C(x_f)]), the mean taken
over the mini-batch:

- discriminator: −E[log D_Ra(x_r, x_f)] − E[log(1 − D_Ra(x_f, x_r))];
- generator: −E[log(1 − D_Ra(x_r, x_f))] − E[log D_Ra(x_f, x_r)].

Both are computed with ``binary_cross_entropy_with_logits`` in fp32.
"""

import torch
import torch.nn.functional as F


def _bce(logits: torch.Tensor, target: float) -> torch.Tensor:
    logits = logits.float()
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target))


def ragan_discriminator_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """``real`` and ``fake``: raw discriminator outputs C(x) of HR and SR images."""
    return _bce(real - fake.mean(), 1.0) + _bce(fake - real.mean(), 0.0)


def ragan_generator_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return _bce(real - fake.mean(), 0.0) + _bce(fake - real.mean(), 1.0)
