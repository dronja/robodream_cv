"""The generator objective of one batch, and the discriminator update (Phase 3, PR 3.2, 3.3).

``Objective(sr, hr)`` returns the weighted sum of the pixel loss, the perceptual loss, the
identity loss and the generator's adversarial loss. With an adversarial weight it also owns the
discriminator and its optimizer: ``after_step`` updates the discriminator on the last batch,
after the generator step, as ESRGAN alternates the two (arXiv:1809.00219, Sec. 4.1). The
discriminator's learning rate follows the generator's schedule: it is scaled by the same factor.

The loss terms are accumulated for logging: ``take_window`` returns their means since the
previous call, ``take_epoch`` since the start of the epoch.
"""

from collections.abc import Callable

import torch
from torch import nn

from facehall.losses.adversarial import ragan_discriminator_loss, ragan_generator_loss
from facehall.losses.identity import IdentityLoss
from facehall.losses.perceptual import PerceptualLoss
from facehall.train.config import LossWeights

LossFunction = Callable[..., torch.Tensor]


class Objective:
    def __init__(
        self,
        pixel: LossFunction,
        weights: LossWeights,
        *,
        perceptual: PerceptualLoss | None = None,
        identity: IdentityLoss | None = None,
        discriminator: nn.Module | None = None,
        d_optimizer: torch.optim.Optimizer | None = None,
        g_lr: float = 1.0,
        device: torch.device | None = None,
        precision: str = "fp32",
    ) -> None:
        if weights.perceptual > 0 and perceptual is None:
            raise ValueError("a perceptual weight needs a perceptual loss")
        if weights.identity > 0 and identity is None:
            raise ValueError("an identity weight needs an identity loss")
        if weights.adversarial > 0 and (discriminator is None or d_optimizer is None):
            raise ValueError("an adversarial weight needs a discriminator and its optimizer")
        self.pixel = pixel
        self.weights = weights
        self.perceptual = perceptual
        self.identity = identity
        self.discriminator = discriminator if weights.adversarial > 0 else None
        self.d_optimizer = d_optimizer if weights.adversarial > 0 else None
        self.d_lr = [group["lr"] for group in d_optimizer.param_groups] if d_optimizer else []
        self.g_lr = g_lr
        self.device = device or torch.device("cpu")
        self.precision = precision
        self._last: tuple[torch.Tensor, torch.Tensor] | None = None
        self._window: dict[str, torch.Tensor] = {}
        self._epoch: dict[str, torch.Tensor] = {}
        self._counts = {"window": 0, "epoch": 0, "d_window": 0, "d_epoch": 0}

    @property
    def adversarial(self) -> bool:
        return self.discriminator is not None

    def _autocast(self) -> torch.autocast:
        enabled = self.precision == "bf16"
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        parts: dict[str, torch.Tensor] = {}
        if self.weights.pixel > 0:
            parts["pixel"] = self.pixel(sr.float(), hr)
        if self.perceptual is not None and self.weights.perceptual > 0:
            with self._autocast():
                parts["perceptual"] = self.perceptual(sr, hr)
        if self.identity is not None and self.weights.identity > 0:
            with self._autocast():
                parts["identity"] = self.identity(sr, hr)
        if self.discriminator is not None:
            self.discriminator.requires_grad_(False)
            with self._autocast():
                with torch.no_grad():
                    real = self.discriminator(hr)
                fake = self.discriminator(sr)
            parts["adversarial"] = ragan_generator_loss(real, fake)
            self._last = (sr.detach(), hr)
        total = sum(
            (getattr(self.weights, name) * value for name, value in parts.items()),
            start=torch.zeros((), device=sr.device),
        )
        self._record(parts, "")
        return total

    def after_step(self, g_lr: float) -> None:
        """Update the discriminator on the last batch; ``g_lr``: the generator's current rate."""
        if self.discriminator is None or self.d_optimizer is None:
            return
        if self._last is None:
            raise RuntimeError("after_step called before a generator loss")
        sr, hr = self._last
        self._last = None
        factor = g_lr / self.g_lr
        for group, base in zip(self.d_optimizer.param_groups, self.d_lr, strict=True):
            group["lr"] = base * factor
        self.discriminator.requires_grad_(True)
        with self._autocast():
            real, fake = self.discriminator(hr), self.discriminator(sr)
        loss = ragan_discriminator_loss(real, fake)
        self.d_optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        self.d_optimizer.step()
        parts = {
            "discriminator": loss.detach(),
            "d_real": real.detach().float().mean(),
            "d_fake": fake.detach().float().mean(),
        }
        self._record(parts, "d_")

    def _record(self, parts: dict[str, torch.Tensor], kind: str) -> None:
        for store in (self._window, self._epoch):
            for name, value in parts.items():
                store[name] = store.get(name, torch.zeros((), device=value.device)) + value.detach()
        self._counts[f"{kind}window"] += 1
        self._counts[f"{kind}epoch"] += 1

    def _take(self, scope: str) -> dict[str, float]:
        store = self._window if scope == "window" else self._epoch
        d_names = {"discriminator", "d_real", "d_fake"}
        result = {}
        for name, value in store.items():
            count = self._counts[f"d_{scope}" if name in d_names else scope]
            if count:
                result[name] = value.item() / count
        store.clear()
        self._counts[scope] = self._counts[f"d_{scope}"] = 0
        return result

    def take_window(self) -> dict[str, float]:
        return self._take("window")

    def take_epoch(self) -> dict[str, float]:
        self._window.clear()
        self._counts["window"] = self._counts["d_window"] = 0
        return self._take("epoch")
