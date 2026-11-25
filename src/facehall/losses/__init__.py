"""Training losses beyond the pixel loss: perceptual (VGG19) and adversarial (relativistic)."""

from facehall.losses.adversarial import ragan_discriminator_loss, ragan_generator_loss
from facehall.losses.perceptual import LAYERS, PerceptualLoss, vgg19_features

__all__ = [
    "LAYERS",
    "PerceptualLoss",
    "ragan_discriminator_loss",
    "ragan_generator_loss",
    "vgg19_features",
]
