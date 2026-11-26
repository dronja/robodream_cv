"""Training losses beyond the pixel loss: perceptual (VGG19), adversarial (relativistic) and
identity (facenet embeddings)."""

from facehall.losses.adversarial import ragan_discriminator_loss, ragan_generator_loss
from facehall.losses.facenet import InceptionResnetV1, load_facenet
from facehall.losses.identity import IdentityLoss, crop_box, gaussian_kernel
from facehall.losses.perceptual import LAYERS, PerceptualLoss, vgg19_features

__all__ = [
    "LAYERS",
    "IdentityLoss",
    "InceptionResnetV1",
    "PerceptualLoss",
    "crop_box",
    "gaussian_kernel",
    "load_facenet",
    "ragan_discriminator_loss",
    "ragan_generator_loss",
    "vgg19_features",
]
