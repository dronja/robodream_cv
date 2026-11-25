"""Tests for facehall.losses (perceptual and relativistic adversarial) and the discriminator."""

import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torchvision.models import vgg19

from facehall.losses import (
    LAYERS,
    PerceptualLoss,
    ragan_discriminator_loss,
    ragan_generator_loss,
    vgg19_features,
)
from facehall.models import count_parameters
from facehall.models.discriminators import VGGDiscriminator


def test_vgg19_layers_end_before_the_activation() -> None:
    with torch.device("meta"):
        features = list(vgg19().features)
    convs = [i for i, layer in enumerate(features) if isinstance(layer, nn.Conv2d)]
    # conv3_4, conv4_4 and conv5_4 are the 8th, 12th and 16th convolutions
    assert {
        name: convs[n - 1] + 1 for name, n in (("conv3_4", 8), ("conv4_4", 12), ("conv5_4", 16))
    } == LAYERS
    for end in LAYERS.values():
        assert isinstance(features[end - 1], nn.Conv2d) and isinstance(features[end], nn.ReLU)


def test_vgg19_features_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown VGG19 layer"):
        vgg19_features(tmp_path / "vgg19.pth", "conv6_1")
    with pytest.raises(FileNotFoundError, match="download_weights.py vgg19"):
        vgg19_features(tmp_path / "vgg19.pth")


def test_perceptual_loss() -> None:
    torch.manual_seed(0)
    loss = PerceptualLoss(nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, 4, 3)))
    hr = torch.rand(2, 3, 16, 16)
    sr = torch.rand(2, 3, 16, 16, requires_grad=True)
    assert loss(hr, hr).item() == 0.0
    value = loss(sr, hr)
    assert value.item() > 0
    torch.autograd.backward(value)
    assert sr.grad is not None and sr.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in loss.parameters())
    loss.train()
    assert loss.training and not loss.features.training
    # inputs are normalized with the ImageNet statistics
    identity = PerceptualLoss(nn.Identity())
    grey = torch.full((1, 3, 1, 1), 0.5)
    expected = (0.5 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(identity.embed(grey).flatten(), expected)


def test_ragan_losses() -> None:
    equal = torch.zeros(4)
    assert ragan_discriminator_loss(equal, equal).item() == pytest.approx(2 * math.log(2))
    assert ragan_generator_loss(equal, equal).item() == pytest.approx(2 * math.log(2))
    real, fake = torch.tensor([3.0, 1.0, 2.0]), torch.tensor([-1.0, 0.5, -2.0])
    # the generator loss is the discriminator loss with the roles swapped
    assert ragan_generator_loss(real, fake) == pytest.approx(
        ragan_discriminator_loss(fake, real).item()
    )
    assert ragan_discriminator_loss(real, fake) < ragan_discriminator_loss(fake, real)
    # relativistic: shifting all scores together changes nothing
    assert ragan_discriminator_loss(real + 5, fake + 5).item() == pytest.approx(
        ragan_discriminator_loss(real, fake).item()
    )
    expected = (
        -torch.log(torch.sigmoid(real - fake.mean())).mean()
        - torch.log(1 - torch.sigmoid(fake - real.mean())).mean()
    )
    assert ragan_discriminator_loss(real, fake).item() == pytest.approx(expected.item())


def test_vgg_discriminator() -> None:
    # The layer layout of BasicSR's VGGStyleDiscriminator at 128 px and 64 channels
    assert count_parameters(VGGDiscriminator(128, 64)) == 14_499_401
    small = VGGDiscriminator(32, 4)
    assert small(torch.rand(3, 3, 32, 32)).shape == (3,)
    with pytest.raises(ValueError, match="not divisible"):
        VGGDiscriminator(48)
