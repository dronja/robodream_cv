"""Tests for facehall.losses (perceptual, relativistic adversarial, identity) and the
discriminator."""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg19

from facehall.losses import (
    LAYERS,
    IdentityLoss,
    InceptionResnetV1,
    PerceptualLoss,
    crop_box,
    gaussian_kernel,
    load_facenet,
    ragan_discriminator_loss,
    ragan_generator_loss,
    vgg19_features,
)
from facehall.losses.facenet import INPUT_SIZE
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


def test_facenet_layout_and_loading(tmp_path: Path) -> None:
    # The embedding path of facenet-pytorch's InceptionResnetV1, without `logits`
    with torch.device("meta"):
        assert count_parameters(InceptionResnetV1()) == 23_482_624
    torch.manual_seed(0)
    network = InceptionResnetV1()
    state = {
        **network.state_dict(),
        "logits.weight": torch.zeros(8631, 512),  # in the published file, dropped on load
        "logits.bias": torch.zeros(8631),
    }
    torch.save(state, tmp_path / "facenet.pt")
    loaded = load_facenet(tmp_path / "facenet.pt")
    assert not loaded.training
    faces = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        embeddings = loaded(faces)
        assert torch.equal(embeddings, network.eval()(faces))
    assert embeddings.shape == (2, 512)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2))
    with pytest.raises(FileNotFoundError, match="download_weights.py facenet_vggface2"):
        load_facenet(tmp_path / "missing.pt")


def test_identity_crop_box() -> None:
    assert crop_box(128, 0.875, 0.55) == (14, 8, 112)  # the default of train.identity
    assert crop_box(32, 1.0, 0.5) == (0, 0, 32)
    for side, center_y in ((0.875, 0.6), (1.1, 0.5), (0.0, 0.5)):
        with pytest.raises(ValueError, match="leaves the image"):
            crop_box(128, side, center_y)


class Recorder(nn.Module):
    """Stands in for the recognition network: records its input, embeds by channel means."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(3))
        self.seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x.detach())
        return x.mean(dim=(2, 3)) * self.scale + torch.tensor([0.0, 0.0, 3.0])


def test_identity_loss() -> None:
    network = Recorder()
    loss = IdentityLoss(network, 32, side=0.5, center_y=0.5)
    assert loss.box == (8, 8, 16)
    hr = torch.rand(2, 3, 32, 32)
    assert loss(hr, hr).item() == pytest.approx(0.0, abs=1e-6)
    # the network sees the standardized square at 160 px: inside, the bilinear resize of the
    # crop (PR 3.3); its outermost pixels sample the image around the square
    loss.embed(hr)
    seen = network.seen[-1]
    assert seen.shape == (2, 3, INPUT_SIZE, INPUT_SIZE)
    crop = F.interpolate(hr[:, :, 8:24, 8:24], size=INPUT_SIZE, mode="bilinear")
    inner = (slice(None), slice(None), slice(5, -5), slice(5, -5))
    assert torch.allclose(seen[inner], ((crop * 255 - 127.5) / 128)[inner], atol=1e-4)
    loss.embed(torch.full((1, 3, 32, 32), 0.75))
    assert torch.allclose(network.seen[-1], torch.tensor((0.75 * 255 - 127.5) / 128))
    # 1 - cosine of the two embeddings, averaged over the batch; gradients reach SR only
    sr = torch.rand(2, 3, 32, 32, requires_grad=True)
    value = loss(sr, hr)
    target, output = (network(x) for x in network.seen[-2:])  # HR is embedded first
    expected = 1 - F.cosine_similarity(output, target, dim=1).mean()
    assert value.item() == pytest.approx(expected.item(), rel=1e-5)
    torch.autograd.backward(value)
    assert sr.grad is not None and sr.grad[:, :, :6].abs().sum() == 0  # outside the square
    assert sr.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in loss.parameters())
    loss.train()
    assert loss.training and not loss.network.training


def test_identity_jitter() -> None:
    def make(seed: int = 0) -> IdentityLoss:
        return IdentityLoss(Recorder(), 32, side=0.5, shift=0.1, rotation=10, scale=0.2, seed=seed)

    loss = make()
    base = loss.transforms(3, jitter=False)
    first, second = loss.transforms(3, jitter=True), loss.transforms(3, jitter=True)
    assert not torch.equal(first, second) and not torch.equal(first, base)
    assert torch.equal(make().transforms(3, jitter=True), first)  # the seed fixes the stream
    assert not torch.equal(make(seed=1).transforms(3, jitter=True), first)
    # limits: the square moves by at most shift x its side, and its size stays within scale
    half = base[0, 0, 0]
    shifts = (first[:, :, 2] - base[:, :, 2]).abs()
    assert (shifts <= 2 * half * 0.1 + 1e-6).all()
    sizes = first[:, :, :2].det().sqrt() / half
    assert ((sizes >= 1 / 1.2 - 1e-6) & (sizes <= 1.2 + 1e-6)).all()
    # SR and HR of one image share one transform: equal images give zero loss
    hr = torch.rand(3, 3, 32, 32)
    assert loss(hr, hr).item() == pytest.approx(0.0, abs=1e-6)
    # the generator state can be saved and restored (training resume)
    state = loss.generator.get_state()
    expected = loss.transforms(2, jitter=True)
    loss.generator.set_state(state)
    assert torch.equal(loss.transforms(2, jitter=True), expected)
    # similarity (validation) never jitters
    assert torch.equal(loss.similarity(hr, hr.flip(3)), loss.similarity(hr, hr.flip(3)))
    with pytest.raises(ValueError, match="non-negative"):
        IdentityLoss(Recorder(), 32, side=0.5, blur=-1.0)


def test_identity_blur() -> None:
    kernel = gaussian_kernel(1.0)
    assert kernel.numel() == 7 and kernel.sum().item() == pytest.approx(1.0)
    plain = IdentityLoss(Recorder(), 32, side=0.5)
    blurred = IdentityLoss(Recorder(), 32, side=0.5, blur=1.0)
    flat = torch.full((1, 3, 32, 32), 0.3)
    theta = plain.transforms(1, jitter=False)
    assert torch.allclose(blurred.faces(flat, theta), plain.faces(flat, theta))
    # a one-pixel checkerboard (the kind of pattern the defence targets) is mostly removed
    board = (torch.arange(32)[:, None] + torch.arange(32)[None, :]) % 2
    images = (0.5 + 0.2 * (board * 2 - 1)).float().expand(1, 3, 32, 32)
    assert blurred.faces(images, theta).std() < 0.1 * plain.faces(images, theta).std()
