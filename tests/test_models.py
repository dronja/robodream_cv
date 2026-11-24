"""Tests for facehall.models: shapes, ranges, parameter counts against the legacy Keras models."""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from facehall.config import load_config
from facehall.models import (
    MODELS,
    Bicubic,
    CNet,
    FNet,
    ModelConfig,
    build_model,
    count_parameters,
)
from facehall.models.layers import ElementwisePReLU

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "model"

# model.summary() outputs saved in legacy/FH.ipynb (24 -> 96 px, x4).
LEGACY_FNET = {"trainable": 734_723, "non_trainable": 1_408}  # cell 19
LEGACY_CNET = {"trainable": 854_979, "non_trainable": 384}  # cell 13


def batch_norm_statistics(model: nn.Module) -> int:
    """BatchNorm moving mean and variance: Keras "Non-trainable params"."""
    total = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert module.running_mean is not None and module.running_var is not None
            total += module.running_mean.numel() + module.running_var.numel()
    return total


def test_fnet_matches_the_legacy_parameter_counts() -> None:
    model = FNet(scale=4)
    assert count_parameters(model) == LEGACY_FNET["trainable"]
    assert batch_norm_statistics(model) == LEGACY_FNET["non_trainable"]


def test_cnet_matches_the_legacy_parameter_counts() -> None:
    model = CNet(scale=4, lr_size=24)
    assert count_parameters(model) == LEGACY_CNET["trainable"]
    assert batch_norm_statistics(model) == LEGACY_CNET["non_trainable"]


def test_extra_stage_for_x8() -> None:
    # One more conv(64 -> 256, 3x3) + PixelShuffle + PReLU(64) stage.
    stage = 3 * 3 * 64 * 256 + 256 + 64
    assert count_parameters(FNet(scale=8)) - count_parameters(FNet(scale=4)) == stage
    # One more upsample + conv(64 -> 64, 3x3) + per-pixel PReLU at 8x the LR size + BN.
    extra = 3 * 3 * 64 * 64 + 64 + 64 * (16 * 8) ** 2 + 2 * 64
    assert (
        count_parameters(CNet(scale=8, lr_size=16)) - count_parameters(CNet(scale=4, lr_size=16))
        == extra
    )


@pytest.mark.parametrize(("scale", "lr_size"), [(4, 32), (8, 16)])
@pytest.mark.parametrize("name", MODELS)
def test_output_shape_and_range(name: str, scale: int, lr_size: int) -> None:
    model = build_model(ModelConfig(name=name), scale, lr_size).eval()
    lr = torch.rand(2, 3, lr_size, lr_size)
    with torch.no_grad():
        sr = model(lr)
    assert sr.shape == (2, 3, 128, 128)
    assert sr.dtype == torch.float32
    assert sr.min() >= 0.0 and sr.max() <= 1.0


@pytest.mark.parametrize("name", ["fnet", "cnet"])
def test_training_step_updates_every_parameter(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(ModelConfig(name=name), 4, 16).train()
    lr, hr = torch.rand(4, 3, 16, 16), torch.rand(4, 3, 64, 64)
    loss = F.mse_loss(model(lr), hr)
    torch.autograd.backward(loss)
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert not missing


def test_keras_default_initialisation() -> None:
    model = FNet(scale=4)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            out_c, in_c, kh, kw = module.weight.shape
            limit = math.sqrt(6 / ((in_c + out_c) * kh * kw))  # Glorot uniform
            assert module.weight.abs().max() <= limit
            assert module.bias is not None and not module.bias.any()
        if isinstance(module, nn.BatchNorm2d):
            assert (module.eps, module.momentum) == (1e-3, 0.01)
        if isinstance(module, nn.PReLU | ElementwisePReLU):
            assert not module.weight.any()


def test_cnet_rejects_other_input_sizes() -> None:
    model = CNet(scale=4, lr_size=24)
    with pytest.raises(ValueError, match="does not match the PReLU parameter shape"):
        model(torch.rand(1, 3, 32, 32))


def test_elementwise_prelu() -> None:
    layer = ElementwisePReLU(1, 1, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[0.5, 2.0]]]))
    x = torch.tensor([[[[-2.0, -1.0]]], [[[3.0, 4.0]]]])
    torch.testing.assert_close(layer(x), torch.tensor([[[[-1.0, -2.0]]], [[[3.0, 4.0]]]]))


def test_bicubic_baseline() -> None:
    model = Bicubic(scale=4)
    assert count_parameters(model) == 0
    lr = torch.rand(1, 3, 8, 8)
    expected = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
    torch.testing.assert_close(model(lr), expected.clamp(0, 1))
    torch.testing.assert_close(
        model(torch.full((1, 3, 8, 8), 0.3)), torch.full((1, 3, 32, 32), 0.3)
    )


@pytest.mark.parametrize("name", MODELS)
def test_repository_configs_build(name: str) -> None:
    config = load_config(CONFIGS / f"{name}.yaml", ModelConfig)
    assert config.name == name
    build_model(config, 4, 32)


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model 'srgan'"):
        build_model(ModelConfig(name="srgan"), 4, 32)
    for scale in (1, 3, 6):
        with pytest.raises(ValueError, match="power of two"):
            FNet(scale=scale)
    with pytest.raises(ValueError, match="positive"):
        Bicubic(scale=0)
