"""Tests for facehall.data.dataset on a tiny synthetic prepared dataset."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from facehall.align import CELEBA_TEMPLATE, alignment_matrix, warp
from facehall.align.template import CELEBA_TRAIN_MEAN_LANDMARKS
from facehall.data.config import DataConfig, DegradeConfig, JitterConfig, SceneConfig
from facehall.data.dataset import AlignedFaces
from facehall.data.degrade import Bicubic

# split, blurry, landmark offset (None: broken landmarks), see conftest.write_synthetic_dataset
ROWS = [
    ("train", False, (0, 0)),
    ("train", True, (1, -1)),
    ("train", False, (-2, 1)),
    ("train", True, (2, 2)),
    ("train", False, (0, -2)),
    ("train", False, (-1, 0)),
    ("val", False, (0, 0)),
    ("val", True, (1, 1)),
    ("val", False, (-70, -100)),  # face moved to the corner: the crop needs padding
    ("test", False, (0, 0)),
    ("test", False, None),
    ("test", False, (2, -1)),
]


@pytest.fixture(scope="module")
def root(
    tmp_path_factory: pytest.TempPathFactory,
    make_dataset: Callable[..., Path],
) -> Path:
    return make_dataset(tmp_path_factory.mktemp("dataset"), ROWS)


@pytest.fixture
def data(root: Path) -> DataConfig:
    return DataConfig(
        root=str(root),
        exclude_blurry=["train"],
        hflip=0.5,
        jitter=JitterConfig(shift=0.03, rotation=5.0, scale=0.05),
    )


def test_rows_are_filtered(data: DataConfig) -> None:
    train = AlignedFaces(data, DegradeConfig(), "train", 4)
    val = AlignedFaces(data, DegradeConfig(), "val", 4)
    test = AlignedFaces(data, DegradeConfig(), "test", 4)
    assert (len(train), train.dropped) == (4, {"upscale": 0, "blurry": 2})
    assert (len(val), val.dropped) == (3, {"upscale": 0, "blurry": 0})
    assert (len(test), test.dropped) == (2, {"upscale": 1, "blurry": 0})
    assert [val[i]["blurry"] for i in range(3)] == [False, True, False]


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("scale", [4, 8])
def test_sample_shapes_and_ranges(data: DataConfig, split: str, scale: int) -> None:
    sample = AlignedFaces(data, DegradeConfig(), split, scale)[0]
    hr, lr = sample["hr"], sample["lr"]
    assert hr.shape == (3, 128, 128) and hr.dtype == torch.float32
    assert lr.shape == (3, 128 // scale, 128 // scale) and lr.dtype == torch.float32
    for tensor in (hr, lr):
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0
        torch.testing.assert_close(tensor * 255, torch.round(tensor * 255), rtol=0, atol=1e-4)
    assert 0.0 <= sample["padded"] <= 1.0


def test_eval_sample_matches_the_reference_pipeline(data: DataConfig, root: Path) -> None:
    dataset = AlignedFaces(data, DegradeConfig(), "val", 4)
    sample = dataset[0]
    source = np.asarray(Image.open(dataset.path(0)).convert("RGB"))
    landmarks = np.round(CELEBA_TRAIN_MEAN_LANDMARKS).astype(int)
    expected = warp(source, alignment_matrix(landmarks, CELEBA_TEMPLATE, 128), 128)
    hr = torch.from_numpy(expected.copy()).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(sample["hr"], hr)
    torch.testing.assert_close(sample["lr"], Bicubic()(hr, 4, np.random.default_rng()).lr)


def test_eval_samples_are_fixed(data: DataConfig) -> None:
    dataset = AlignedFaces(data, DegradeConfig(), "val", 4)
    first = dataset[1]
    dataset.set_epoch(5)
    torch.testing.assert_close(dataset[1]["hr"], first["hr"])
    plain = AlignedFaces(data, DegradeConfig(), "val", 4, seed=123)
    torch.testing.assert_close(plain[1]["hr"], first["hr"])  # nothing random without augmentation


def test_train_augmentation_is_reproducible(data: DataConfig) -> None:
    def hr(seed: int, epoch: int) -> torch.Tensor:
        dataset = AlignedFaces(data, DegradeConfig(), "train", 4, seed=seed)
        dataset.set_epoch(epoch)
        result: torch.Tensor = dataset[0]["hr"]
        return result

    torch.testing.assert_close(hr(0, 0), hr(0, 0))
    assert not torch.equal(hr(0, 0), hr(0, 1))
    assert not torch.equal(hr(0, 0), hr(1, 0))
    unaugmented = AlignedFaces(data, DegradeConfig(), "train", 4, augment=False)[0]["hr"]
    assert not torch.equal(hr(0, 0), unaugmented)


def test_hflip_mirrors_the_crop(data: DataConfig) -> None:
    flip_only = replace(data, hflip=1.0, jitter=JitterConfig())
    flipped = AlignedFaces(flip_only, DegradeConfig(), "train", 4)[0]["hr"]
    plain = AlignedFaces(flip_only, DegradeConfig(), "train", 4, augment=False)[0]["hr"]
    torch.testing.assert_close(flipped, plain.flip(-1), rtol=0, atol=2 / 255)


def test_padded_fraction(data: DataConfig) -> None:
    val = AlignedFaces(data, DegradeConfig(), "val", 4)
    assert val[0]["padded"] == 0.0
    assert 0.1 < val[2]["padded"] < 0.9


def test_dataloader_batches(data: DataConfig) -> None:
    loader = DataLoader(AlignedFaces(data, DegradeConfig(), "train", 8), batch_size=3)
    batch = next(iter(loader))
    assert batch["hr"].shape == (3, 3, 128, 128)
    assert batch["lr"].shape == (3, 3, 16, 16)
    assert batch["index"].tolist() == [0, 1, 2]
    assert [json.loads(params) for params in batch["degradation"]] == [{}, {}, {}]


def test_pipeline_degradation_in_samples(data: DataConfig) -> None:
    degrade = DegradeConfig(kind="pipeline", grayscale=1.0)
    dataset = AlignedFaces(data, degrade, "val", 4)
    sample = dataset[0]
    params = json.loads(sample["degradation"])
    assert params["gray"] is True and len(params["stages"]) == 1
    for image in (sample["hr"], sample["lr"]):
        torch.testing.assert_close(image[0], image[1], rtol=0, atol=0)
    colour = AlignedFaces(data, DegradeConfig(), "val", 4)[0]["hr"]
    assert not torch.equal(colour[0], colour[1])
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert len(batch["degradation"]) == 2
    assert dataset.describe()["degradation"]["kind"] == "pipeline"  # type: ignore[index]


def test_invalid_settings_are_rejected(data: DataConfig) -> None:
    with pytest.raises(ValueError, match="does not divide"):
        AlignedFaces(data, DegradeConfig(), "train", 3)
    with pytest.raises(ValueError, match="unknown split"):
        AlignedFaces(data, DegradeConfig(), "validation", 4)
    with pytest.raises(ValueError, match="unknown template"):
        AlignedFaces(replace(data, template="ffhq"), DegradeConfig(), "train", 4)
    with pytest.raises(ValueError, match="exclude_blurry"):
        AlignedFaces(replace(data, exclude_blurry=["dev"]), DegradeConfig(), "train", 4)


def test_describe(data: DataConfig) -> None:
    info = AlignedFaces(data, DegradeConfig(), "test", 8).describe()
    assert info["samples"] == 2
    assert info["dropped"] == {"upscale": 1, "blurry": 0}
    assert info["augment"] is False
    assert info["degradation"]["kind"] == "bicubic"  # type: ignore[index]
    assert info["data"]["hr_size"] == 128  # type: ignore[index]


def test_scene_without_transform_gives_the_grid_crop(data: DataConfig) -> None:
    """An untransformed scene puts the crop on its LR grid: away from the borders, the LR crop
    equals the plain degradation of the HR crop (bicubic, with antialias)."""
    plain = AlignedFaces(data, DegradeConfig(), "val", 4)[0]
    still = SceneConfig(prob=1.0, shift=0.0, rotation=0.0, scale=0.0)
    dataset = AlignedFaces(replace(data, scene=still), DegradeConfig(), "val", 4)
    sample = dataset[0]
    torch.testing.assert_close(sample["hr"], plain["hr"], rtol=0, atol=0)
    inner = (slice(None), slice(3, -3), slice(3, -3))
    difference = (sample["lr"][inner] - plain["lr"][inner]).abs().max().item()
    assert difference <= 1.0 / 255.0 + 1e-6
    params = json.loads(sample["degradation"])
    assert params["scene"] == {"rotation": 0.0, "scale": 1.0, "shift": [0.0, 0.0]}


@pytest.mark.parametrize("scale", [4, 8])
def test_scene_samples(data: DataConfig, scale: int) -> None:
    scene = SceneConfig(prob=1.0, shift=0.5, rotation=10.0, scale=0.25)
    dataset = AlignedFaces(replace(data, scene=scene), DegradeConfig(), "train", scale)
    plain = AlignedFaces(data, DegradeConfig(), "train", scale)
    side = dataset.describe()["scene_side"]
    assert isinstance(side, int) and side % scale == 0 and (side - 128) % (2 * scale) == 0
    dataset.set_epoch(1)
    first, again = dataset[1], dataset[1]
    torch.testing.assert_close(first["lr"], again["lr"], rtol=0, atol=0)  # reproducible
    assert first["lr"].shape == (3, 128 // scale, 128 // scale)
    assert first["lr"].min() >= 0.0 and first["lr"].max() <= 1.0
    drawn = json.loads(first["degradation"])["scene"]
    assert abs(drawn["rotation"]) <= 10.0 and 0.8 - 1e-6 <= drawn["scale"] <= 1.25 + 1e-6
    assert max(abs(v) for v in drawn["shift"]) <= 0.5
    # the HR target does not depend on the scene: the same jitter and flip are drawn first
    plain.set_epoch(1)
    torch.testing.assert_close(first["hr"], plain[1]["hr"], rtol=0, atol=0)
    # the LR crop is close to the grid crop, but resampled
    difference = (first["lr"] - plain[1]["lr"]).abs().mean().item()
    assert 0.0 < difference < 0.15


def test_scene_grayscale_turns_the_target_grey(data: DataConfig) -> None:
    scene = SceneConfig(prob=1.0)
    degrade = DegradeConfig(kind="pipeline", grayscale=1.0)
    sample = AlignedFaces(replace(data, scene=scene), degrade, "val", 4)[0]
    assert json.loads(sample["degradation"])["gray"] is True
    for image in (sample["hr"], sample["lr"]):
        torch.testing.assert_close(image[0], image[1], rtol=0, atol=0)


def test_scene_probability_zero_draws_nothing(data: DataConfig) -> None:
    off = AlignedFaces(
        replace(data, scene=SceneConfig(prob=0.0, rotation=30.0)),
        DegradeConfig(kind="pipeline"),
        "val",
        4,
    )
    plain = AlignedFaces(data, DegradeConfig(kind="pipeline"), "val", 4)
    for index in range(len(plain)):
        assert off[index]["degradation"] == plain[index]["degradation"]
        torch.testing.assert_close(off[index]["lr"], plain[index]["lr"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="data.scene"):
        AlignedFaces(replace(data, scene=SceneConfig(prob=1.5)), DegradeConfig(), "val", 4)
