# Data

> Composition, sources, verified source facts, layout and reproduction of the datasets.
> Last updated: **2026-11-24**.
>
> - Every number here comes from reading the source files directly or from a real run of
>   `scripts/prepare_dataset.py`.
> - Status, plan and known data limitations: [`PROJECT.md`](PROJECT.md) §6.
> - License terms: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) §2.

## Contents

1. [Datasets in use](#1-datasets-in-use)
2. [Source: the CelebA parquet mirror](#2-source-the-celeba-parquet-mirror)
3. [Splits](#3-splits)
4. [Verified properties of the mirror](#4-verified-properties-of-the-mirror)
5. [Prepared dataset](#5-prepared-dataset)
6. [How to reproduce](#6-how-to-reproduce)

## 1. Datasets in use

| Dataset | Used for | Location | Terms |
|---|---|---|---|
| CelebA aligned (`img_align_celeba`) from the Hugging Face mirror `eurecom-ds/celeba` | Training and evaluation, Phases 1–3 (D8) | Raw: `/mnt/storage/AI/Face/celeba_eurecom/data/`. Prepared: `/mnt/storage/AI/Face/celeba/` | Non-commercial research only, no redistribution |

The legacy 89×109 dataset is not used (D9). No image, annotation or manifest of these datasets is
committed to the repository (D14). This file holds only counts and checksums.

## 2. Source: the CelebA parquet mirror

- Hugging Face dataset `eurecom-ds/celeba`, downloaded on 2026-10-24 (file dates; download log in
  `/mnt/storage/AI/Face/download.log`). The dataset card says it was produced from
  `torchvision.datasets.CelebA`.
- 20 parquet shards, 8.92 GB (8,917,038,019 bytes). The shard name gives the split: `train`
  (18 shards), `validation` and `test` (1 shard each).
- The mirror publishes no checksums of its own. The sha256 values below were computed by
  `scripts/prepare_dataset.py` on 2026-11-24, and every run recomputes them.

| Shard | Split | Rows | sha256 |
|---|---|---|---|
| `train-00000-of-00018.parquet` | train | 9,043 | `7bb88c060773ce506255568b0208becb67798395b72f7204d702009ee44d3162` |
| `train-00001-of-00018.parquet` | train | 9,043 | `f98a402b2989300a2e4e86f5a15e7c54b97e30ca8c3b4601e29625aa76989a36` |
| `train-00002-of-00018.parquet` | train | 9,043 | `8d31c5d38ac7e18b5185d85648f9f6ac63d8a21b1c8f6b986f1cb166636d8bc6` |
| `train-00003-of-00018.parquet` | train | 9,043 | `9ae64a41be6942be1aa2e9b664f45441282794df19e5e1eb8f0542f47074f763` |
| `train-00004-of-00018.parquet` | train | 9,043 | `753c3fea30352982d6c9b2b6a8eb4201ec18cc951e0c20a28894d7bcc5ed4414` |
| `train-00005-of-00018.parquet` | train | 9,043 | `a0a0d0ea07c0c997300e1e55b63e858e6174ca00da18024cf4a4900f15be8d35` |
| `train-00006-of-00018.parquet` | train | 9,043 | `e88f766a4f0a027c5bb0bbd3d9259bd09d386e333efb908a369bd9042d45befc` |
| `train-00007-of-00018.parquet` | train | 9,043 | `23ed63e1c31297fbef45d7a3063b71c7f98f15092099acbb4a9ea6d7a7a8a2f3` |
| `train-00008-of-00018.parquet` | train | 9,043 | `44a208a41c79e3547e452a6561ca6e325004fdff21be24649352684813102820` |
| `train-00009-of-00018.parquet` | train | 9,043 | `9c6f39f49d3d9d2f382b46660037860f4e1f8863b2ad29897ca9af3363f7a59d` |
| `train-00010-of-00018.parquet` | train | 9,043 | `377ccdbb5c6a8931609148cf072d788900d783e66ed4789b079c5990c4cbe48e` |
| `train-00011-of-00018.parquet` | train | 9,043 | `97b402f512a868b3a047d6895cf0e5db2c8b78bfe16157292b08774d24d7a211` |
| `train-00012-of-00018.parquet` | train | 9,043 | `058bd47c61e548c7c00dfec8755b19ab2c769484434e1fe6f7f3cf39daa55292` |
| `train-00013-of-00018.parquet` | train | 9,043 | `9853c35f1e55e911d040c05b1efd5c8457ba38e8e076cafb896ea6bf3212c250` |
| `train-00014-of-00018.parquet` | train | 9,042 | `07d148eea86250511772e267ee28d5e74904387dd8c5d4d824a838a4c9811e5d` |
| `train-00015-of-00018.parquet` | train | 9,042 | `388b87899e1e7b303dbcc19c28395a6514727185b04cac9696f26ecc2336a0b7` |
| `train-00016-of-00018.parquet` | train | 9,042 | `ac480bd709e222ae74821c1777f7f18644759678f5e1d77ef8d66163518cd942` |
| `train-00017-of-00018.parquet` | train | 9,042 | `9ee16a454ab481dcd903f44665532f5e62dd1a9ca8642321d6ff998fd17cdeab` |
| `validation-00000-of-00001.parquet` | val | 19,867 | `fffd0f86ce6703f284074bcca188cee86e4e47249badfe0f85753d35d015bdfe` |
| `test-00000-of-00001.parquet` | test | 19,962 | `8ede8774f071f19febd13b18498226fe772a29f51fd99491dd55fd7bbef50a25` |

## 3. Splits

The official CelebA partition (D8):

| Split | Images | Identities | `Blurry = 1` | Stored format |
|---|---|---|---|---|
| train | 162,770 | 8,192 | 8,362 (5.14%) | PNG |
| val | 19,867 | 985 | 940 (4.73%) | JPEG |
| test | 19,962 | 1,000 | 1,010 (5.06%) | JPEG |
| **total** | **202,599** | **10,177** | **10,312** (5.09%) | |

Samples used by `facehall.data.dataset.AlignedFaces` with `configs/data/celeba_128.yaml`:

| Split | Samples | Dropped |
|---|---|---|
| train | 154,408 | 8,362 with `Blurry = 1` (D15) |
| val | 19,867 | none |
| test | 19,961 | 1, `img_align/test/001557.jpg`: broken landmarks, its crop would be an upscale (§4.5) |

- **Identity-disjoint:** no identity appears in two splits. `scripts/prepare_dataset.py` checks
  this on every run, and the manifest validation checks it again on every read.
- **Usage:** train is for training; validation is for model selection and tuning. The test
  split is frozen: it is used only for the final comparison of models and never for tuning. The
  frozen test set is the official test split minus the one image above.

## 4. Verified properties of the mirror

Checked on 2026-11-23 and 2026-11-24 by reading the parquet files and the prepared manifest
directly.

### 4.1 Schema and fields

Schema: `image` (struct `bytes`, `path`), `attributes` int8[40], `identity` int64, `bbox`
int32[4], `landmarks` int32[10] (fixed-size lists).

| Field | Usable | Evidence |
|---|---|---|
| `image` | yes | All 202,599 images decode as RGB 178×218. Storage format: §4.2. `path` is empty, so the original file names are not stored |
| `landmarks` | yes | Aligned-image coordinates `[lx, ly, rx, ry, nose_x, nose_y, lmouth_x, lmouth_y, rmouth_x, rmouth_y]`. Statistics: `PROJECT.md` §6.3 |
| `bbox` | **no** | Coordinates of the in-the-wild images, not of the aligned ones: max x+w = 4628; 88.5% of test boxes fall outside 178×218. Not copied to the manifest |
| `attributes` | yes | The 40 binary CelebA attributes, values {0, 1}; the mirror stores no names. Order: §4.3 |

### 4.2 Image encoding

The train split and the validation/test splits are stored differently:

| Split | Format (all rows) | Mean size |
|---|---|---|
| train (all 18 shards) | PNG | 51.6–51.9 KB per shard |
| validation, test | JPEG | 6.8 KB |

The train PNGs hold decoded JPEG pixels, not a different, cleaner source. Evidence: JPEG
compresses 8×8 blocks, which leaves small jumps at block borders. On the first 300 images of a
shard, the mean absolute luma difference across 8-px block borders is larger than inside the
blocks by this ratio (median):

| Shard | Ratio | Same images, block grid shifted by 4 px |
|---|---|---|
| train-00000-of-00018 | 1.211 | 0.984 |
| train-00017-of-00018 | 1.210 | 0.988 |
| test-00000-of-00001 | 1.215 | 0.986 |

So train and test carry the same JPEG artifacts. The prepared dataset keeps the stored bytes and
does not re-encode them (decision of 2026-11-24 in `DECISIONS.md`).

### 4.3 Attribute order

The mirror stores the attributes without names. They follow the order of the official
`list_attr_celeba.txt` (`facehall.data.celeba.ATTRIBUTE_NAMES`). Evidence over all 202,599
images:

| Check | Result |
|---|---|
| P(Wearing_Lipstick \| Male) / P(Wearing_Lipstick \| not Male) | 0.6% / 80.6% |
| P(Heavy_Makeup \| Male) / P(Heavy_Makeup \| not Male) | 0.3% / 66.1% |
| P(attribute \| not Male) for Goatee, Mustache, Sideburns, 5_o_Clock_Shadow, Bald, Wearing_Necktie | 0.0% each (rounded to 0.1%) |
| P(No_Beard \| not Male) | 99.88% |
| P(Gray_Hair \| Young) / P(Gray_Hair \| not Young) | 0.24% / 17.70% |

Frequencies over all images: Male 41.7%, Young 77.4%, Blurry 5.1%, Pale_Skin 4.3%, Eyeglasses
6.5%. These checks confirm Male, Young and the facial-hair and make-up attributes directly. The
other positions, Pale_Skin included, rest on the standard order and match it by frequency.

### 4.4 Row order

The original file names (`000001.jpg` …) are not stored. The official partition assigns
contiguous file-number ranges to the splits, and the split sizes match it exactly. Whether the rows
inside a split follow the official file order still *needs verification*, for example against the
official landmark file. This matters for joining external annotations keyed by file name (D10).
Until then, the manifest traces every row by `source_file` and `source_row`.

### 4.5 Landmark quality

Measured over the prepared manifest on 2026-11-24. The fit residuals use the alignment from PR 1.3
(`PROJECT.md` §6.3) and are given in pixels of a 128×128 crop.

- **Interocular distance** (source px), the same in every split: median 39, p5 33, p95 43. The
  minimum is 4–5 px (profile faces); the maximum is 66 in train and 52–53 in val/test.
- **Fit to the template:** the mean landmark residual is 2.6 px in every split. Per image, the
  largest residual has p95 9.5 px (train). More than 15 px: train 159, val 14, test 19 images; more
  than 20 px: 1 train image. Checked visually on the largest test residuals: they are profile
  faces, not annotation errors.
- **Broken landmarks:** `img_align/test/001557.jpg` shows a normal-size frontal face, but all five
  points are clustered between the eyes (eye distance 10 px). Checked visually. Its crop would be a
  ×3.3 zoom into the eye region. It is the only image whose crop side is below 140 source px; the
  "HR must not be an upscale" rule excludes it (`PROJECT.md` §9, PR 1.4).

## 5. Prepared dataset

Built by `scripts/prepare_dataset.py` (code: `facehall.data.celeba`, `facehall.data.manifest`).

```
/mnt/storage/AI/Face/celeba/
├── img_align/
│   ├── train/000000.png … 162769.png    # stored bytes, not re-encoded
│   ├── val/000000.jpg … 019866.jpg
│   └── test/000000.jpg … 019961.jpg
├── manifest.parquet                     # one row per image, 51 columns
└── prepare_info.json                    # provenance and the checksums below
```

The file index counts rows within the split, in shard order. Disk use, as file sizes in 10^9
bytes: train 8.63 GB, val 0.14 GB, test 0.14 GB; the manifest is 17.9 MB.

### 5.1 Manifest columns

| Column | Type | Meaning |
|---|---|---|
| `path` | string | Image file relative to the dataset root (POSIX) |
| `sha256` | string | sha256 of the image file |
| `split` | string | `train`, `val` or `test` |
| `identity_id` | int64 | CelebA identity |
| `width`, `height` | int32 | Image size (always 178×218) |
| `source` | string | `celeba-aligned` |
| `license` | string | `CelebA: non-commercial research only, no redistribution` |
| `landmarks` | int32[10] | Aligned 5-point landmarks, order as in §4.1 |
| `source_file`, `source_row` | string, int64 | Mirror shard and row within it |
| `attr_<name>` | bool ×40 | CelebA attributes, lower-case names: `attr_male`, `attr_young`, `attr_blurry`, `attr_pale_skin`, … |

The first eight columns are required in every manifest (`facehall.data.manifest.REQUIRED_FIELDS`).
`read_manifest` checks the following on every read:
- column types and missing values;
- split names;
- unique relative paths;
- identity-disjoint splits.

### 5.2 `prepare_info.json`

It records:
- the creation time, source directory and git commit;
- the versions of facehall, numpy, pillow and pyarrow;
- the sha256 of every shard (§2);
- per-split counts (§3);
- two manifest checksums:
  - `content_sha256`: the column names and every row as compact JSON, in table order. It does
    not depend on the Parquet encoding;
  - `file_sha256`: the Parquet file bytes. These depend on the pyarrow version and writer
    settings.

## 6. How to reproduce

```bash
uv run python scripts/prepare_dataset.py \
    --src /mnt/storage/AI/Face/celeba_eurecom/data --out /mnt/storage/AI/Face/celeba
```

- Runtime: about 2 min on the dev machine (`PROJECT.md` §4); peak memory about 1.5 GB.
- Expected `manifest.content_sha256` in `prepare_info.json`:

  ```
  b6ff52ea1a7c28ba6b6af47791e53575cd5e0c68b54eddb68cb2ee6c6db94c91
  ```

- Reference run: commit `2dd74f2`, 2026-11-24T07:07:01+00:00, with pyarrow 26.0.1, pillow
  13.0.0 and numpy 2.5.4.
- Two runs on 2026-11-24 produced the same content checksum.
- The script refuses a non-empty output directory unless `--overwrite` is given. It checks the
  shard set, the official counts and identity disjointness before it writes anything.
- `--no-count-check` skips the count comparison, for partial copies of the mirror.
