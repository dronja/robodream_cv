# Decisions

Log of project decisions: date, decision, reason. Open questions are marked **OPEN**.

## 2026-10-24

### D1 — Scenario and audience
**Decision:** offline restoration of photos and video frames.
**Reason:** default from the project plan; matches the README use cases and keeps the pipeline simple.

### D2 — Commercial vs. non-commercial use
**Decision:** non-commercial / research use.
**Reason:** the existing data (CelebA, FFHQ) and candidate models (ArcFace weights from InsightFace) are non-commercial only.
**Follow-ups:**
- Images from "Internet sources" in the legacy dataset have unknown origin and consent. They are excluded from the test set and documented in `docs/DATA.md`.
- If the use case changes to commercial, CelebA, FFHQ and NC-licensed models must be replaced (see `docs/THIRD_PARTY_LICENSES.md`).

### D3 — Framework
**Decision:** PyTorch. Keras code is kept in `legacy/` for reference only.
**Reason:** one framework, easier ONNX export, the legacy SRGAN generator is already PyTorch.

### D4 — Scales and resolution
**Decision:** x4 and x8 on aligned 128x128 crops; 512 px later.
**Reason:** 96 px targets are too small for a current product; 512 is deferred until the 128 px pipeline works.
**OPEN:** the legacy dataset is 89x109 px, so a 128 px HR target built from it would be an upscale, which violates the "HR must not be an upscale" rule. An HR source with resolution >= 128 px (and a compatible license) is needed before Phase 1, or the target must be reduced for the legacy data.

### D5 — Deployment
**Decision:** CLI + REST API (FastAPI) in Docker (CPU and GPU images).
**Reason:** default from the project plan.

### D6 — Training location
**Decision:** local training. Configs target 8 GB VRAM, with an optional larger profile.
**Reason:** the project plan assumes a laptop GPU (RTX 4070 Laptop, 8 GB); the current dev machine has an RTX 6000 Ada (48 GB), so the 8 GB profile stays the portable baseline.

### D7 — Weights in git
**Decision:** do not rewrite git history. Legacy files move to `legacy/` with `git mv`; weights are removed from HEAD and published in GitHub Releases only after separate explicit confirmation.
**Reason:** rewriting history is risky and unnecessary; removing weights before they are published elsewhere could lose the only convenient copy.

### Tooling
**Decision:** package name `facehall`; `uv` for environments (installed to `~/.local/bin`, uv 0.13.5); `requires-python >=3.11`, development on Python 3.12.
**Reason:** Python 3.14 (system default) is risky for the PyTorch ecosystem.
