# legacy/

Original research artifacts. **Read only** — do not modify or build on them.

| File | What it is |
|---|---|
| `FH.ipynb` | Original notebook: Keras FNet/CNet, PyTorch SRGAN generator, training and demo code |
| `FNet_24.keras` | Simplified SRResNet, 24x24 -> 96x96 (x4), Keras 3.9.2 |
| `CNet_24.keras` | Simplified C-Face, 24x24 -> 96x96 (x4) |
| `SRgan_20.pth` | PyTorch SRGAN generator `state_dict` (the notebook expects `best_generator_20.pth`) |

Known problems of this code are listed in `CLAUDE.md`, section 4.
The weights are planned to move to GitHub Releases (see `docs/DECISIONS.md`, D7).
