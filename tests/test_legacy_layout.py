from pathlib import Path

import pytest

LEGACY = Path(__file__).resolve().parents[1] / "legacy"


@pytest.mark.parametrize(
    "name", ["FH.ipynb", "FNet_24.keras", "CNet_24.keras", "SRgan_20.pth", "README.md"]
)
def test_legacy_file_present(name: str) -> None:
    assert (LEGACY / name).is_file()
