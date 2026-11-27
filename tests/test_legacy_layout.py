import json
from pathlib import Path

import pytest

LEGACY = Path(__file__).resolve().parents[1] / "legacy"


@pytest.mark.parametrize(
    "name", ["FH.ipynb", "FNet_24.keras", "CNet_24.keras", "SRgan_20.pth", "README.md"]
)
def test_legacy_file_present(name: str) -> None:
    assert (LEGACY / name).is_file()


def test_legacy_notebook_holds_no_images() -> None:
    """Its figures showed a real person's face from the legacy dataset (D40)."""
    notebook = json.loads((LEGACY / "FH.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        assert not cell.get("attachments")
        for output in cell.get("outputs", []):
            assert not any(kind.startswith("image/") for kind in output.get("data", {}))
