import facehall


def test_package_has_version() -> None:
    assert isinstance(facehall.__version__, str)
    assert facehall.__version__


def test_the_package_version_is_the_project_version() -> None:
    """`__version__` and `pyproject.toml` must name the same release (CHANGELOG.md)."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    assert facehall.__version__ == project["version"]
