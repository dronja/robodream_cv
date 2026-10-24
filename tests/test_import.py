import facehall


def test_package_has_version() -> None:
    assert isinstance(facehall.__version__, str)
    assert facehall.__version__
