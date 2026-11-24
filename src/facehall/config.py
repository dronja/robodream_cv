"""Typed YAML configs: a dataclass schema merged with a YAML file through OmegaConf.

Unknown keys and wrong types are errors, so typos in a config fail early. YAML values may use
OmegaConf interpolation, e.g. ``${oc.env:FACEHALL_DATA,/default/path}``.

A top-level ``include`` mapping fills whole sections from other YAML files, with paths relative
to the including file::

    include:
      data: ../data/celeba_128.yaml
    data:
      hflip: 0.0        # overrides the included value

Values are merged in this order, later ones winning: schema defaults, included files, the file
itself, then ``overrides`` (OmegaConf dotlist, e.g. ``["train.batch_size=32"]``).
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

INCLUDE = "include"


def load_config[T](path: Path | str, schema: type[T], overrides: Sequence[str] = ()) -> T:
    """Load ``path`` into an instance of the dataclass ``schema``, resolving interpolations."""
    path = Path(path)
    raw = OmegaConf.load(path)
    if not isinstance(raw, DictConfig):
        raise TypeError(f"{path} must hold a mapping")
    layers = [OmegaConf.structured(schema)]
    includes = raw.pop(INCLUDE, None)
    if includes is not None:
        if not isinstance(includes, DictConfig):
            raise TypeError(f"{path}: '{INCLUDE}' must map section names to files")
        for section, file in includes.items():
            layers.append(OmegaConf.create({section: OmegaConf.load(path.parent / str(file))}))
    layers += [raw, OmegaConf.from_dotlist(list(overrides))]
    config = OmegaConf.to_object(OmegaConf.merge(*layers))
    if not isinstance(config, schema):
        raise TypeError(f"{path} did not produce a {schema.__name__}")
    return config


def require_choice(value: str, options: Iterable[str], kind: str) -> None:
    """Raise ``ValueError`` unless ``value`` is one of ``options``."""
    if value not in options:
        raise ValueError(f"unknown {kind} {value!r}; expected one of {sorted(options)}")
