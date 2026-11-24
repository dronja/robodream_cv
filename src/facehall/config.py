"""Typed YAML configs: a dataclass schema merged with a YAML file through OmegaConf.

Unknown keys and wrong types are errors, so typos in a config fail early. YAML values may use
OmegaConf interpolation, e.g. ``${oc.env:FACEHALL_DATA,/default/path}``.
"""

from pathlib import Path

from omegaconf import OmegaConf


def load_config[T](path: Path | str, schema: type[T]) -> T:
    """Load ``path`` into an instance of the dataclass ``schema``, resolving interpolations."""
    merged = OmegaConf.merge(OmegaConf.structured(schema), OmegaConf.load(path))
    config = OmegaConf.to_object(merged)
    if not isinstance(config, schema):
        raise TypeError(f"{path} did not produce a {schema.__name__}")
    return config
