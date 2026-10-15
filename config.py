import os
import yaml
import functools
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
# Set this environment variable to run the pipeline on another config, e.g. config_backcast.yaml.
CONFIG_ENV = "REGIME_CONFIG"


def config_path() -> Path:
    """The active config file: $REGIME_CONFIG if set (relative to the project root), else config.yaml."""
    name = os.environ.get(CONFIG_ENV)
    if not name:
        return CONFIG_PATH
    path = Path(name)
    return path if path.is_absolute() else PROJECT_ROOT / path


@functools.lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {key: str(PROJECT_ROOT / value) for key, value in cfg["paths"].items()}
    return cfg


def load_config() -> dict:
    """Load the active config with every entry under `paths` made absolute.

    Paths in the file are relative to the project root, so scripts work from any
    working directory.
    """
    return _load(str(config_path()))


load_config.cache_clear = _load.cache_clear
