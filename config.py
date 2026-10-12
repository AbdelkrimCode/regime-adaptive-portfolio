import yaml
import functools
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    """Load config.yaml with every entry under `paths` made absolute.

    Paths in the file are relative to the project root, so scripts work from any
    working directory.
    """
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {key: str(PROJECT_ROOT / value) for key, value in cfg["paths"].items()}
    return cfg
