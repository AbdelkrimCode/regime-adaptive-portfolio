import yaml
import functools
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"

@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)