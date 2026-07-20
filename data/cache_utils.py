import json
import os


def meta_path_for(data_path: str) -> str:
    root, _ = os.path.splitext(data_path)
    return root + ".meta.json"


def is_cache_valid(data_path: str, expected_signature: dict) -> bool:
    meta_path = meta_path_for(data_path)
    if not os.path.exists(data_path) or not os.path.exists(meta_path):
        return False
    with open(meta_path, "r") as f:
        cached_signature = json.load(f)
    return cached_signature == expected_signature


def write_cache_signature(data_path: str, signature: dict) -> None:
    meta_path = meta_path_for(data_path)
    with open(meta_path, "w") as f:
        json.dump(signature, f)