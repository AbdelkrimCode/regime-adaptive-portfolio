from pathlib import Path

from config import load_config, PROJECT_ROOT


def test_config_paths_are_absolute_under_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    load_config.cache_clear()
    try:
        paths = load_config()["paths"]
    finally:
        load_config.cache_clear()

    for key, value in paths.items():
        path = Path(value)
        assert path.is_absolute(), key
        assert PROJECT_ROOT in path.parents, key
    assert Path(paths["returns"]) == PROJECT_ROOT / "data" / "processed" / "returns.parquet"
