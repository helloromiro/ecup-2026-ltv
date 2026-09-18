from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_shared_config() -> dict[str, Any]:
    """configs/data.yaml + configs/features.yaml + configs/cv.yaml: the fixed
    problem setup (data location, feature definitions, CV scheme) shared by
    every experiment, independent of which model config is being run."""
    configs_dir = REPO_ROOT / "configs"
    cfg: dict[str, Any] = {}
    for shared in ("data.yaml", "features.yaml", "cv.yaml"):
        cfg.update(_load_yaml(configs_dir / shared))
    return cfg


def load_config(train_config_name: str) -> dict[str, Any]:
    """Merge the shared configs with a configs/<train_config_name>.yaml
    experiment config. The experiment config is the only thing that changes
    between v1/v2/v3 runs."""
    cfg = load_shared_config()
    configs_dir = REPO_ROOT / "configs"
    train_path = configs_dir / train_config_name
    if not train_path.suffix:
        train_path = train_path.with_suffix(".yaml")
    train_cfg = _load_yaml(train_path)
    cfg["experiment"] = train_cfg
    cfg["_train_config_path"] = str(train_path.relative_to(REPO_ROOT))
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    return REPO_ROOT / cfg[key]
