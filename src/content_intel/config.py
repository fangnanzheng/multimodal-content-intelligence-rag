from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from content_intel.paths import CONFIG_DIR, ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "config.yaml")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def config_path(key: str) -> Path:
    cfg = load_config()
    section, name = key.split(".", 1)
    return resolve_path(cfg[section][name])

