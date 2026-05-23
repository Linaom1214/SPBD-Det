from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value


def _to_config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _to_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_config(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _to_config(data)


def merge_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    cfg = copy.deepcopy(cfg)
    if not overrides:
        return cfg
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or target[part] is None:
                target[part] = Config()
            target = target[part]
        target[parts[-1]] = _to_config(value)
    return cfg
