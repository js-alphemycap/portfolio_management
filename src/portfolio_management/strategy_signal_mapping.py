from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from portfolio_management.helpers.config import BASE_DIR


DEFAULT_MAPPING_PATH = BASE_DIR / "configs" / "strategy_signal_mapping.yaml"


def load_strategy_signal_mapping(path: str | Path = DEFAULT_MAPPING_PATH) -> dict[str, str]:
    mapping_path = Path(path)
    raw = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
    strategies = raw.get("strategies")
    if not isinstance(strategies, dict):
        raise ValueError("strategy_signal_mapping.yaml must define a top-level 'strategies' mapping.")

    resolved: dict[str, str] = {}
    for strategy_id, config in strategies.items():
        if not isinstance(config, dict):
            raise ValueError(f"Mapping for strategy {strategy_id!r} must be a mapping.")
        slug = config.get("signal_strategy_slug")
        if not isinstance(slug, str) or not slug.strip():
            raise ValueError(f"Mapping for strategy {strategy_id!r} is missing signal_strategy_slug.")
        resolved[str(strategy_id)] = slug.strip()
    return resolved


__all__ = ["DEFAULT_MAPPING_PATH", "load_strategy_signal_mapping"]
