"""Utilities for loading job-specific configuration files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import BASE_DIR


CONFIG_ROOT = BASE_DIR / "configs" / "jobs"

_PROFILE_KEYS = ("profiles", "profile_overrides")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_job_config(job_name: str, *, use_profile: bool = True) -> dict[str, Any]:
    """Load the YAML configuration for a given job.

    If the environment variable JOB_PROFILE is set (e.g., "vm" or "local"),
    attempt to load a profile-specific file named
    "{job_name}.{JOB_PROFILE}.yaml" from configs/jobs first; if it does not
    exist, fall back to the default "{job_name}.yaml".

    Additionally, a single config file may contain a top-level mapping under
    'profiles' (or legacy 'profile_overrides') keyed by profile name. When
    present and JOB_PROFILE is set, the matching mapping is deep-merged into
    the base config.
    """
    profile = os.getenv("JOB_PROFILE") if use_profile else None
    candidates: list[Path] = []
    if profile:
        candidates.append(CONFIG_ROOT / f"{job_name}.{profile}.yaml")
    candidates.append(CONFIG_ROOT / f"{job_name}.yaml")

    path: Path | None = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None:
        expected = " or ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Job config '{job_name}' not found. Expected {expected}."
        )

    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Job config '{job_name}' must contain a mapping at the root.")

    config = dict(data)
    if profile:
        profile_block: Mapping[str, Any] | None = None
        for key in _PROFILE_KEYS:
            raw_profiles = config.get(key)
            if isinstance(raw_profiles, Mapping):
                candidate = raw_profiles.get(profile)
                if isinstance(candidate, Mapping):
                    profile_block = candidate
                    break
        if profile_block:
            config = _deep_merge(config, profile_block)
        for key in _PROFILE_KEYS:
            config.pop(key, None)

    return config


def dump_job_config(job_name: str, data: Mapping[str, Any]) -> None:
    """Write a job configuration to disk (convenience for bootstrapping)."""
    path = CONFIG_ROOT / f"{job_name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(json.loads(json.dumps(data)), file, sort_keys=True)
