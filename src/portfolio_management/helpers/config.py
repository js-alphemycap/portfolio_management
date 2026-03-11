"""Shared environment handling for the price_data ingestion toolkit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _discover_base_dir() -> Path:
    """Locate the project root by walking up until we find a marker file."""
    path = Path(__file__).resolve()
    for candidate in path.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    # Fallback to the package directory if no marker found.
    return path.parents[2]


BASE_DIR = _discover_base_dir()
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = DATA_DIR / "databases"
LOG_DIR = DATA_DIR / "logs"
DEFAULT_DB_PATH = DATABASE_DIR / "ohlcv.db"

# Load environment variables from a local .env if present.
load_dotenv(BASE_DIR / ".env", override=False)


def _get_symbol_list() -> list[str]:
    """Return the list of symbols from ENV or default set."""
    raw = os.getenv("SYMBOLS")
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD"]


@dataclass(frozen=True)
class ApiConfig:
    coindesk_api_key: str = os.getenv("COINDESK_API_KEY", "")


@dataclass(frozen=True)
class EmailConfig:
    tenant_id: str | None = os.getenv("MS_TENANT_ID")
    client_id: str | None = os.getenv("MS_CLIENT_ID")
    client_secret: str | None = os.getenv("MS_CLIENT_SECRET")
    sender: str | None = os.getenv("MS_SENDER")
    recipient: str | None = os.getenv("MS_RECIPIENT")


def _resolve_path(value: Path | str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


@dataclass(frozen=True)
class PipelineConfig:
    symbols: tuple[str, ...] = tuple(_get_symbol_list())
    granularity: str = os.getenv("GRANULARITY", "hours")
    overlap_days: int = int(os.getenv("OVERLAP_DAYS", "2"))
    db_url: str | None = os.getenv("DB_URL")
    db_path: Path = _resolve_path(os.getenv("DB_PATH", DEFAULT_DB_PATH))


API_CONFIG = ApiConfig()
EMAIL_CONFIG = EmailConfig()
PIPELINE_CONFIG = PipelineConfig()
