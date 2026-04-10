from __future__ import annotations

from datetime import datetime
from pathlib import Path

from portfolio_management.bootstrap_paths import ensure_repo_paths

ensure_repo_paths()

from price_data_infra.data import fetch_ohlcv

from portfolio_management.helpers.config import BASE_DIR


def resolve_db_path(db_path_value: str | None) -> Path | None:
    if not db_path_value:
        return None
    path_candidate = Path(db_path_value)
    if not path_candidate.is_absolute():
        path_candidate = BASE_DIR / path_candidate
    return path_candidate


def load_daily_close(
    symbol: str,
    *,
    close_hour: int,
    start_date: datetime | None,
    db_url: str | None,
    db_path: Path | None,
):
    df = fetch_ohlcv(
        symbol,
        frequency="daily",
        close_hour=close_hour,
        start=start_date,
        db_url=db_url,
        db_path=db_path,
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbol}.")
    if "close" not in df.columns:
        raise ValueError(f"{symbol} data missing required column: close")
    return df["close"].astype(float).copy()


def load_daily_ohlc(
    symbol: str,
    *,
    close_hour: int,
    start_date: datetime | None,
    db_url: str | None,
    db_path: Path | None,
):
    df = fetch_ohlcv(
        symbol,
        frequency="daily",
        close_hour=close_hour,
        start=start_date,
        db_url=db_url,
        db_path=db_path,
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbol}.")
    missing = {"high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"{symbol} data missing required columns: {missing}")
    return df[["high", "low", "close"]].copy()


__all__ = ["resolve_db_path", "load_daily_close", "load_daily_ohlc"]
