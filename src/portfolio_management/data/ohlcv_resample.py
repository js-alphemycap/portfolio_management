"""Helpers for resampling stored OHLCV data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from .storage import OHLCV_TABLE_NAME, get_storage

Frequency = Literal["hourly", "daily", "weekly", "monthly"]
Weekday = Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
_VALID_WEEKDAYS: set[Weekday] = {"MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"}


def _load_hourly_frame(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    table_name: str = OHLCV_TABLE_NAME,
    limit: int = 10_000_000,
    db_url: str | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    resolved_db_path: Path | None
    if isinstance(db_path, str):
        resolved_db_path = Path(db_path)
    else:
        resolved_db_path = db_path

    with get_storage(db_url=db_url, db_path=resolved_db_path) as storage:
        rows = storage.fetch_rows(
            symbol=symbol,
            start=start,
            end=end,
            limit=limit,
            table_name=table_name,
        )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def _verify_hourly_continuity(df: pd.DataFrame) -> None:
    if df.empty:
        return
    diffs = df.index.to_series().diff().dropna()
    bad = diffs[diffs != pd.Timedelta(hours=1)]
    if not bad.empty:
        raise ValueError(
            "Hourly continuity check failed: found non-1h gaps.\n"
            f"Examples:\n{bad.head()}"
        )


def fetch_ohlcv(
    ticker: str,
    *,
    frequency: Frequency = "daily",
    close_hour: int = 0,
    week_cutoff: Weekday = "SUN",
    start: datetime | None = None,
    end: datetime | None = None,
    table_name: str = OHLCV_TABLE_NAME,
    limit: int = 10_000_000,
    db_url: str | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Fetch hourly OHLCV from storage and resample to daily/weekly/monthly using a rolling 24 hour window.

    Daily output is end-labelled: the index equals the close timestamp (e.g. 23:00 when close_hour=23).
    Weekly and monthly results inherit that end-labelled timestamp.
    """
    if frequency not in {"hourly", "daily", "weekly", "monthly"}:
        raise ValueError("frequency must be 'hourly', 'daily', 'weekly', or 'monthly'")
    if not (0 <= close_hour <= 23):
        raise ValueError("close_hour must be an integer in [0, 23]")

    df = _load_hourly_frame(
        ticker,
        start=start,
        end=end,
        table_name=table_name,
        limit=limit,
        db_url=db_url,
        db_path=db_path,
    )
    if df.empty:
        return df

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    _verify_hourly_continuity(df)

    if frequency == "hourly":
        return df.dropna()

    window = "24h"
    open_series = df["open"].rolling(window, closed="right").apply(lambda x: x.iloc[0], raw=False)
    high_series = df["high"].rolling(window, closed="right").max()
    low_series = df["low"].rolling(window, closed="right").min()
    close_series = df["close"].rolling(window, closed="right").apply(lambda x: x.iloc[-1], raw=False)
    volume_series = df["volume"].rolling(window, closed="right").sum()

    rolled = pd.concat(
        [open_series.rename("open"), high_series.rename("high"), low_series.rename("low"),
         close_series.rename("close"), volume_series.rename("volume")],
        axis=1,
    )

    full_window = df["close"].rolling(window, closed="right").count() == 24
    at_close = (
        (rolled.index.hour == close_hour)
        & (rolled.index.minute == 0)
        & (rolled.index.second == 0)
    )

    daily = rolled.loc[at_close & full_window].copy()
    daily.index.name = "timestamp"
    daily = daily.dropna()

    if frequency == "daily":
        return daily

    close_offset = pd.Timedelta(hours=close_hour)
    base_index = (daily.index - close_offset).normalize()
    daily_base = daily.copy()
    daily_base.index = base_index

    if frequency == "weekly":
        week_cutoff = week_cutoff.upper()
        if week_cutoff not in _VALID_WEEKDAYS:
            raise ValueError("week_cutoff must be one of MON..SUN")

        rule = f"W-{week_cutoff}"
        weekly = daily_base.resample(rule, label="right", closed="right").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        weekly.index = weekly.index + close_offset
        weekly.index.name = "timestamp"
        return weekly.dropna()

    monthly = daily_base.resample("M", label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    monthly.index = monthly.index + close_offset
    monthly.index.name = "timestamp"
    return monthly.dropna()


__all__ = ["fetch_ohlcv"]
