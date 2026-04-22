#!/usr/bin/env python
"""One-time YTD daily OHLCV export for selected symbols."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from price_data_infra.data import fetch_ohlcv


DEFAULT_SYMBOLS = ("HYPE-USD", "SOL-USD", "ETH-USD", "BTC-USD")


def _default_start_date() -> datetime:
    now = datetime.now(timezone.utc)
    return datetime(now.year, 1, 1, tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YTD daily OHLCV prices for selected symbols."
    )
    parser.add_argument(
        "profile",
        choices=("local", "vm"),
        help="Storage profile to read from.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Symbols to export.",
    )
    parser.add_argument(
        "--start-date",
        default=_default_start_date().date().isoformat(),
        help="UTC start date in YYYY-MM-DD format. Defaults to Jan 1 of the current year.",
    )
    parser.add_argument(
        "--close-hour",
        type=int,
        default=12,
        help="Daily candle close hour in UTC. Defaults to 12 to match strategy configs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/extract",
        help="Directory for exported CSV files.",
    )
    return parser.parse_args()


def _load_symbol_frame(
    symbol: str,
    *,
    profile: str,
    start_date: datetime,
    close_hour: int,
) -> pd.DataFrame:
    df = fetch_ohlcv(
        symbol,
        frequency="daily",
        close_hour=close_hour,
        start=start_date,
        profile=profile,
    ).copy()
    if df.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])

    df = df.reset_index()
    df.insert(0, "symbol", symbol)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]


def main() -> int:
    args = parse_args()
    os.environ["JOB_PROFILE"] = args.profile

    start_date = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        _load_symbol_frame(
            symbol,
            profile=args.profile,
            start_date=start_date,
            close_hour=args.close_hour,
        )
        for symbol in args.symbols
    ]
    combined = pd.concat(frames, ignore_index=True)

    as_of = datetime.now(timezone.utc).date().isoformat()
    long_path = output_dir / f"prices_ytd_{args.profile}_{as_of}.csv"
    wide_path = output_dir / f"prices_ytd_close_wide_{args.profile}_{as_of}.csv"

    combined.to_csv(long_path, index=False)

    wide = combined.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    wide.to_csv(wide_path)

    print(long_path)
    print(wide_path)
    print(f"rows={len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
