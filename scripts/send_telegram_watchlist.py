#!/usr/bin/env python3
"""Send a 24h watchlist return summary to a Telegram chat."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import requests

from price_data_infra.data import fetch_ohlcv
from price_data_infra.helpers.job_config import load_job_config as load_price_data_job_config
from portfolio_management.helpers.config import BASE_DIR
from portfolio_management.helpers.http import get_requests_verify


def _resolve_db_path(db_path_value: str | None) -> Path | None:
    if not db_path_value:
        return None
    path_candidate = Path(db_path_value)
    if not path_candidate.is_absolute():
        path_candidate = BASE_DIR / path_candidate
    return path_candidate


def _normalize_symbol(symbol: str) -> str:
    base = symbol.split("-")[0]
    return base.upper()


def _compute_symbol_return(
    symbol: str,
    hours: int,
    *,
    db_url: str | None,
    db_path: Path | None,
) -> dict[str, object] | None:
    lookback = timedelta(hours=hours + 6)
    start = datetime.now(timezone.utc) - lookback
    df = fetch_ohlcv(
        symbol,
        frequency="hourly",
        start=start,
        db_url=db_url,
        db_path=db_path,
    )
    if df.empty or "close" not in df.columns:
        return None

    df = df.dropna(subset=["close"])
    if df.empty:
        return None

    latest_ts = df.index.max()
    latest_close = float(df.loc[latest_ts, "close"])
    target_ts = latest_ts - timedelta(hours=hours)
    history = df[df.index <= target_ts]
    if history.empty:
        return None
    past_close = float(history.iloc[-1]["close"])
    if past_close == 0:
        return None

    pct_change = (latest_close / past_close - 1.0) * 100.0
    return {
        "symbol": symbol,
        "label": _normalize_symbol(symbol),
        "latest_ts": latest_ts,
        "pct_change": pct_change,
    }


def build_watchlist_lines(
    symbols: Sequence[str],
    hours: int,
    *,
    db_url: str | None,
    db_path: Path | None,
) -> list[str]:
    results: list[dict[str, object]] = []
    for symbol in symbols:
        entry = _compute_symbol_return(symbol, hours, db_url=db_url, db_path=db_path)
        if entry:
            results.append(entry)

    if not results:
        raise ValueError("No watchlist data available for the requested window.")

    results.sort(key=lambda item: item["pct_change"], reverse=True)

    lines = ["Watchlist 24h", ""]
    for item in results:
        pct = float(item["pct_change"])
        sign = "+" if pct >= 0 else ""
        emoji = "🟢" if pct >= 0 else "🔻"
        lines.append(f"{item['label']}: {sign}{pct:.2f}% {emoji}")
    return lines


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> dict[str, object]:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
    }
    response = requests.post(url, json=payload, timeout=15, verify=get_requests_verify())
    if not response.ok:
        try:
            info = response.json()
            desc = info.get("description")
        except Exception:  # pragma: no cover - best-effort diagnostics
            desc = response.text
        raise SystemExit(f"Telegram API error ({response.status_code}): {desc}")
    return response.json()


def _resolve_symbols(cli_symbols: Iterable[str], default_symbols: Sequence[str]) -> list[str]:
    if cli_symbols:
        return [symbol.strip() for symbol in cli_symbols if symbol.strip()]
    return list(default_symbols)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a Telegram message with 24h returns for a symbol watchlist."
    )
    parser.add_argument(
        "chat_id",
        help="Telegram chat ID (e.g., -1001234567890).",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        default=[],
        help="Symbol to include (repeat for multiple). Defaults to the hourly job symbols.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of trailing hours used to compute returns.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=("local", "vm"),
        help="Job profile to use (local or vm).",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Optional database URL override (e.g. postgresql://...).",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional SQLite path override (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message without sending it to Telegram.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.environ["JOB_PROFILE"] = args.profile

    db_url = args.db_url
    db_path = _resolve_db_path(args.db_path) if args.db_path is not None else None
    price_data_conf = load_price_data_job_config("hourly_ohlcv")
    symbols = _resolve_symbols(args.symbols, price_data_conf.get("symbols", []))
    if not symbols:
        parser.error("No symbols configured. Provide --symbol arguments or update the job config.")

    message_lines = build_watchlist_lines(
        symbols,
        args.hours,
        db_url=db_url,
        db_path=db_path,
    )
    message = "\n".join(message_lines)

    print("📨 Telegram watchlist message:")
    print(message)

    if args.dry_run:
        print("Dry-run mode enabled; message not sent.")
        return

    bot_token: str | None = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        parser.error("Telegram bot token is required via TELEGRAM_BOT_TOKEN environment variable.")

    response = send_telegram_message(bot_token, args.chat_id, message)
    print("✅ Message sent. Telegram response:")
    print(response)


if __name__ == "__main__":
    main()
