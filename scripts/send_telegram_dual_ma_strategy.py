#!/usr/bin/env python3
"""Send the reserve-portfolio dual-MA signal message to a Telegram chat."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from pathlib import Path

import requests

from price_data_infra.data import fetch_ohlcv
from portfolio_management.helpers.config import BASE_DIR
from portfolio_management.helpers.http import get_requests_verify
from portfolio_management.helpers.job_config import load_job_config
from portfolio_management.message_archive import archive_strategy_message
from portfolio_management.strategies.dual_ma_strategy_reserve_portfolio import (
    generate_reserve_portfolio_dual_ma_telegram_message,
    load_reserve_portfolio_dual_ma_config,
)


def _resolve_db_path(db_path_value: str | None) -> Path | None:
    if not db_path_value:
        return None
    path_candidate = Path(db_path_value)
    if not path_candidate.is_absolute():
        path_candidate = BASE_DIR / path_candidate
    return path_candidate


def _load_daily_ohlc(
    symbol: str,
    *,
    close_hour: int,
    start_date,
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


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> dict[str, object]:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, json=payload, timeout=15, verify=get_requests_verify())
    if not response.ok:
        try:
            info = response.json()
            desc = info.get("description")
        except Exception:  # pragma: no cover - best-effort diagnostics
            desc = response.text
        raise SystemExit(f"Telegram API error ({response.status_code}): {desc}")
    return response.json()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send the reserve-portfolio dual-MA daily signal message to Telegram."
    )
    parser.add_argument(
        "chat_id",
        help="Telegram chat ID (e.g., -1001234567890).",
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

    strategy_conf = load_reserve_portfolio_dual_ma_config(
        load_job_config("dual_ma_strategy", use_profile=False)
    )
    try:
        db_conf = load_job_config("market_data_access")
    except FileNotFoundError:
        db_conf = {}

    db_url = db_conf.get("db_url") if isinstance(db_conf, dict) else None
    db_path = _resolve_db_path(db_conf.get("db_path") if isinstance(db_conf, dict) else None)
    if args.db_url is not None:
        db_url = args.db_url
    if args.db_path is not None:
        db_path = _resolve_db_path(args.db_path)

    ohlc_btc = _load_daily_ohlc(
        strategy_conf.btc_symbol,
        close_hour=strategy_conf.close_hour,
        start_date=strategy_conf.start_date,
        db_url=db_url,
        db_path=db_path,
    )
    ohlc_eth = _load_daily_ohlc(
        strategy_conf.eth_symbol,
        close_hour=strategy_conf.close_hour,
        start_date=strategy_conf.start_date,
        db_url=db_url,
        db_path=db_path,
    )

    message = generate_reserve_portfolio_dual_ma_telegram_message(
        ohlc_btc=ohlc_btc,
        ohlc_eth=ohlc_eth,
        config=strategy_conf,
    )

    print("📨 Telegram dual MA strategy message:")
    print(message)
    archive_path = archive_strategy_message(strategy_slug="reserve_dual_ma", message=message)
    print(f"🗂️ Archived message -> {archive_path}")

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
