#!/usr/bin/env python3
"""Send the HYPE/ETH rotation signal message to a Telegram chat."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

from price_data_infra.data import fetch_ohlcv
from portfolio_management.helpers.config import BASE_DIR
from portfolio_management.helpers.http import get_requests_verify
from portfolio_management.helpers.job_config import load_job_config
from portfolio_management.message_archive import archive_strategy_message
from portfolio_management.strategies.hype_eth_rotation_strategy import (
    generate_hype_eth_rotation_snapshot,
    load_hype_eth_rotation_config,
)
from portfolio_management.strategies.hype_eth_rotation_strategy_telegram import (
    build_hype_eth_rotation_telegram_message,
)

FULL_HISTORY_START_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc)


def _resolve_db_path(db_path_value: str | None) -> Path | None:
    if not db_path_value:
        return None
    path_candidate = Path(db_path_value)
    if not path_candidate.is_absolute():
        path_candidate = BASE_DIR / path_candidate
    return path_candidate


def _load_daily_close(
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
        description="Send the HYPE/ETH rotation daily signal message to Telegram."
    )
    parser.add_argument(
        "chat_id",
        nargs="?",
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

    strategy_raw = load_job_config("hype_eth_rotation_strategy")
    strategy_conf = load_hype_eth_rotation_config(strategy_raw)
    db_url = args.db_url
    db_path = _resolve_db_path(args.db_path) if args.db_path is not None else None

    hype_close = _load_daily_close(
        strategy_conf.hype_symbol,
        close_hour=strategy_conf.close_hour,
        start_date=FULL_HISTORY_START_DATE,
        db_url=db_url,
        db_path=db_path,
    )
    eth_close = _load_daily_close(
        strategy_conf.eth_symbol,
        close_hour=strategy_conf.close_hour,
        start_date=FULL_HISTORY_START_DATE,
        db_url=db_url,
        db_path=db_path,
    )

    snapshot = generate_hype_eth_rotation_snapshot(
        hype_close=hype_close,
        eth_close=eth_close,
        config=strategy_conf,
    )
    message = build_hype_eth_rotation_telegram_message(
        snapshot=snapshot,
        config=strategy_conf,
    )

    telegram_conf = strategy_raw.get("telegram", {}) if isinstance(strategy_raw, dict) else {}
    dry_run_chat_id = (
        telegram_conf.get("dry_run_chat_id")
        if isinstance(telegram_conf, dict)
        else None
    )
    chat_id = args.chat_id

    print("📨 Telegram HYPE/ETH rotation strategy message:")
    print(message)
    archive_path = archive_strategy_message(strategy_slug="hype_eth_rotation", message=message)
    print(f"🗂️ Archived message -> {archive_path}")

    if args.dry_run:
        if not chat_id:
            chat_id = dry_run_chat_id
        if chat_id:
            print(f"Dry-run chat_id: {chat_id}")
        print("Dry-run mode enabled; message not sent.")
        return

    if not chat_id:
        parser.error("chat_id is required unless --dry-run uses configured telegram.dry_run_chat_id.")

    bot_token: str | None = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        parser.error("Telegram bot token is required via TELEGRAM_BOT_TOKEN environment variable.")

    response = send_telegram_message(bot_token, chat_id, message)
    print("✅ Message sent. Telegram response:")
    print(response)


if __name__ == "__main__":
    main()
