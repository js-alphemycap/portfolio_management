#!/usr/bin/env python3
"""Send the HYPE/ETH rotation signal message to a Telegram chat."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from datetime import datetime, timezone

from portfolio_management.helpers.job_config import load_job_config
from portfolio_management.market_data import load_daily_close, resolve_db_path
from portfolio_management.telegram_delivery import emit_telegram_message
from portfolio_management.strategies.hype_eth_rotation_strategy import (
    generate_hype_eth_rotation_snapshot,
    load_hype_eth_rotation_config,
)
from portfolio_management.strategies.hype_eth_rotation_strategy_telegram import (
    build_hype_eth_rotation_telegram_message,
)

FULL_HISTORY_START_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc)


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
    db_path = resolve_db_path(args.db_path) if args.db_path is not None else None

    hype_close = load_daily_close(
        strategy_conf.hype_symbol,
        close_hour=strategy_conf.close_hour,
        start_date=FULL_HISTORY_START_DATE,
        db_url=db_url,
        db_path=db_path,
    )
    eth_close = load_daily_close(
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

    emit_telegram_message(
        parser=parser,
        chat_id=chat_id,
        dry_run_chat_id=dry_run_chat_id,
        dry_run=args.dry_run,
        message_label="📨 Telegram HYPE/ETH rotation strategy message:",
        strategy_slug="hype_eth_rotation",
        message=message,
    )


if __name__ == "__main__":
    main()
