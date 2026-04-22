#!/usr/bin/env python3
"""Send the reserve-portfolio dual-MA signal message to a Telegram chat."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from portfolio_management.helpers.job_config import load_job_config
from portfolio_management.market_data import load_daily_ohlc, resolve_db_path
from portfolio_management.telegram_delivery import emit_telegram_message
from portfolio_management.strategies.dual_ma_strategy_reserve_portfolio import (
    build_reserve_portfolio_asset_telegram_message,
    generate_reserve_portfolio_snapshot,
    load_reserve_portfolio_dual_ma_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send the reserve-portfolio dual-MA daily signal message to Telegram."
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

    strategy_conf = load_reserve_portfolio_dual_ma_config(
        load_job_config("dual_ma_strategy", use_profile=False)
    )
    db_url = args.db_url
    db_path = resolve_db_path(args.db_path) if args.db_path is not None else None

    ohlc_btc = load_daily_ohlc(
        strategy_conf.btc_symbol,
        close_hour=strategy_conf.close_hour,
        start_date=strategy_conf.start_date,
        db_url=db_url,
        db_path=db_path,
    )
    ohlc_eth = load_daily_ohlc(
        strategy_conf.eth_symbol,
        close_hour=strategy_conf.close_hour,
        start_date=strategy_conf.start_date,
        db_url=db_url,
        db_path=db_path,
    )

    snapshot = generate_reserve_portfolio_snapshot(
        ohlc_btc=ohlc_btc,
        ohlc_eth=ohlc_eth,
        config=strategy_conf,
    )
    btc_message = build_reserve_portfolio_asset_telegram_message(snapshot, asset="BTC")
    emit_telegram_message(
        parser=parser,
        chat_id=args.chat_id,
        dry_run_chat_id=None,
        dry_run=args.dry_run,
        message_label="📨 Telegram reserve BTC strategy message:",
        strategy_slug="reserve_dual_ma_btc",
        message=btc_message,
    )
    eth_message = build_reserve_portfolio_asset_telegram_message(snapshot, asset="ETH")
    emit_telegram_message(
        parser=parser,
        chat_id=args.chat_id,
        dry_run_chat_id=None,
        dry_run=args.dry_run,
        message_label="📨 Telegram reserve ETH strategy message:",
        strategy_slug="reserve_dual_ma_eth",
        message=eth_message,
    )


if __name__ == "__main__":
    main()
