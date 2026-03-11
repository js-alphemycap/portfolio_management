#!/usr/bin/env python3
"""Send the SMA dashboard headline summary to a Telegram group."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import requests

try:
    import sma_dashboard as dashboard
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    if exc.name == "matplotlib":
        raise SystemExit(
            "matplotlib is required to load sma_dashboard utilities. "
            "Install it (see requirements.txt) before running this script."
        ) from exc
    raise
from portfolio_management.helpers.config import BASE_DIR
from portfolio_management.helpers.http import get_requests_verify
from portfolio_management.helpers.job_config import load_job_config


def _parse_asset_pairs(raw_values: Iterable[str]) -> list[tuple[str, int]]:
    if raw_values:
        return dashboard.parse_asset_pairs(raw_values)
    return list(dashboard.DEFAULT_ASSET_MA_PAIRS)


def _resolve_db_path(db_path_value: str | None) -> Path | None:
    if not db_path_value:
        return None
    path_candidate = Path(db_path_value)
    if not path_candidate.is_absolute():
        path_candidate = BASE_DIR / path_candidate
    return path_candidate


def build_summary_text(
    asset_ma_pairs: Sequence[tuple[str, int]],
    close_hour: int,
    start_date: datetime | None,
    *,
    db_url: str | None = None,
    db_path: Path | None = None,
) -> str:
    prices = dashboard._load_prices(  # pylint: disable=protected-access
        asset_ma_pairs,
        close_hour,
        start_date,
        db_url=db_url,
        db_path=db_path,
    )
    if prices.empty:
        raise ValueError("No price data available after applying filters.")

    changes: list[dict[str, object]] = []
    latest_levels: list[dict[str, object]] = []
    for symbol, window in asset_ma_pairs:
        series = prices[symbol].dropna()
        if series.empty:
            continue
        _, _, _, _, change_info, latest_snapshot = dashboard._compute_signal_metadata(  # pylint: disable=protected-access
            series, window
        )
        if change_info:
            changes.append(change_info)
        if latest_snapshot:
            latest_levels.append(latest_snapshot)

    date_line = dashboard._build_date_line(prices)  # pylint: disable=protected-access
    summary_line = dashboard._build_signal_summary(changes, prices)  # pylint: disable=protected-access
    levels_line = dashboard._build_levels_summary(latest_levels)  # pylint: disable=protected-access

    summary_lines = [line for line in (date_line, summary_line, levels_line) if line]
    summary_html = "<br/><br/>".join(summary_lines)
    return summary_html.replace("<br/>", "\n").strip()


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send the SMA dashboard headline summary to a Telegram group."
    )
    # Bot token is sourced from TELEGRAM_BOT_TOKEN env var only.
    # Chat ID must be provided as an argument (positional for clarity).
    parser.add_argument(
        "chat_id",
        help="Telegram chat ID (e.g., -1001234567890).",
    )
    parser.add_argument(
        "--asset",
        action="append",
        dest="assets",
        default=[],
        help="Asset/window pair as SYMBOL:WINDOW (repeatable). Defaults to the dashboard set.",
    )
    parser.add_argument(
        "--close-hour",
        type=int,
        default=11,
        help="Close hour passed to fetch_ohlcv (0-23, end-labelled).",
    )
    parser.add_argument(
        "--start-date",
        type=lambda value: datetime.fromisoformat(value).replace(tzinfo=timezone.utc),
        default=None,
        help="ISO timestamp (UTC assumed) for the earliest daily bar to include.",
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

    chat_id: str | None = args.chat_id

    os.environ["JOB_PROFILE"] = args.profile

    try:
        job_conf = load_job_config("market_data_access")
    except FileNotFoundError:
        job_conf = {}

    db_url = job_conf.get("db_url") if isinstance(job_conf, dict) else None
    db_path = _resolve_db_path(job_conf.get("db_path") if isinstance(job_conf, dict) else None)
    if args.db_url is not None:
        db_url = args.db_url
    if args.db_path is not None:
        db_path = _resolve_db_path(args.db_path)

    pairs = _parse_asset_pairs(args.assets)

    summary_text = build_summary_text(
        asset_ma_pairs=pairs,
        close_hour=args.close_hour,
        start_date=args.start_date,
        db_url=db_url,
        db_path=db_path,
    )

    print("📨 Telegram summary message:")
    print(summary_text)

    if args.dry_run:
        print("Dry-run mode enabled; message not sent.")
        return

    bot_token: str | None = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        parser.error("Telegram bot token is required via TELEGRAM_BOT_TOKEN environment variable.")

    response = send_telegram_message(bot_token, chat_id, summary_text)
    print("✅ Message sent. Telegram response:")
    print(response)


if __name__ == "__main__":
    main()
