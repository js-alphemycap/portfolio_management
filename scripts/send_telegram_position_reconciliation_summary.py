#!/usr/bin/env python3
"""Send the latest two-day position reconciliation summary to Telegram."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from pathlib import Path

import pandas as pd
import requests


DEFAULT_DRY_RUN_CHAT_ID = "1782689756"
USD_ALERT_THRESHOLD = 10_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send the latest two-day position reconciliation summary to Telegram.")
    parser.add_argument("workbook_path", help="Path to the position monitoring workbook.")
    parser.add_argument("chat_id", nargs="?", help="Telegram chat ID.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message without sending it.",
    )
    return parser


def _format_day(row: pd.Series) -> str:
    date_label = str(row["date"])
    entries: list[str] = []
    for column in row.index:
        if column == "date":
            continue
        value = pd.to_numeric(row[column], errors="coerce")
        if pd.isna(value):
            continue
        integer_value = int(round(float(value)))
        if column == "USD" and abs(integer_value) <= USD_ALERT_THRESHOLD:
            continue
        if integer_value != 0:
            entries.append(f"{column} {integer_value:+d}")
    if not entries:
        return f"{date_label}: no reconciliation breaks"
    return f"{date_label}: " + " | ".join(entries)


def _build_message(workbook_path: Path) -> str:
    reconciliation = pd.read_excel(workbook_path, sheet_name="reconciliation")
    if reconciliation.empty:
        raise SystemExit(f"Reconciliation sheet is empty: {workbook_path}")
    latest = reconciliation.tail(2).copy()
    lines = [
        "Position reconciliation check",
        "",
        *[_format_day(row) for _, row in latest.iterrows()],
    ]
    return "\n".join(lines)


def _send_telegram_message(bot_token: str, chat_id: str, message: str) -> dict[str, object]:
    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": message},
        timeout=30,
    )
    if not response.ok:
        try:
            info = response.json()
            desc = info.get("description")
        except Exception:
            desc = response.text
        raise SystemExit(f"Telegram API error ({response.status_code}): {desc}")
    return response.json()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    workbook_path = Path(args.workbook_path)
    message = _build_message(workbook_path)
    print("📨 Telegram position reconciliation summary:")
    print(message)

    effective_chat_id = args.chat_id or DEFAULT_DRY_RUN_CHAT_ID
    if args.dry_run:
        print(f"Dry-run chat_id: {effective_chat_id}")
        print("Dry-run mode enabled; message not sent.")
        return

    if not args.chat_id:
        parser.error("chat_id is required unless --dry-run is used.")

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        parser.error("Telegram bot token is required via TELEGRAM_BOT_TOKEN environment variable.")

    response = _send_telegram_message(bot_token, args.chat_id, message)
    print("✅ Message sent. Telegram response:")
    print(response)


if __name__ == "__main__":
    main()
