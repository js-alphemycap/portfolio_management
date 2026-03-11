#!/usr/bin/env python3
"""Send a Telegram message comparing UTC 12:00 vs 13:00 hourly closes."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import requests

from price_data_infra.data import fetch_ohlcv
from portfolio_management.helpers.config import BASE_DIR
from portfolio_management.helpers.http import get_requests_verify
from portfolio_management.helpers.job_config import load_job_config

SYMBOLS = ("BTC-USD", "ETH-USD", "SOL-USD")


def _resolve_db_path(db_path_value: str | None) -> Path | None:
    if not db_path_value:
        return None
    path_candidate = Path(db_path_value)
    if not path_candidate.is_absolute():
        path_candidate = BASE_DIR / path_candidate
    return path_candidate


def _fmt_price(value: float) -> str:
    x = float(value)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:,.2f}"
    return f"{x:,.4f}"


def _fmt_pct(value: float) -> str:
    pct = value * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def _symbol_label(symbol: str) -> str:
    return symbol.split("-")[0].upper()


def _load_hourly_close(
    symbol: str,
    *,
    start_date: datetime,
    db_url: str | None,
    db_path: Path | None,
):
    df = fetch_ohlcv(
        symbol,
        frequency="hourly",
        start=start_date,
        db_url=db_url,
        db_path=db_path,
    )
    if df.empty or "close" not in df.columns:
        raise ValueError(f"No hourly close data returned for {symbol}.")
    return df["close"].astype(float).copy()


def _find_latest_day_with_12_and_13(series_map: dict[str, object]) -> date:
    common_days: set[date] | None = None
    for _, series in series_map.items():
        s = series.dropna()
        day_hours: dict[date, set[int]] = {}
        for ts in s.index:
            d = ts.date()
            if d not in day_hours:
                day_hours[d] = set()
            day_hours[d].add(int(ts.hour))
        valid_days = {d for d, hours in day_hours.items() if 12 in hours and 13 in hours}
        if common_days is None:
            common_days = valid_days
        else:
            common_days = common_days.intersection(valid_days)
    if not common_days:
        raise ValueError("No common day found with both 12:00 and 13:00 closes for all symbols.")
    return max(common_days)


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def build_message(
    *,
    target_day: date | None,
    db_url: str | None,
    db_path: Path | None,
) -> str:
    lookback_start = datetime.now(timezone.utc) - timedelta(days=14)
    series_map = {
        symbol: _load_hourly_close(symbol, start_date=lookback_start, db_url=db_url, db_path=db_path)
        for symbol in SYMBOLS
    }

    day = target_day or _find_latest_day_with_12_and_13(series_map)
    ts12 = datetime.combine(day, time(12, 0), tzinfo=timezone.utc)
    ts13 = datetime.combine(day, time(13, 0), tzinfo=timezone.utc)

    lines = [
        f"UTC 12h vs 13h Price Check - {day.isoformat()}",
        "Comparing exact hourly closes:",
    ]
    for symbol in SYMBOLS:
        s = series_map[symbol]
        if ts12 not in s.index:
            raise ValueError(f"{symbol}: missing 12:00 close on {day.isoformat()}.")
        if ts13 not in s.index:
            raise ValueError(f"{symbol}: missing 13:00 close on {day.isoformat()}.")
        p12 = float(s.loc[ts12])
        p13 = float(s.loc[ts13])
        diff_abs = p13 - p12
        diff_pct = 0.0 if p12 == 0.0 else (p13 / p12 - 1.0)
        sign = "+" if diff_abs >= 0 else ""
        lines.append(
            f"- {_symbol_label(symbol)}: 12h={_fmt_price(p12)} | 13h={_fmt_price(p13)} | "
            f"delta={sign}{_fmt_price(diff_abs)} ({_fmt_pct(diff_pct)})"
        )
    return "\n".join(lines)


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
        description="Send a Telegram message comparing BTC/ETH/SOL hourly closes at UTC 12:00 vs 13:00."
    )
    parser.add_argument(
        "chat_id",
        help="Telegram chat ID (e.g., -1001234567890).",
    )
    parser.add_argument(
        "--date",
        type=_parse_day,
        default=None,
        help="Optional target date in YYYY-MM-DD. Defaults to latest common day with both 12h and 13h data.",
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

    message = build_message(
        target_day=args.date,
        db_url=db_url,
        db_path=db_path,
    )

    print("📨 Telegram 12h/13h check message:")
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
