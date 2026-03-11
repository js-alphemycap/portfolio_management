#!/usr/bin/env python3
"""Send the alts SMA monitoring matrix image to a Telegram chat."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from portfolio_management.helpers.config import BASE_DIR
from portfolio_management.helpers.http import get_requests_verify
from portfolio_management.helpers.job_config import load_job_config

from alts_sma_monitoring import build_status_frame, render_status_matrix_png


def _resolve_db_path(db_path_value: str | None) -> Path | None:
    if not db_path_value:
        return None
    path_candidate = Path(db_path_value)
    if not path_candidate.is_absolute():
        path_candidate = BASE_DIR / path_candidate
    return path_candidate


def _parse_start_date(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def send_telegram_photo(
    *,
    bot_token: str,
    chat_id: str,
    photo_bytes: bytes,
    filename: str,
    caption: str,
) -> dict[str, object]:
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {"photo": (filename, photo_bytes, "image/png")}
    data = {"chat_id": chat_id, "caption": caption}
    response = requests.post(url, data=data, files=files, timeout=30, verify=get_requests_verify())
    if not response.ok:
        try:
            info = response.json()
            desc = info.get("description")
        except Exception:  # pragma: no cover - best-effort diagnostics
            desc = response.text
        if response.status_code == 404:
            raise SystemExit(
                "Telegram API returned 404 Not Found. This usually means TELEGRAM_BOT_TOKEN is missing/invalid."
            )
        raise SystemExit(f"Telegram API error ({response.status_code}): {desc}")
    return response.json()


def validate_telegram_token(bot_token: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/getMe"
    response = requests.get(url, timeout=15, verify=get_requests_verify())
    if response.status_code == 404:
        raise SystemExit(
            "Telegram API returned 404 Not Found for getMe. TELEGRAM_BOT_TOKEN is missing/invalid."
        )
    if not response.ok:
        raise SystemExit(f"Telegram token check failed ({response.status_code}): {response.text[:2000]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send the alts SMA monitoring matrix image to a Telegram chat."
    )
    parser.add_argument(
        "chat_id",
        help="Telegram chat ID (e.g., -1001234567890).",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=("local", "vm"),
        help="Job profile to use (local or vm) for DB settings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and print the caption without sending to Telegram.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.environ["JOB_PROFILE"] = args.profile
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    job_conf = load_job_config("alts_sma_monitoring", use_profile=False)
    alts = [str(item).strip() for item in job_conf.get("alts", []) if str(item).strip()]
    if not alts:
        raise SystemExit("alts_sma_monitoring config has no alts.")

    window = int(job_conf.get("window", 50))
    close_hour = int(job_conf.get("close_hour", 12))
    start_date = _parse_start_date(job_conf.get("start_date"))

    try:
        db_conf = load_job_config("market_data_access")
    except FileNotFoundError:
        db_conf = {}

    db_url = db_conf.get("db_url") if isinstance(db_conf, dict) else None
    db_path = _resolve_db_path(db_conf.get("db_path") if isinstance(db_conf, dict) else None)

    status, details = build_status_frame(
        alts=alts,
        window=window,
        close_hour=close_hour,
        start_date=start_date,
        db_url=db_url,
        db_path=db_path,
    )
    if details.empty:
        raise SystemExit("No status data produced; check DB contents and symbols.")

    as_of = pd.to_datetime(details["date"].max(), utc=True)
    png = render_status_matrix_png(status, window=window, as_of=as_of)

    caption = f"Alts SMA Monitoring (window={window}) — {as_of:%Y-%m-%d} UTC"

    print("📨 Telegram alts SMA monitoring caption:")
    print(caption)

    if args.dry_run:
        print(f"Dry-run mode enabled; image not sent. PNG bytes={len(png)}")
        return

    bot_token: str | None = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        parser.error("Telegram bot token is required via TELEGRAM_BOT_TOKEN environment variable.")

    validate_telegram_token(bot_token)

    response = send_telegram_photo(
        bot_token=bot_token,
        chat_id=args.chat_id,
        photo_bytes=png,
        filename=f"alts_sma_monitoring_{as_of:%Y%m%d}.png",
        caption=caption,
    )
    print("✅ Photo sent. Telegram response:")
    print(response)


if __name__ == "__main__":
    main()
