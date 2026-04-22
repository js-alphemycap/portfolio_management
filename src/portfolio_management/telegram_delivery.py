"""Lightweight Telegram message sender."""
from __future__ import annotations

import requests


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> dict[str, object]:
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
