from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

from portfolio_management.helpers.http import get_requests_verify
from portfolio_management.message_archive import archive_strategy_message


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> dict[str, object]:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, json=payload, timeout=15, verify=get_requests_verify())
    if not response.ok:
        try:
            info = response.json()
            desc = info.get("description")
        except Exception:
            desc = response.text
        raise SystemExit(f"Telegram API error ({response.status_code}): {desc}")
    return response.json()


def send_telegram_document(
    bot_token: str,
    chat_id: str,
    document_path: str | Path,
    *,
    caption: str | None = None,
) -> dict[str, object]:
    path = Path(document_path)
    if not path.exists():
        raise SystemExit(f"Document path does not exist: {path}")
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    with path.open("rb") as handle:
        files = {"document": (path.name, handle)}
        response = requests.post(
            url,
            data=data,
            files=files,
            timeout=30,
            verify=get_requests_verify(),
        )
    if not response.ok:
        try:
            info = response.json()
            desc = info.get("description")
        except Exception:
            desc = response.text
        raise SystemExit(f"Telegram API error ({response.status_code}): {desc}")
    return response.json()


def emit_telegram_message(
    *,
    parser: argparse.ArgumentParser,
    chat_id: str | None,
    dry_run_chat_id: str | None,
    dry_run: bool,
    message_label: str,
    strategy_slug: str,
    message: str,
) -> None:
    print(message_label)
    print(message)
    archive_path = archive_strategy_message(strategy_slug=strategy_slug, message=message)
    print(f"🗂️ Archived message -> {archive_path}")

    if dry_run:
        effective_chat_id = chat_id or dry_run_chat_id
        if effective_chat_id:
            print(f"Dry-run chat_id: {effective_chat_id}")
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


def emit_telegram_document(
    *,
    parser: argparse.ArgumentParser,
    chat_id: str | None,
    dry_run_chat_id: str | None,
    dry_run: bool,
    document_path: str | Path,
    caption: str | None,
) -> None:
    path = Path(document_path)
    print(f"Attachment: {path}")
    if caption:
        print(f"Caption: {caption}")

    if dry_run:
        effective_chat_id = chat_id or dry_run_chat_id
        if effective_chat_id:
            print(f"Dry-run chat_id: {effective_chat_id}")
        print("Dry-run mode enabled; document not sent.")
        return

    if not chat_id:
        parser.error("chat_id is required unless --dry-run uses configured telegram.dry_run_chat_id.")

    bot_token: str | None = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        parser.error("Telegram bot token is required via TELEGRAM_BOT_TOKEN environment variable.")

    response = send_telegram_document(bot_token, chat_id, path, caption=caption)
    print("✅ Document sent. Telegram response:")
    print(response)


__all__ = [
    "emit_telegram_document",
    "emit_telegram_message",
    "send_telegram_document",
    "send_telegram_message",
]
