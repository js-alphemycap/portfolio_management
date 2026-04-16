#!/usr/bin/env python3
"""Send a file attachment to a Telegram chat via the PM bot."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from portfolio_management.telegram_delivery import emit_telegram_document


DEFAULT_DRY_RUN_CHAT_ID = "1782689756"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a file attachment to Telegram.")
    parser.add_argument("document_path", help="Path to the file to send.")
    parser.add_argument("chat_id", nargs="?", help="Telegram chat ID.")
    parser.add_argument("--caption", default=None, help="Optional caption text.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved send without delivering it.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    emit_telegram_document(
        parser=parser,
        chat_id=args.chat_id,
        dry_run_chat_id=DEFAULT_DRY_RUN_CHAT_ID,
        dry_run=args.dry_run,
        document_path=Path(args.document_path),
        caption=args.caption,
    )


if __name__ == "__main__":
    main()
