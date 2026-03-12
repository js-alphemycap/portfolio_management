from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path


def _discover_base_dir() -> Path:
    path = Path(__file__).resolve()
    for candidate in path.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    return path.parents[2]


BASE_DIR = _discover_base_dir()
ARCHIVE_DIR = BASE_DIR / "outputs" / "daily_strategy_messages"
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _archive_glob(*, strategy_slug: str, message_date: date) -> str:
    return f"{message_date.isoformat()}_*_{strategy_slug}.txt"


def infer_message_date(message: str) -> date:
    match = _DATE_RE.search(message)
    if not match:
        raise ValueError("Could not infer YYYY-MM-DD date from strategy message.")
    return date.fromisoformat(match.group(1))


def archive_strategy_message(*, strategy_slug: str, message: str) -> Path:
    message_date = infer_message_date(message)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%H-%M-%S")
    out_path = ARCHIVE_DIR / f"{message_date.isoformat()}_{timestamp}_{strategy_slug}.txt"
    out_path.write_text(message.strip() + "\n", encoding="utf-8")
    return out_path


def load_archived_strategy_message(*, strategy_slug: str, message_date: date) -> str:
    paths = sorted(ARCHIVE_DIR.glob(_archive_glob(strategy_slug=strategy_slug, message_date=message_date)))
    if not paths:
        raise FileNotFoundError(
            f"No archived message found for strategy={strategy_slug!r} on {message_date.isoformat()}."
        )
    latest = max(paths, key=lambda path: path.stat().st_mtime)
    return latest.read_text(encoding="utf-8").strip()


__all__ = [
    "ARCHIVE_DIR",
    "archive_strategy_message",
    "infer_message_date",
    "load_archived_strategy_message",
]
