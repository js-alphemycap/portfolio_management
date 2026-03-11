from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_management.helpers.config import BASE_DIR


@dataclass(frozen=True)
class SolEthTradeEvent:
    date: pd.Timestamp
    event: str
    cross_price: float


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = BASE_DIR / resolved
    return resolved


def _parse_event_date(value: Any, *, close_hour: int) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Trade log field 'date' must be a non-empty date string (YYYY-MM-DD).")
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    # Keep only the calendar date from the input, then apply configured close hour.
    day = ts.date()
    return pd.Timestamp(
        year=day.year,
        month=day.month,
        day=day.day,
        hour=int(close_hour),
        tz="UTC",
    )


def _to_float(value: Any, *, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Trade log field {field_name!r} must be numeric, got {value!r}.") from exc


def load_sol_eth_trade_log(
    path: str | Path,
    *,
    close_hour: int,
) -> tuple[list[SolEthTradeEvent], tuple[str, ...]]:
    file_path = _resolve_path(path)
    if not file_path.exists():
        return [], (f"Trade log not found at {file_path}; treating as empty.",)

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_events = payload.get("events", [])
    elif isinstance(payload, list):
        raw_events = payload
    else:
        raise ValueError("Trade log JSON must be a list or an object with an 'events' array.")

    if not isinstance(raw_events, list):
        raise ValueError("Trade log events must be a JSON array.")

    events: list[SolEthTradeEvent] = []
    for i, row in enumerate(raw_events):
        if not isinstance(row, dict):
            raise ValueError(f"Trade log row {i} must be an object.")
        event = str(row.get("event", "")).strip().upper()
        if event not in {"ENTRY", "EXIT"}:
            raise ValueError(f"Trade log row {i} has invalid event={event!r}; expected ENTRY or EXIT.")
        cross_price = _to_float(row.get("cross_price"), field_name="cross_price")
        if cross_price <= 0:
            raise ValueError(f"Trade log row {i} has non-positive cross_price={cross_price!r}.")
        date = _parse_event_date(row.get("date"), close_hour=close_hour)
        events.append(
            SolEthTradeEvent(
                date=date,
                event=event,
                cross_price=cross_price,
            )
        )

    warnings: list[str] = []
    sorted_events = sorted(events, key=lambda e: e.date)
    if sorted_events != events:
        warnings.append("Trade log rows were not sorted by date; sorted in-memory.")
    events = sorted_events

    in_trade = False
    for i, event in enumerate(events):
        if event.event == "ENTRY":
            if in_trade:
                raise ValueError(
                    f"Trade log row {i} has ENTRY while another trade is already open."
                )
            in_trade = True
        else:
            if not in_trade:
                raise ValueError(f"Trade log row {i} has EXIT without a prior ENTRY.")
            in_trade = False

    return events, tuple(warnings)


def sample_sol_eth_trade_log_json() -> str:
    return (
        '[\n'
        "  {\n"
        '    "date": "2026-02-21",\n'
        '    "event": "ENTRY",\n'
        '    "cross_price": 0.043338\n'
        "  },\n"
        "  {\n"
        '    "date": "2026-02-25",\n'
        '    "event": "EXIT",\n'
        '    "cross_price": 0.041798\n'
        "  }\n"
        "]\n"
    )


__all__ = ["SolEthTradeEvent", "load_sol_eth_trade_log", "sample_sol_eth_trade_log_json"]
