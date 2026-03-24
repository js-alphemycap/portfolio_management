#!/usr/bin/env python3
"""Build a concise daily strategy summary from archived strategy messages."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import re
from datetime import date, datetime, timezone

from portfolio_management.message_archive import (
    latest_archived_message_date,
    load_archived_strategy_message,
)


def _extract(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _shorten_reserve_watch_line(line: str) -> str:
    return (
        line.replace("Re-risking Entry if close > ", "> ")
        .replace("Confirm Full Risk-On if close > ", "> ")
        .replace("Re-confirm Full Risk-On if close > ", "> ")
        .replace("De-risk to Partial Risk-Off if close < ", "< ")
        .replace("Exit to Full Risk-Off if close < ", "< ")
        .replace("Abort to Full Risk-Off if close < ", "< ")
    )


def _reserve_next_level(message: str, asset: str) -> str:
    pattern = rf"(?s)--- {asset} ---.*?Watching next\n(.*?)(?:\n\n--- |\Z)"
    block = _extract(pattern, message)
    if not block:
        return "NA"
    lines = [_shorten_reserve_watch_line(line.strip()) for line in block.splitlines() if line.strip()]
    return " / ".join(lines) if lines else "NA"


def summarize_reserve(message: str) -> list[str]:
    weights = _extract(r"^Target weights:\s*(.+)$", message) or "NA"
    btc_state = _extract(r"--- BTC ---\nState:\s*(.+)", message) or "NA"
    eth_state = _extract(r"--- ETH ---\nState:\s*(.+)", message) or "NA"
    btc_next = _reserve_next_level(message, "BTC")
    eth_next = _reserve_next_level(message, "ETH")
    return [
        "Reserve Portfolio",
        "",
        f"Recommended weights {weights}",
        f"BTC {btc_state} | next {btc_next}",
        f"ETH {eth_state} | next {eth_next}",
        "",
    ]


def _reserve_action_summary(message: str) -> str:
    trigger_today = _extract(r"^Trigger today:\s*(YES|NO)$", message) or "NO"
    weights = _extract(r"^Target weights:\s*(.+)$", message) or "NA"
    if trigger_today == "YES":
        return f"Reserve rebalance to {weights}"
    return "No action"


def _alt_trigger_summary(message: str) -> str:
    action_today = _extract(r"^- Trigger today:\s*(YES|NO)$", message) or "NO"
    if action_today == "NO":
        return ""

    for pattern, label in [
        (r"^- RSI early-exit condition .*:\s*YES$", "RSI exit"),
        (r"^- EMA exit condition .*:\s*YES$", "EMA exit"),
        (r"^- Entry signal today:\s*YES$", "EMA entry"),
    ]:
        if re.search(pattern, message, flags=re.MULTILINE):
            return label
    return "triggered"


def _alt_action_label(message: str, title: str) -> str:
    action_today = _extract(r"^- Trigger today:\s*(YES|NO)$", message) or "NO"
    in_trade = (_extract(r"^- State:\s*(.+)$", message) or "").upper()
    if title == "SOL/ETH":
        return "Exit SOL to ETH" if action_today == "YES" and "IN TRADE" in in_trade else (
            "Buy SOL vs ETH" if action_today == "YES" else "No action"
        )
    return "Exit HYPE to ETH" if action_today == "YES" and "IN TRADE" in in_trade else (
        "Buy HYPE vs ETH" if action_today == "YES" else "No action"
    )


def _pretty_position(raw: str) -> str:
    mapping = {
        "OFF TRADE (ETH)": "Off trade (ETH)",
        "IN TRADE (SOL)": "In trade (SOL)",
        "IN TRADE (HYPE)": "In trade (HYPE)",
        "FULL RISK-OFF": "Full risk-off",
        "PARTIAL RISK-OFF": "Partial risk-off",
        "FULL RISK-ON": "Full risk-on",
        "RE-RISKING ENTRY": "Re-risking entry",
    }
    normalized = raw.strip()
    return mapping.get(normalized.upper(), normalized[:1].upper() + normalized[1:].lower() if normalized.isupper() else normalized)


def summarize_alt(message: str, title: str) -> str:
    position = _pretty_position(_extract(r"^- State:\s*(.+)$", message) or "NA")
    action = _extract(r"^- Trigger today:\s*(YES|NO)$", message) or "NO"
    trigger = _alt_trigger_summary(message)
    action_label = _alt_action_label(message, title)
    trade = _extract(r"^- Relative outperformance:\s*([^\n]+)$", message) or "NA"
    if trade == "NA":
        trade = _extract(r"^- Return:\s*([^\n]+)$", message) or "NA"
    cumulative = _extract(r"^- Cumulative return:\s*([^\n]+)$", message) or "NA"
    stop_line = re.search(
        r"^- Stop-loss rule: threshold ([^,]+), current ([^,]+), trigger=.*$",
        message,
        flags=re.MULTILINE,
    )
    stop_text = "NA"
    if stop_line:
        stop_text = f"{stop_line.group(2).strip()} vs {stop_line.group(1).strip()}"
    parts = [
        f"- {title}: {position}",
        f"Action today {action_label}",
    ]
    if action == "YES" and trigger:
        parts.append(f"trigger {trigger}")
    parts.append(f"current trade {trade}")
    parts.append(f"cumulative {cumulative}")
    parts.append(f"stoploss {stop_text}")
    return " | ".join(parts)


def _action_header_lines(reserve: str, sol: str, hype: str) -> list[str]:
    reserve_action = _reserve_action_summary(reserve)
    sol_action = _alt_action_label(sol, "SOL/ETH")
    hype_action = _alt_action_label(hype, "HYPE/ETH")
    has_action = any(
        action != "No action"
        for action in (reserve_action, sol_action, hype_action)
    )
    return [
        f"Today's Actions: {'YES' if has_action else 'NO'}",
        "",
        f"- Reserve: {reserve_action}",
        f"- SOL/ETH: {sol_action}",
        f"- HYPE/ETH: {hype_action}",
    ]


def build_summary_text(message_date: date) -> str:
    reserve = load_archived_strategy_message(
        strategy_slug="reserve_dual_ma",
        message_date=message_date,
    )
    sol = load_archived_strategy_message(
        strategy_slug="sol_eth_rotation",
        message_date=message_date,
    )
    hype = load_archived_strategy_message(
        strategy_slug="hype_eth_rotation",
        message_date=message_date,
    )

    divider = "-" * 26
    lines = [
        f"Daily Strategy Summary — {message_date.isoformat()}",
        "",
        divider,
        "",
        *_action_header_lines(reserve, sol, hype),
        "",
        divider,
        "",
    ]
    lines.extend(summarize_reserve(reserve))
    lines.append(divider)
    lines.append("")
    lines.append("Alternative Strategies")
    lines.append("")
    lines.append(summarize_alt(sol, "SOL/ETH"))
    lines.append("")
    lines.append(summarize_alt(hype, "HYPE/ETH"))
    lines.append("")
    lines.append(divider)
    return "\n".join(lines)


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> dict[str, object]:
    import requests
    from portfolio_management.helpers.http import get_requests_verify

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a concise daily strategy summary built from archived strategy messages."
    )
    parser.add_argument("chat_id", nargs="?", help="Telegram chat ID.")
    parser.add_argument(
        "--date",
        default=None,
        help="Date to summarize in YYYY-MM-DD. Defaults to today UTC.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message without sending it.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    today_utc = datetime.now(timezone.utc).date()
    requested_date = (
        date.fromisoformat(args.date)
        if args.date
        else latest_archived_message_date(strategy_slug="reserve_dual_ma")
    )
    warning_lines: list[str] = []
    try:
        message_date = requested_date
        summary_text = build_summary_text(message_date)
    except FileNotFoundError:
        fallback_date = latest_archived_message_date(strategy_slug="reserve_dual_ma")
        summary_text = build_summary_text(fallback_date)
        warning_lines.append(
            f"WARNING: No archived summary inputs found for requested date {requested_date.isoformat()}; "
            f"using latest available date {fallback_date.isoformat()}."
        )
        message_date = fallback_date

    if message_date != today_utc:
        warning_lines.append(
            f"WARNING: Summary date is {message_date.isoformat()}, "
            f"not today UTC {today_utc.isoformat()}."
        )
    if warning_lines:
        summary_text = "\n\n".join(warning_lines + [summary_text])

    print("📨 Telegram daily strategy summary:")
    print(summary_text)

    if args.dry_run:
        print("Dry-run mode enabled; message not sent.")
        return

    if not args.chat_id:
        parser.error("chat_id is required unless --dry-run is used.")

    bot_token: str | None = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        parser.error("Telegram bot token is required via TELEGRAM_BOT_TOKEN environment variable.")

    response = send_telegram_message(bot_token, args.chat_id, summary_text)
    print("✅ Message sent. Telegram response:")
    print(response)


if __name__ == "__main__":
    main()
