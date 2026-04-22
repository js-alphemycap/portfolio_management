#!/usr/bin/env python3
"""Send a consolidated email summarizing systematic signal state changes.

Reads today's/yesterday's signals from strategy_signal_history_matrix.csv,
rich per-strategy context (MAs, EMAs, RSI, levels) from
strategy_signal_context.json, and latest strategy allocations from
raw_strategy_allocation_history*.csv to split open vs closed strategies.

No recomputation, no trade-log reads.  Returns / drawdown / stop-loss
monitoring is delegated to the daily performance report.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import os

import pandas as pd

import requests

from portfolio_management.helpers.email import EmailClient
from portfolio_management.telegram_delivery import send_telegram_message


DUAL_MA_STRATEGIES = {"ACTIVE_BTC_MA", "ACTIVE_ETH_MA"}
ROTATION_STRATEGIES = {"ACTIVE_SOL_ETH", "ACTIVE_HYPE_ETH"}
STRATEGY_ORDER = ["ACTIVE_BTC_MA", "ACTIVE_ETH_MA", "ACTIVE_SOL_ETH", "ACTIVE_HYPE_ETH"]
STRATEGY_LONG_ASSET = {
    "ACTIVE_BTC_MA": "BTC",
    "ACTIVE_ETH_MA": "ETH",
    "ACTIVE_SOL_ETH": "SOL",
    "ACTIVE_HYPE_ETH": "HYPE",
}


def _dual_ma_state_label(asset: str, signal_value: float) -> str:
    """Matrix convention (dual MA): 0=full risk-on, 0.5=half, 1=full risk-off."""
    if pd.isna(signal_value):
        return "(no data)"
    val = float(signal_value)
    if val <= 0.0 + 1e-9:
        return "Full Risk-On"
    if val >= 1.0 - 1e-9:
        return "Full Risk-Off"
    if abs(val - 0.5) < 1e-9:
        return "Half Risk-On"
    return f"Partial {val:.2f}"


def _rotation_state_label(long_asset: str, signal_value: float) -> str:
    """Matrix convention (rotation): 1 = in {long_asset}, 0 = in ETH."""
    if pd.isna(signal_value):
        return "(no data)"
    return f"In {long_asset}" if int(round(float(signal_value))) == 1 else "In ETH"


def _dual_ma_action(asset: str, prev_value: float, curr_value: float) -> str:
    # Matrix convention: 0 = full risk-on, 0.5 = half, 1 = full risk-off.
    if pd.isna(prev_value) or pd.isna(curr_value):
        return "NO ACTION (insufficient history)"
    prev = float(prev_value)
    curr = float(curr_value)
    if abs(prev - curr) < 1e-9:
        if curr <= 0.0 + 1e-9:
            return f"STAY FULL RISK-ON ({asset})"
        if curr >= 1.0 - 1e-9:
            return f"STAY FULL RISK-OFF (no {asset})"
        return f"HOLD (partial {asset})"
    if curr < prev:
        return f"RE-RISK {asset} (enter/add)"
    return f"DE-RISK {asset} (reduce/exit)"


def _rotation_action(long_asset: str, prev_value: float, curr_value: float) -> str:
    # Matrix convention: 1 = in trade (long alt), 0 = off trade (in ETH).
    if pd.isna(prev_value) or pd.isna(curr_value):
        return "NO ACTION (insufficient history)"
    prev = int(round(float(prev_value)))
    curr = int(round(float(curr_value)))
    if prev == curr:
        return f"STAY IN {long_asset}" if curr == 1 else "STAY IN ETH"
    if prev == 0 and curr == 1:
        return f"ENTER {long_asset} (rotate out of ETH)"
    if prev == 1 and curr == 0:
        return f"EXIT {long_asset} (rotate to ETH)"
    return "CHANGE"


def _dual_ma_compact_action(asset: str, prev_value: float, curr_value: float) -> str:
    # Returns "no action" if state unchanged, else the new state in lowercase.
    if pd.isna(prev_value) or pd.isna(curr_value):
        return "no action"
    prev = float(prev_value)
    curr = float(curr_value)
    if abs(prev - curr) < 1e-9:
        return "no action"
    if curr <= 0.0 + 1e-9:
        return "full risk on"
    if curr >= 1.0 - 1e-9:
        return "full risk off"
    # Partial — infer direction relative to previous.
    if curr < prev:
        return "half risk on"
    return "half risk off"


def _rotation_compact_action(long_asset: str, prev_value: float, curr_value: float) -> str:
    if pd.isna(prev_value) or pd.isna(curr_value):
        return "no action"
    prev = int(round(float(prev_value)))
    curr = int(round(float(curr_value)))
    if prev == curr:
        return "no action"
    if prev == 0 and curr == 1:
        return "enter"
    if prev == 1 and curr == 0:
        return "exit"
    return "no action"


def _fmt_price(x: float | None) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    f = float(x)
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    if abs(f) >= 10:
        return f"{f:,.2f}"
    return f"{f:,.4f}"


def _fmt_ratio(x: float | None) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.6f}"


def _fmt_rsi(x: float | None) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.1f}"


def _render_dual_ma_section(asset: str, ctx: dict, signal_prev: float, signal_today: float, as_of: str, prev_date: str) -> str:
    close = ctx.get("close")
    atr = ctx.get("atr")
    fast = ctx.get("ma_fast")
    slow = ctx.get("ma_slow")
    fast_band = ctx.get("ma_fast_band") or [None, None]
    slow_band = ctx.get("ma_slow_band") or [None, None]
    fast_days = ctx.get("fast_days")
    slow_days = ctx.get("slow_days")
    atr_days = ctx.get("atr_days")

    lines = [
        f"Previous ({prev_date}): signal={signal_prev:g} → {_dual_ma_state_label(asset, signal_prev)}",
        f"Current  ({as_of}): signal={signal_today:g} → {_dual_ma_state_label(asset, signal_today)}",
        "",
        "Levels",
        f"- Close: {_fmt_price(close)} | ATR({atr_days}d): {_fmt_price(atr)}",
        f"- Fast MA({fast_days}d): {_fmt_price(fast)} | Band [{_fmt_price(fast_band[0])}, {_fmt_price(fast_band[1])}]",
        f"- Slow MA({slow_days}d): {_fmt_price(slow)} | Band [{_fmt_price(slow_band[0])}, {_fmt_price(slow_band[1])}]",
        "",
        "Watching next",
        f"- Confirm Full Risk-On if close > {_fmt_price(slow_band[1])}",
        f"- Abort to Full Risk-Off if close < {_fmt_price(fast_band[0])}",
    ]
    return "\n".join(lines)


def _render_rotation_section(long_asset: str, ctx: dict, signal_prev: float, signal_today: float, as_of: str, prev_date: str) -> str:
    long_close = ctx.get("sol_close") if long_asset == "SOL" else ctx.get("hype_close")
    eth_close = ctx.get("eth_close")
    ratio = ctx.get("ratio")
    ema_fast = ctx.get("ema_fast")
    ema_slow = ctx.get("ema_slow")
    ema_fast_prev = ctx.get("ema_fast_prev")
    ema_slow_prev = ctx.get("ema_slow_prev")
    rsi = ctx.get("rsi")
    rsi_prev = ctx.get("rsi_prev")
    fast_span = ctx.get("fast_span")
    slow_span = ctx.get("slow_span")
    rsi_period = ctx.get("rsi_period")
    rsi_exit_level = ctx.get("rsi_exit_level")

    def _cmp(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return "?"
        return ">" if float(a) > float(b) else ("<=" if float(a) <= float(b) else "?")

    lines = [
        f"Previous ({prev_date}): signal={signal_prev:g} → {_rotation_state_label(long_asset, signal_prev)}",
        f"Current  ({as_of}): signal={signal_today:g} → {_rotation_state_label(long_asset, signal_today)}",
        "",
        "Prices",
        f"- {long_asset} close: {_fmt_price(long_close)} | ETH close: {_fmt_price(eth_close)} | Ratio: {_fmt_ratio(ratio)}",
        "",
        "Price-ratio EMA",
        f"- Today     : fast({fast_span}) {_fmt_ratio(ema_fast)} {_cmp(ema_fast, ema_slow)} slow({slow_span}) {_fmt_ratio(ema_slow)}",
        f"- Yesterday : fast({fast_span}) {_fmt_ratio(ema_fast_prev)} {_cmp(ema_fast_prev, ema_slow_prev)} slow({slow_span}) {_fmt_ratio(ema_slow_prev)}",
        "",
        f"RSI({rsi_period})",
        f"- Today / Yesterday: {_fmt_rsi(rsi)} / {_fmt_rsi(rsi_prev)}",
        f"- Early-exit level: {_fmt_rsi(rsi_exit_level)}",
    ]
    return "\n".join(lines)


def _load_latest_allocations(allocations_path: Path) -> dict[str, float]:
    df = pd.read_csv(allocations_path)
    if df.empty:
        return {}
    if "is_deleted" in df.columns:
        df = df.loc[~df["is_deleted"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    df["effective_from"] = pd.to_datetime(df["effective_from"], utc=True, errors="coerce")
    df["allocation"] = pd.to_numeric(df["allocation"], errors="coerce").fillna(0.0)
    df = df.loc[df["effective_from"].notna()].copy()
    df.sort_values(["strategy_name", "effective_from"], inplace=True)
    latest = df.groupby("strategy_name", sort=False).tail(1)
    return dict(zip(latest["strategy_name"].astype(str), latest["allocation"].astype(float)))


def _build_sections(matrix: pd.DataFrame, context: dict, allocations: dict[str, float]) -> tuple[str, list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    matrix = matrix.sort_values("date").reset_index(drop=True)
    if matrix.empty:
        raise SystemExit("Signal matrix is empty.")
    latest = matrix.iloc[-1]
    prev = matrix.iloc[-2] if len(matrix) >= 2 else None
    as_of = str(latest["date"])
    prev_date = str(prev["date"]) if prev is not None else "n/a"

    strategies = context.get("strategies", {})
    open_sections: list[tuple[str, str]] = []
    closed_sections: list[tuple[str, str]] = []
    actions: list[tuple[str, str]] = []

    for strategy_key in STRATEGY_ORDER:
        if strategy_key not in matrix.columns:
            continue
        curr_val = float(latest[strategy_key])
        prev_val = float(prev[strategy_key]) if prev is not None else float("nan")
        ctx = strategies.get(strategy_key, {})
        alloc = float(allocations.get(strategy_key, 0.0))

        long_asset = STRATEGY_LONG_ASSET[strategy_key]
        if strategy_key in DUAL_MA_STRATEGIES:
            body = _render_dual_ma_section(long_asset, ctx, prev_val, curr_val, as_of, prev_date)
        elif strategy_key in ROTATION_STRATEGIES:
            body = _render_rotation_section(long_asset, ctx, prev_val, curr_val, as_of, prev_date)
        else:
            body = f"Previous: {prev_val:g}\nCurrent: {curr_val:g}"

        if strategy_key in DUAL_MA_STRATEGIES:
            action = _dual_ma_compact_action(long_asset, prev_val, curr_val)
        elif strategy_key in ROTATION_STRATEGIES:
            action = _rotation_compact_action(long_asset, prev_val, curr_val)
        else:
            action = "no action"
        if alloc > 0:
            actions.append((strategy_key, action))

        if alloc > 0:
            open_sections.append((strategy_key, body))
        else:
            closed_sections.append((strategy_key, body))

    return as_of, open_sections, closed_sections, actions


def _render_group(heading: str, sections: list[tuple[str, str]]) -> str:
    if not sections:
        return (
            "<div style=\"margin:8px 0 24px;\">"
            f"<h2 style=\"margin:0 0 8px; font-size:22px; font-weight:700; color:#1f1f1c; border-bottom:2px solid #1f1f1c; padding-bottom:6px;\">{html.escape(heading)}</h2>"
            "<p style=\"color:#6d6a63; font-style:italic; font-size:14px;\">(none)</p>"
            "</div>"
        )
    body_blocks = []
    for title, message in sections:
        body_blocks.append(
            "<div style=\"border:1px solid #ddd8cf; background:#ffffff; padding:22px 24px;\">"
            f"<h3 style=\"margin:0 0 14px; font-size:16px; font-weight:600;\">• {html.escape(title)}</h3>"
            f"<pre style=\"margin:0; white-space:pre-wrap; word-break:break-word; "
            "font:13px/1.6 Menlo,Consolas,monospace; color:#242424;\">"
            f"{html.escape(message)}</pre>"
            "</div>"
            "<div style=\"height:16px;\"></div>"
        )
    return (
        "<div style=\"margin:8px 0 24px;\">"
        f"<h2 style=\"margin:0 0 12px; font-size:22px; font-weight:700; color:#1f1f1c; border-bottom:2px solid #1f1f1c; padding-bottom:6px;\">{html.escape(heading)}</h2>"
        + "".join(body_blocks)
        + "</div>"
    )


def _render_actions_html(actions: list[tuple[str, str]]) -> str:
    heading = "Summary"
    if not actions:
        body = "<p style=\"color:#6d6a63; font-style:italic; font-size:14px;\">(no open strategies)</p>"
    else:
        items = "".join(
            f"<li style=\"margin:4px 0; font-variant-numeric: tabular-nums;\"><strong>{html.escape(strategy_key)}</strong>: {html.escape(action)}</li>"
            for strategy_key, action in actions
        )
        body = f"<ul style=\"margin:6px 0 0 22px; padding:0; font-size:14px; list-style:none;\">{items}</ul>"
    return (
        "<div style=\"margin:8px 0 24px;\">"
        f"<h2 style=\"margin:0 0 6px; font-size:22px; font-weight:700; color:#1f1f1c; border-bottom:2px solid #1f1f1c; padding-bottom:6px;\">{html.escape(heading)}</h2>"
        f"{body}"
        "</div>"
    )


def _build_html(subject: str, as_of: str, open_sections: list[tuple[str, str]], closed_sections: list[tuple[str, str]], actions: list[tuple[str, str]]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%B %-d, %Y %H:%M UTC")
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>"
        "body{margin:0;padding:24px;background:#ffffff;color:#1f1f1c;font-family:Georgia,serif;}"
        ".page{max-width:860px;margin:0 auto;}"
        "h1{margin:0 0 8px;font-size:30px;font-weight:600;}"
        ".meta{margin:0 0 24px;color:#6d6a63;font-size:14px;}"
        "</style></head><body><main class=\"page\">"
        f"<h1>{html.escape(subject)}</h1>"
        f"<p class=\"meta\">As of {html.escape(as_of)}<br>Generated at {html.escape(generated_at)}</p>"
        + _render_actions_html(actions)
        + _render_group("Open Strategies", open_sections)
        + _render_group("Closed Strategies", closed_sections)
        + "</main></body></html>"
    )


def _build_text(subject: str, as_of: str, open_sections: list[tuple[str, str]], closed_sections: list[tuple[str, str]], actions: list[tuple[str, str]]) -> str:
    blocks = [subject, f"As of {as_of}", ""]
    blocks.extend(["SUMMARY", "=" * len("SUMMARY"), ""])
    if actions:
        blocks.extend(f"- {sk}: {act}" for sk, act in actions)
    else:
        blocks.append("(no open strategies)")
    blocks.append("")
    for heading, sections in (("Open Strategies", open_sections), ("Closed Strategies", closed_sections)):
        blocks.extend([heading.upper(), "=" * len(heading), ""])
        if not sections:
            blocks.extend(["(none)", ""])
            continue
        for title, message in sections:
            blocks.extend([title, "-" * len(title), message, ""])
    return "\n".join(blocks).strip()


HEADER_TARGET_LEN = 27


def _extract_state_action(message: str) -> tuple[str, str]:
    state = "?"
    action = "?"
    for line in message.splitlines():
        line = line.strip()
        if line.startswith("Current") and "→" in line:
            state = line.split("→", 1)[1].strip()
        elif line.startswith("Action today:"):
            raw = line.split(":", 1)[1].strip()
            # Strip parenthetical clarifiers like "HOLD (no state change)".
            action = raw.split(" (", 1)[0].strip() if " (" in raw else raw
    return state, action


def _extract_dual_ma_close(message: str) -> str | None:
    """Pull the current close price from a dual-MA rendered body."""
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith("- Close:"):
            m = re.search(r"Close:\s*([\d,\.]+)", stripped)
            if m:
                return m.group(1)
    return None


def _extract_dual_ma_key_levels(message: str) -> str | None:
    """Pull upper/lower watch levels from a dual-MA rendered body."""
    upper = None
    lower = None
    for line in message.splitlines():
        stripped = line.strip()
        m = re.search(r"close >\s+([\d,\.]+)", stripped)
        if m:
            upper = m.group(1)
            continue
        m = re.search(r"close <\s+([\d,\.]+)", stripped)
        if m:
            lower = m.group(1)
    if upper and lower:
        return f"> {upper} / < {lower}"
    return None


def _padded_header(heading: str) -> str:
    # Produce a header like "---- OPEN STRATEGIES ----" where the total line
    # width equals HEADER_TARGET_LEN for visual alignment across sections.
    inner = f" {heading} "
    remaining = max(HEADER_TARGET_LEN - len(inner), 2)
    left = remaining // 2
    right = remaining - left
    return f"{'-' * left}{inner}{'-' * right}"


def _render_telegram_group(heading: str, sections: list[tuple[str, str]]) -> list[str]:
    lines = [_padded_header(heading), ""]
    if not sections:
        lines.extend(["(none)", ""])
        return lines
    for title, body in sections:
        state, _ = _extract_state_action(body)
        lines.append(f"• {title}")
        lines.append(f"  State: {state}")
        current_close = _extract_dual_ma_close(body)
        if current_close:
            lines.append(f"  Current: {current_close}")
        key_levels = _extract_dual_ma_key_levels(body)
        if key_levels:
            lines.append(f"  Key levels: {key_levels}")
        lines.append("")
    return lines


def _build_telegram(as_of: str, open_sections: list[tuple[str, str]], closed_sections: list[tuple[str, str]], actions: list[tuple[str, str]]) -> str:
    """Telegram body: header + ACTIONS + per-strategy state grouped by open/closed."""
    lines = [f"📊 Daily Active Strategy Signals — {as_of}", ""]
    lines.append(_padded_header("SUMMARY"))
    lines.append("")
    if actions:
        for strategy_key, action in actions:
            lines.append(f"• {strategy_key}: {action}")
    else:
        lines.append("(no open strategies)")
    lines.append("")
    lines.extend(_render_telegram_group("OPEN STRATEGIES", open_sections))
    lines.extend(_render_telegram_group("CLOSED STRATEGIES", closed_sections))
    return "\n".join(lines).rstrip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send consolidated daily signal summary derived from the signal matrix + context JSON + allocations."
    )
    parser.add_argument("--signals-path", type=Path, required=True, help="Signal matrix CSV path.")
    parser.add_argument("--context-path", type=Path, required=True, help="Signal context JSON path.")
    parser.add_argument("--allocations-path", type=Path, required=True, help="Raw strategy allocation history CSV path.")
    parser.add_argument("--recipient", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument(
        "--telegram-chat-id",
        default=None,
        help="If provided, also send the state summary to this Telegram chat.",
    )
    parser.add_argument(
        "--telegram-attachment",
        action="append",
        default=[],
        help="Path to attach to the Telegram send. If one or more are passed, a single "
             "media group message is sent with the signal summary as the caption on the "
             "first attachment. Pass multiple times for multiple files.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _send_telegram_media_group(bot_token: str, chat_id: str, attachments: list[Path], caption: str) -> dict:
    """Send a media group of documents to Telegram with a caption on the first item."""
    import json as _json

    media = []
    files = {}
    for i, path in enumerate(attachments):
        key = f"file{i}"
        item = {"type": "document", "media": f"attach://{key}"}
        if i == 0:
            item["caption"] = caption
        media.append(item)
        files[key] = (path.name, open(path, "rb"))

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMediaGroup",
            data={"chat_id": chat_id, "media": _json.dumps(media)},
            files=files,
            timeout=120,
        )
    finally:
        for _, file_tuple in files.items():
            file_tuple[1].close()
    if not resp.ok:
        try:
            info = resp.json()
            desc = info.get("description")
        except Exception:
            desc = resp.text
        raise SystemExit(f"Telegram sendMediaGroup error ({resp.status_code}): {desc}")
    return resp.json()


def main() -> None:
    args = build_parser().parse_args()
    matrix = pd.read_csv(args.signals_path)
    with open(args.context_path) as f:
        context = json.load(f)
    allocations = _load_latest_allocations(args.allocations_path)
    as_of, open_sections, closed_sections, actions = _build_sections(matrix, context, allocations)
    subject = args.subject or f"Daily Active Strategy Signals - {as_of}"
    html_body = _build_html(subject, as_of, open_sections, closed_sections, actions)
    telegram_body = _build_telegram(as_of, open_sections, closed_sections, actions)

    if args.dry_run:
        print(f"Subject: {subject}")
        print(_build_text(subject, as_of, open_sections, closed_sections, actions))
        if args.telegram_chat_id:
            print("")
            print(f"Telegram chat_id: {args.telegram_chat_id}")
            if args.telegram_attachment:
                attach_names = ", ".join(Path(p).name for p in args.telegram_attachment)
                print(f"Media group attachments: {attach_names}")
                print("--- caption ---")
            print(telegram_body)
        return

    client = EmailClient()
    if args.recipient:
        client.recipient = args.recipient
    client.send_html(subject, html_body)
    print(f"Sent daily active strategy signals email: {subject}")

    if args.telegram_chat_id:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not bot_token:
            raise SystemExit("TELEGRAM_BOT_TOKEN env var required for --telegram-chat-id.")
        if args.telegram_attachment:
            reports_caption = f"📎 Daily Portfolio Management Reports — {as_of}"
            _send_telegram_media_group(
                bot_token,
                args.telegram_chat_id,
                [Path(p) for p in args.telegram_attachment],
                reports_caption,
            )
            attach_names = ", ".join(Path(p).name for p in args.telegram_attachment)
            print(f"Sent Telegram reports media group ({attach_names}) to chat {args.telegram_chat_id}.")
        send_telegram_message(bot_token, args.telegram_chat_id, telegram_body)
        print(f"Sent Telegram daily active strategy signals summary to chat {args.telegram_chat_id}.")


if __name__ == "__main__":
    main()
