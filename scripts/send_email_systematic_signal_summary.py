#!/usr/bin/env python3
"""Send a consolidated email summarizing systematic signal state changes.

Reads today's/yesterday's signals from strategy_signal_history_matrix.csv and
the rich per-strategy context (MAs, EMAs, RSI, levels) from
strategy_signal_context.json. No recomputation, no trade-log reads.
Return / drawdown / stop-loss monitoring is delegated to the daily
performance report.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio_management.helpers.email import EmailClient


STRATEGY_DISPLAY = {
    "ACTIVE_BTC_MA": "BTC Dual MA (Reserve)",
    "ACTIVE_ETH_MA": "ETH Dual MA (Reserve)",
    "ACTIVE_SOL_ETH": "SOL/ETH Rotation",
    "ACTIVE_HYPE_ETH": "HYPE/ETH Rotation",
}


def _state_label_dual_ma(asset: str, signal_value: float) -> str:
    if pd.isna(signal_value):
        return "(no data)"
    return (
        f"Risk-On (holding {asset})"
        if int(round(float(signal_value))) == 1
        else f"Risk-Off (no {asset} position)"
    )


def _state_label_rotation(long_asset: str, signal_value: float) -> str:
    if pd.isna(signal_value):
        return "(no data)"
    return (
        f"In-trade (holding {long_asset})"
        if int(round(float(signal_value))) == 1
        else "Off-trade (holding ETH)"
    )


def _action_label(prev_value: float, curr_value: float) -> str:
    if pd.isna(prev_value) or pd.isna(curr_value):
        return "NO ACTION (insufficient history)"
    prev = int(round(float(prev_value)))
    curr = int(round(float(curr_value)))
    if prev == curr:
        return "HOLD (no state change)"
    if prev == 0 and curr == 1:
        return "ENTER"
    if prev == 1 and curr == 0:
        return "EXIT"
    return "CHANGE"


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


def _render_dual_ma_section(asset: str, strategy_key: str, ctx: dict, signal_prev: float, signal_today: float, as_of: str, prev_date: str) -> str:
    close = ctx.get("close")
    atr = ctx.get("atr")
    fast = ctx.get("ma_fast")
    slow = ctx.get("ma_slow")
    fast_band = ctx.get("ma_fast_band") or [None, None]
    slow_band = ctx.get("ma_slow_band") or [None, None]
    fast_days = ctx.get("fast_days")
    slow_days = ctx.get("slow_days")
    atr_days = ctx.get("atr_days")
    target = ctx.get("target_weight")

    lines = [
        f"Previous ({prev_date}): signal={int(round(signal_prev))} → {_state_label_dual_ma(asset, signal_prev)}",
        f"Current  ({as_of}): signal={int(round(signal_today))} → {_state_label_dual_ma(asset, signal_today)}",
        f"Action today: {_action_label(signal_prev, signal_today)}",
        "",
        f"Target weight if risk-on: {int(round((target or 0.0) * 100))}%",
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
        f"Previous ({prev_date}): signal={int(round(signal_prev))} → {_state_label_rotation(long_asset, signal_prev)}",
        f"Current  ({as_of}): signal={int(round(signal_today))} → {_state_label_rotation(long_asset, signal_today)}",
        f"Action today: {_action_label(signal_prev, signal_today)}",
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


def _build_sections(matrix: pd.DataFrame, context: dict) -> tuple[str, str, list[tuple[str, str]]]:
    matrix = matrix.sort_values("date").reset_index(drop=True)
    if matrix.empty:
        raise SystemExit("Signal matrix is empty.")
    latest = matrix.iloc[-1]
    prev = matrix.iloc[-2] if len(matrix) >= 2 else None
    as_of = str(latest["date"])
    prev_date = str(prev["date"]) if prev is not None else "n/a"

    strategies = context.get("strategies", {})
    sections: list[tuple[str, str]] = []

    for strategy_key, display_name in STRATEGY_DISPLAY.items():
        if strategy_key not in matrix.columns:
            continue
        curr_val = float(latest[strategy_key])
        prev_val = float(prev[strategy_key]) if prev is not None else float("nan")
        ctx = strategies.get(strategy_key, {})

        if strategy_key == "ACTIVE_BTC_MA":
            body = _render_dual_ma_section("BTC", strategy_key, ctx, prev_val, curr_val, as_of, prev_date)
        elif strategy_key == "ACTIVE_ETH_MA":
            body = _render_dual_ma_section("ETH", strategy_key, ctx, prev_val, curr_val, as_of, prev_date)
        elif strategy_key == "ACTIVE_SOL_ETH":
            body = _render_rotation_section("SOL", ctx, prev_val, curr_val, as_of, prev_date)
        elif strategy_key == "ACTIVE_HYPE_ETH":
            body = _render_rotation_section("HYPE", ctx, prev_val, curr_val, as_of, prev_date)
        else:
            body = f"Previous: {prev_val:g}\nCurrent: {curr_val:g}"

        sections.append((display_name, body))
    return as_of, prev_date, sections


def _build_html(subject: str, as_of: str, sections: list[tuple[str, str]]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%B %-d, %Y %H:%M UTC")
    body_sections = []
    for title, message in sections:
        body_sections.append(
            "<div style=\"border:1px solid #ddd8cf; background:#ffffff; padding:22px 24px;\">"
            f"<h2 style=\"margin:0 0 14px; font-size:18px; font-weight:600;\">{html.escape(title)}</h2>"
            f"<pre style=\"margin:0; white-space:pre-wrap; word-break:break-word; "
            "font:13px/1.6 Menlo,Consolas,monospace; color:#242424;\">"
            f"{html.escape(message)}</pre>"
            "</div>"
            "<div style=\"height:20px;\"></div>"
        )
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>"
        "body{margin:0;padding:24px;background:#ffffff;color:#1f1f1c;font-family:Georgia,serif;}"
        ".page{max-width:860px;margin:0 auto;}"
        "h1{margin:0 0 8px;font-size:30px;font-weight:600;}"
        ".meta{margin:0 0 24px;color:#6d6a63;font-size:14px;}"
        "</style></head><body><main class=\"page\">"
        f"<h1>{html.escape(subject)}</h1>"
        f"<p class=\"meta\">As of {html.escape(as_of)}<br>Generated at {html.escape(generated_at)}</p>"
        + "".join(body_sections)
        + "</main></body></html>"
    )


def _build_text(subject: str, as_of: str, sections: list[tuple[str, str]]) -> str:
    blocks = [subject, f"As of {as_of}", ""]
    for title, message in sections:
        blocks.extend([title, "=" * len(title), message, ""])
    return "\n".join(blocks).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send consolidated daily signal summary derived from the signal matrix + context JSON."
    )
    parser.add_argument("--signals-path", type=Path, required=True, help="Signal matrix CSV path.")
    parser.add_argument("--context-path", type=Path, required=True, help="Signal context JSON path.")
    parser.add_argument("--recipient", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    matrix = pd.read_csv(args.signals_path)
    with open(args.context_path) as f:
        context = json.load(f)
    as_of, _, sections = _build_sections(matrix, context)
    subject = args.subject or f"Daily Systematic Signal Summary - {as_of}"
    html_body = _build_html(subject, as_of, sections)
    if args.dry_run:
        print(f"Subject: {subject}")
        print(_build_text(subject, as_of, sections))
        return
    client = EmailClient()
    if args.recipient:
        client.recipient = args.recipient
    client.send_html(subject, html_body)
    print(f"Sent consolidated systematic signal email: {subject}")


if __name__ == "__main__":
    main()
