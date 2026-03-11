from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _fmt_pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100.0 * float(x):.{digits}f}%"


def _fmt_num(x: float) -> str:
    if pd.isna(x):
        return "NA"
    x = float(x)
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 10:
        return f"{x:,.2f}"
    return f"{x:,.4f}"


def _infer_state(sig: float, *, re_risking: bool, tol: float = 1e-9) -> str:
    if sig is None or pd.isna(sig):
        return "NA"
    s = float(sig)
    rr = bool(re_risking)
    if abs(s - 0.0) <= tol:
        return "Full Risk-Off"
    if abs(s - 0.5) <= tol:
        return "Partial Risk-Off"
    if abs(s - 1.0) <= tol:
        return "Re-Risking Entry" if rr else "Full Risk-On"
    return f"Signal {s:.3f}"


def _asset_levels(row: pd.Series, prefix: str) -> Dict[str, float]:
    close = row[f"{prefix}_close"]
    atr = row[f"{prefix}_atr"]
    ma_f = row[f"{prefix}_ma_fast"]
    ma_s = row[f"{prefix}_ma_slow"]

    fast_lo = row[f"{prefix}_ma_fast_lo"]
    fast_hi = row[f"{prefix}_ma_fast_hi"]
    slow_lo = row[f"{prefix}_ma_slow_lo"]
    slow_hi = row[f"{prefix}_ma_slow_hi"]

    upper_hi = max(float(fast_hi), float(slow_hi))
    lower_hi = min(float(fast_hi), float(slow_hi))
    lower_lo = min(float(fast_lo), float(slow_lo))

    return {
        "close": close,
        "atr": atr,
        "ma_f": ma_f,
        "ma_s": ma_s,
        "fast_lo": fast_lo,
        "fast_hi": fast_hi,
        "slow_lo": slow_lo,
        "slow_hi": slow_hi,
        "upper_hi": upper_hi,
        "lower_hi": lower_hi,
        "lower_lo": lower_lo,
    }


def _watching_next_lines(state: str, lv: Dict[str, float]) -> List[str]:
    fast_lo = lv["fast_lo"]
    upper_hi = lv["upper_hi"]
    lower_hi = lv["lower_hi"]
    lower_lo = lv["lower_lo"]

    if state == "Full Risk-On":
        return [
            f"De-risk to Partial Risk-Off if close < {_fmt_num(fast_lo)}",
            f"Exit to Full Risk-Off if close < {_fmt_num(lower_lo)}",
        ]
    if state == "Partial Risk-Off":
        return [
            f"Re-confirm Full Risk-On if close > {_fmt_num(upper_hi)}",
            f"Exit to Full Risk-Off if close < {_fmt_num(lower_lo)}",
        ]
    if state == "Full Risk-Off":
        return [f"Re-risking Entry if close > {_fmt_num(lower_hi)}"]
    if state == "Re-Risking Entry":
        return [
            f"Confirm Full Risk-On if close > {_fmt_num(upper_hi)}",
            f"Abort to Full Risk-Off if close < {_fmt_num(lower_lo)}",
        ]
    return ["NA"]


def build_dual_ma_strategy_asset_block(
    *,
    row: pd.Series,
    prefix: str,
    fast_days: int,
    slow_days: int,
    atr_days: int,
    sig: float,
    re_risking: bool,
    target: float,
) -> str:
    state = _infer_state(sig, re_risking=re_risking)
    lv = _asset_levels(row, prefix)

    lines: List[str] = []
    lines.append("")
    lines.append(f"--- {prefix} ---")
    lines.append(f"State: {state}")
    lines.append(f"Signal: {float(sig):.1f} → Target {_fmt_pct(target)}")
    lines.append("")
    lines.append("Levels")
    lines.append(f"Close {_fmt_num(lv['close'])} | ATR({atr_days}d) {_fmt_num(lv['atr'])}")
    lines.append(
        f"Fast MA({fast_days}d) {_fmt_num(lv['ma_f'])} | Band [{_fmt_num(lv['fast_lo'])}, {_fmt_num(lv['fast_hi'])}]"
    )
    lines.append(
        f"Slow MA({slow_days}d) {_fmt_num(lv['ma_s'])} | Band [{_fmt_num(lv['slow_lo'])}, {_fmt_num(lv['slow_hi'])}]"
    )
    lines.append("")
    lines.append("Watching next")
    lines.extend(_watching_next_lines(state, lv))
    return "\n".join(lines)


def build_dual_ma_strategy_reserve_portfolio_message(
    *,
    as_of: pd.Timestamp,
    row: pd.Series,
    btc_sig: float,
    eth_sig: float,
    btc_rerisk: bool,
    eth_rerisk: bool,
    trigger_today: bool,
    btc_fast_days: int,
    btc_slow_days: int,
    btc_atr_days: int,
    eth_fast_days: int,
    eth_slow_days: int,
    eth_atr_days: int,
) -> str:
    header = [
        f"Reserve Portfolio — {as_of.date()}",
        f"Trigger today: {'YES' if bool(trigger_today) else 'NO'}",
        "Target weights: "
        f"BTC {_fmt_pct(row['BTC_target'])} | "
        f"ETH {_fmt_pct(row['ETH_target'])} | "
        f"CASH {_fmt_pct(row['CASH_target'])}",
        "",
    ]

    btc_block = build_dual_ma_strategy_asset_block(
        row=row,
        prefix="BTC",
        fast_days=btc_fast_days,
        slow_days=btc_slow_days,
        atr_days=btc_atr_days,
        sig=float(btc_sig),
        re_risking=bool(btc_rerisk),
        target=float(row["BTC_target"]),
    )
    eth_block = build_dual_ma_strategy_asset_block(
        row=row,
        prefix="ETH",
        fast_days=eth_fast_days,
        slow_days=eth_slow_days,
        atr_days=eth_atr_days,
        sig=float(eth_sig),
        re_risking=bool(eth_rerisk),
        target=float(row["ETH_target"]),
    )
    return "\n".join(header) + btc_block + "\n\n" + eth_block


__all__ = [
    "build_dual_ma_strategy_reserve_portfolio_message",
    "build_dual_ma_strategy_asset_block",
]

