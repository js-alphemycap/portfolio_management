from __future__ import annotations

import pandas as pd

from .hype_eth_rotation_strategy import HypeEthRotationConfig, HypeEthRotationSnapshot


def _fmt_ratio(value: float) -> str:
    return f"{float(value):,.6f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    pct = float(value) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.{digits}f}%"


def _fmt_num(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def _position_text(in_position: bool) -> str:
    return "IN TRADE (HYPE)" if in_position else "OFF TRADE (ETH)"


def build_hype_eth_rotation_telegram_message(
    *,
    snapshot: HypeEthRotationSnapshot,
    config: HypeEthRotationConfig,
) -> str:
    lines: list[str] = []
    divider = "-" * 36

    if snapshot.in_position:
        trigger_today = bool(snapshot.signal_flip_today or snapshot.early_exit_today or snapshot.stop_loss_triggered)
        action_text = (
            "EXIT TO ETH"
            if trigger_today
            else "HOLD HYPE"
        )
    else:
        entry_signal = bool(
            snapshot.signal_flip_today
            and snapshot.entry_filter_ok_today
            and snapshot.ema_fast > snapshot.ema_slow
            and snapshot.strategy_on
        )
        trigger_today = entry_signal
        action_text = (
            "ENTER HYPE TRADE"
            if entry_signal
            else "STAY IN ETH"
        )

    lines.append(f"HYPE/ETH Rotation Signal - {snapshot.as_of.date()}")
    lines.append(divider)
    lines.append(f"- Trigger today: {'YES' if trigger_today else 'NO'}")
    lines.append(f"- Action: {action_text}")
    lines.append(f"- HYPE close: {_fmt_ratio(snapshot.hype_close)}")
    lines.append(f"- ETH close: {_fmt_ratio(snapshot.eth_close)}")
    ema_relation_today = ">" if snapshot.price_ratio_ema_fast > snapshot.price_ratio_ema_slow else "<="
    lines.append(
        f"- Price-ratio EMA today: fast({_fmt_num(config.fast_span, 0)}) {_fmt_ratio(snapshot.price_ratio_ema_fast)} "
        f"{ema_relation_today} slow({_fmt_num(config.slow_span, 0)}) {_fmt_ratio(snapshot.price_ratio_ema_slow)}"
    )
    if snapshot.price_ratio_ema_fast_prev is not None and snapshot.price_ratio_ema_slow_prev is not None:
        ema_relation_prev = ">" if snapshot.price_ratio_ema_fast_prev > snapshot.price_ratio_ema_slow_prev else "<="
        lines.append(
            f"- Price-ratio EMA yesterday: fast({_fmt_num(config.fast_span, 0)}) {_fmt_ratio(snapshot.price_ratio_ema_fast_prev)} "
            f"{ema_relation_prev} slow({_fmt_num(config.slow_span, 0)}) {_fmt_ratio(snapshot.price_ratio_ema_slow_prev)}"
        )
    if snapshot.rsi_prev is not None:
        lines.append(
            f"- RSI({config.rsi_period}) today / yesterday: "
            f"{_fmt_num(snapshot.rsi, 1)} / {_fmt_num(snapshot.rsi_prev, 1)}"
        )
    else:
        lines.append(f"- RSI({config.rsi_period}) today: {_fmt_num(snapshot.rsi, 1)}")
    lines.append(divider)

    if snapshot.in_position:
        lines.append("Exit Criteria Check")
        lines.append(
            f"- EMA exit condition (fast ratio EMA < slow ratio EMA): "
            f"{'YES' if snapshot.ema_fast < snapshot.ema_slow else 'NO'}"
        )
        if config.use_rsi_early_exit:
            lines.append(
                f"- RSI early-exit condition (cross down { _fmt_num(config.rsi_long_exit_level) }): "
                f"{'YES' if snapshot.early_exit_today else 'NO'}"
            )
        lines.append(
            f"- Exit signal today: "
            f"{'YES' if (snapshot.signal_flip_today or snapshot.early_exit_today) else 'NO'}"
        )
    else:
        lines.append("Entry Criteria Check")
        lines.append(
            f"- EMA entry condition (fast ratio EMA > slow ratio EMA): "
            f"{'YES' if snapshot.ema_fast > snapshot.ema_slow else 'NO'}"
        )
        if config.signal_return_window > 0:
            lines.append(
                f"- Return filter ({config.signal_return_window}D > 0): "
                f"{_fmt_pct(snapshot.ratio_return_window)} -> "
                f"{'PASS' if snapshot.entry_filter_ok_today else 'BLOCK'}"
            )
        lines.append(
            f"- Entry signal today: "
            f"{'YES' if (snapshot.signal_flip_today and snapshot.entry_filter_ok_today and snapshot.ema_fast > snapshot.ema_slow and snapshot.strategy_on) else 'NO'}"
        )
    lines.append(divider)
    lines.append("Current Trade Return")
    if snapshot.current_trade_entry is None:
        lines.append("- No open HYPE trade")
    else:
        lines.append(
            f"- Entry: {snapshot.current_trade_entry.date()} "
            f"({snapshot.current_trade_days} days open)"
        )
        lines.append(
            f"- HYPE return: {_fmt_pct(snapshot.current_trade_hype_return)}"
        )
        lines.append(
            f"- ETH return: {_fmt_pct(snapshot.current_trade_eth_return)}"
        )
        lines.append(
            f"- Relative outperformance: {_fmt_pct(snapshot.current_trade_relative_outperformance)}"
        )
    lines.append(divider)
    lines.append("Monitoring Status")
    lines.append(
        f"- Stop-loss rule (total relative loss vs ETH): threshold {_fmt_pct(-config.stop_loss_threshold)}, "
        f"current {_fmt_pct(snapshot.current_loss_historical)}, "
        f"trigger={'YES' if snapshot.stop_loss_triggered else 'NO'}"
    )
    lines.append(
        f"- Strategy status: {'ON' if snapshot.strategy_on else 'OFF'}"
    )
    if snapshot.warnings:
        lines.append(divider)
        lines.append("Warnings")
        for warning in snapshot.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)


__all__ = ["build_hype_eth_rotation_telegram_message"]
