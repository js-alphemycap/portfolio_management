from __future__ import annotations

import pandas as pd

from .sol_eth_rotation_strategy import SolEthRotationConfig, SolEthRotationSnapshot


def _fmt_price(value: float) -> str:
    x = float(value)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:,.2f}"
    return f"{x:,.4f}"


def _fmt_ratio(value: float) -> str:
    return f"{float(value):,.6f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    pct = float(value) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.{digits}f}%"


def _position_text(in_position: bool) -> str:
    return "IN TRADE (SOL)" if in_position else "OFF TRADE (ETH)"


def build_sol_eth_rotation_telegram_message(
    *,
    snapshot: SolEthRotationSnapshot,
    config: SolEthRotationConfig,
) -> str:
    lines: list[str] = []
    divider = "-" * 36

    lines.append(f"SOL/ETH Rotation Signal - {snapshot.as_of.date()}")
    lines.append(divider)
    lines.append(f"- State: {_position_text(snapshot.in_position)}")
    lines.append(f"- Trigger today: {'YES' if snapshot.trigger_today else 'NO'}")
    if snapshot.in_position:
        action_text = "EXIT TO ETH" if snapshot.trigger_today else "HOLD SOL"
    else:
        entry_signal = bool(
            snapshot.signal_flip_today
            and snapshot.entry_filter_ok_today
            and snapshot.ema_fast > snapshot.ema_slow
            and not (snapshot.review.streak_triggered or snapshot.review.stop_loss_triggered)
        )
        action_text = "ENTER SOL TRADE" if entry_signal else "STAY IN ETH"
    lines.append(f"- Action: {action_text}")
    lines.append(
        f"- Now: SOL {_fmt_price(snapshot.sol_close)} | ETH {_fmt_price(snapshot.eth_close)} | "
        f"Ratio {_fmt_ratio(snapshot.ratio_close)}"
    )
    lines.append(
        f"- Price-ratio EMA today: fast({_config_fast(config)}) {_fmt_ratio(snapshot.price_ratio_ema_fast)} "
        f"{'>' if snapshot.price_ratio_ema_fast > snapshot.price_ratio_ema_slow else '<='} "
        f"slow({_config_slow(config)}) {_fmt_ratio(snapshot.price_ratio_ema_slow)}"
    )
    if snapshot.price_ratio_ema_fast_prev is not None and snapshot.price_ratio_ema_slow_prev is not None:
        lines.append(
            f"- Price-ratio EMA yesterday: fast({_config_fast(config)}) {_fmt_ratio(snapshot.price_ratio_ema_fast_prev)} "
            f"{'>' if snapshot.price_ratio_ema_fast_prev > snapshot.price_ratio_ema_slow_prev else '<='} "
            f"slow({_config_slow(config)}) {_fmt_ratio(snapshot.price_ratio_ema_slow_prev)}"
        )
    if snapshot.rsi_prev is not None:
        lines.append(
            f"- RSI({config.rsi_period}) today / yesterday: "
            f"{snapshot.rsi:.1f} / {snapshot.rsi_prev:.1f}"
        )
    else:
        lines.append(f"- RSI({config.rsi_period}) today: {snapshot.rsi:.1f}")
    lines.append(divider)
    if snapshot.in_position:
        lines.append("Exit Criteria Check")
        lines.append(
            f"- EMA exit condition (fast ratio EMA < slow ratio EMA): "
            f"{'YES' if snapshot.ema_fast < snapshot.ema_slow else 'NO'}"
        )
        if config.use_rsi_early_exit:
            lines.append(
                f"- RSI early-exit condition (cross down {config.rsi_exit_level:.1f}): "
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
            f"{'YES' if (snapshot.ema_fast > snapshot.ema_slow) else 'NO'}"
        )
        if config.entry_signal_return_window > 0:
            lines.append(
                f"- Entry return filter ({config.entry_signal_return_window}D > 0): "
                f"{_fmt_pct(snapshot.ratio_ret_entry_window)} -> "
                f"{'PASS' if snapshot.entry_filter_ok_today else 'BLOCK'}"
            )
        lines.append(
            f"- Entry signal today: "
            f"{'YES' if (snapshot.signal_flip_today and snapshot.entry_filter_ok_today and snapshot.ema_fast > snapshot.ema_slow and not (snapshot.review.streak_triggered or snapshot.review.stop_loss_triggered)) else 'NO'}"
        )
    lines.append(divider)
    lines.append("Current Trade Return")
    if snapshot.current_trade is None:
        lines.append("- No open SOL trade")
    else:
        current = snapshot.current_trade
        lines.append(
            f"- Entry: {current.entry_date.date()} "
            f"({current.hold_days} days open)"
        )
        lines.append(
            f"- SOL return: {_fmt_pct(current.exit_sol / current.entry_sol - 1.0)}"
        )
        lines.append(
            f"- ETH return: {_fmt_pct(current.exit_eth / current.entry_eth - 1.0)}"
        )
        lines.append(f"- Relative outperformance: {_fmt_pct(current.trade_return_abs)}")
    lines.append(divider)
    lines.append("Monitoring Status")
    review = snapshot.review
    lines.append(
        f"- Cumulative return: {_fmt_pct(review.cumulative_return)}"
    )
    q25_text = _fmt_pct(review.q25_tail_threshold) if review.q25_tail_threshold is not None else "NA"
    streak_vals = ", ".join(_fmt_pct(x) for x in review.last_streak_returns) if review.last_streak_returns else "NA"
    lines.append(
        f"- Streak rule: x={review.streak_x}, q25={q25_text}, last={streak_vals}, "
        f"trigger={'YES' if review.streak_triggered else 'NO'}"
    )
    lines.append(
        f"- Stop-loss rule: threshold {_fmt_pct(-review.stop_loss_threshold)}, "
        f"current {_fmt_pct(review.current_loss_historical)}, "
        f"trigger={'YES' if review.stop_loss_triggered else 'NO'}"
    )
    strategy_ok = not (review.streak_triggered or review.stop_loss_triggered)
    lines.append(f"- Strategy status: {'ON' if strategy_ok else 'OFF'}")
    if snapshot.warnings:
        lines.append(divider)
        lines.append("Warnings")
        for warning in snapshot.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def _config_fast(config: SolEthRotationConfig) -> int:
    return int(config.fast_span)


def _config_slow(config: SolEthRotationConfig) -> int:
    return int(config.slow_span)


__all__ = ["build_sol_eth_rotation_telegram_message"]
