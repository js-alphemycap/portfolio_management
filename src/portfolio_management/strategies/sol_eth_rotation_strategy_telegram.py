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
    return "LONG SOL / SHORT ETH" if in_position else "FLAT (passive ETH)"


def build_sol_eth_rotation_telegram_message(
    *,
    snapshot: SolEthRotationSnapshot,
    config: SolEthRotationConfig,
) -> str:
    lines: list[str] = []
    divider = "-" * 36

    lines.append(f"SOL/ETH Rotation Signal - {snapshot.as_of.date()}")
    lines.append(divider)
    lines.append("State")
    lines.append(f"- Position: {_position_text(snapshot.in_position)}")
    lines.append(f"- Trigger today: {'YES' if snapshot.trigger_today else 'NO'}")
    lines.append(
        f"- Now: SOL {_fmt_price(snapshot.sol_close)} | ETH {_fmt_price(snapshot.eth_close)} | "
        f"Ratio {_fmt_ratio(snapshot.ratio_close)}"
    )
    lines.append(
        f"- EMA: fast({_config_fast(config)}) {_fmt_ratio(snapshot.ema_fast)} vs "
        f"slow({_config_slow(config)}) {_fmt_ratio(snapshot.ema_slow)}"
    )
    lines.append(divider)
    if snapshot.in_position:
        lines.append("Exit criteria")
        lines.append(f"- Exit if EMA{config.fast_span} < EMA{config.slow_span}")
        lines.append(f"- Exit signal today: {'YES' if snapshot.bear_cross_today else 'NO'}")
    else:
        lines.append("Entry criteria (both true)")
        lines.append(
            f"- EMA{config.fast_span} > EMA{config.slow_span}: "
            f"{'YES' if (snapshot.ema_fast > snapshot.ema_slow) else 'NO'}"
        )
        if config.entry_pos_return_window > 0:
            lines.append(
                f"- SOL/ETH {config.entry_pos_return_window}D return > 0: "
                f"{'YES' if snapshot.entry_filter_ok_today else 'NO'}"
            )
        else:
            lines.append("- Filter: OFF")
    lines.append(divider)
    lines.append("Current trade")
    if snapshot.current_trade is None:
        lines.append("- No open trade")
    else:
        current = snapshot.current_trade
        lines.append(
            f"- Entry: {current.entry_date.date()} | Ratio {_fmt_ratio(current.entry_ratio)}"
        )
        lines.append(
            f"- Current: {snapshot.as_of.date()} | Ratio {_fmt_ratio(snapshot.ratio_close)}"
        )
        lines.append(f"- Return: {_fmt_pct(current.trade_return_abs)}")
    lines.append(divider)
    lines.append("Monitoring status")
    review = snapshot.review
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
    lines.append(f"- Strategy status: {'OK to run' if strategy_ok else 'NOT OK to run'}")
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
