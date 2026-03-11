from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from portfolio_management.helpers.config import BASE_DIR

from .hype_eth_trade_log import HypeEthTradeEvent, load_hype_eth_trade_log


@dataclass(frozen=True)
class HypeEthRotationConfig:
    close_hour: int
    start_date: datetime | None
    hype_symbol: str
    eth_symbol: str
    fast_span: int
    slow_span: int
    execution_lag_bars: int
    signal_return_window: int
    rsi_period: int
    rsi_long_exit_level: float
    rsi_short_exit_level: float
    use_rsi_early_exit: bool
    trading_fee_bps_per_leg: float
    slippage_bps_per_leg: float
    stop_loss_threshold: float
    trade_log_path: Path


@dataclass(frozen=True)
class HypeEthRotationSnapshot:
    as_of: pd.Timestamp
    hype_close: float
    eth_close: float
    ratio_close: float
    price_ratio_close: float
    ratio_return_window: float | None
    ema_fast: float
    ema_slow: float
    ema_fast_prev: float | None
    ema_slow_prev: float | None
    price_ratio_ema_fast: float
    price_ratio_ema_slow: float
    price_ratio_ema_fast_prev: float | None
    price_ratio_ema_slow_prev: float | None
    rsi: float
    rsi_prev: float | None
    desired_side: float
    side_filtered: float
    alloc_signal: float
    alloc_after_lag: float
    in_position: bool
    trigger_today: bool
    signal_flip_today: bool
    early_exit_today: bool
    entry_filter_ok_today: bool
    current_trade_entry: pd.Timestamp | None
    current_trade_days: int | None
    current_trade_hype_return: float | None
    current_trade_eth_return: float | None
    current_trade_relative_outperformance: float | None
    current_loss_historical: float | None
    max_loss_historical: float | None
    stop_loss_triggered: bool
    strategy_on: bool
    warnings: tuple[str, ...]


def _dt_from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _as_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid int for {name}: {value!r}") from exc


def _as_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid float for {name}: {value!r}") from exc


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid bool for {name}: {value!r}")


def load_hype_eth_rotation_config(raw: Mapping[str, Any]) -> HypeEthRotationConfig:
    pair = raw.get("pair") or {}
    signal = raw.get("signal") or {}
    execution = raw.get("execution") or {}
    rsi = raw.get("rsi") or {}
    costs = raw.get("costs") or {}
    risk = raw.get("risk") or {}
    trade_log = raw.get("trade_log") or {}

    trade_log_path_raw = trade_log.get("path", "configs/state/hype_eth_trade_log.json")
    trade_log_path = Path(str(trade_log_path_raw))
    if not trade_log_path.is_absolute():
        trade_log_path = BASE_DIR / trade_log_path

    return HypeEthRotationConfig(
        close_hour=_as_int(raw.get("close_hour", 12), name="close_hour"),
        start_date=_dt_from_iso(raw.get("start_date")),
        hype_symbol=str(pair.get("hype_symbol", "HYPE-USD")),
        eth_symbol=str(pair.get("eth_symbol", "ETH-USD")),
        fast_span=_as_int(signal.get("fast_span", 7), name="signal.fast_span"),
        slow_span=_as_int(signal.get("slow_span", 14), name="signal.slow_span"),
        execution_lag_bars=_as_int(execution.get("lag_bars", 1), name="execution.lag_bars"),
        signal_return_window=_as_int(
            signal.get("return_window", 1),
            name="signal.return_window",
        ),
        rsi_period=_as_int(rsi.get("period", 14), name="rsi.period"),
        rsi_long_exit_level=_as_float(
            rsi.get("long_exit_level", 60),
            name="rsi.long_exit_level",
        ),
        rsi_short_exit_level=_as_float(
            rsi.get("short_exit_level", 40),
            name="rsi.short_exit_level",
        ),
        use_rsi_early_exit=_as_bool(
            rsi.get("use_early_exit", True),
            name="rsi.use_early_exit",
        ),
        trading_fee_bps_per_leg=_as_float(
            costs.get("trading_fee_bps_per_leg", 0.0),
            name="costs.trading_fee_bps_per_leg",
        ),
        slippage_bps_per_leg=_as_float(
            costs.get("slippage_bps_per_leg", 0.0),
            name="costs.slippage_bps_per_leg",
        ),
        stop_loss_threshold=abs(
            _as_float(risk.get("stop_loss_threshold", 0.30), name="risk.stop_loss_threshold")
        ),
        trade_log_path=trade_log_path,
    )


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def _lookup_price(series: pd.Series, ts: pd.Timestamp) -> float:
    if ts in series.index:
        return float(series.loc[ts])
    if len(series.index) == 0:
        raise ValueError("Price series is empty.")
    pos = series.index.get_indexer([ts], method="nearest")[0]
    if pos < 0:
        raise ValueError("Could not locate price for trade log timestamp.")
    return float(series.iloc[pos])


def _realized_drawdown_from_trade_returns(
    *,
    completed_returns: tuple[float, ...],
    open_trade_mtm_return: float | None,
) -> tuple[float | None, float | None]:
    if not completed_returns and open_trade_mtm_return is None:
        return None, None
    equity = 1.0
    peak = 1.0
    current_dd = 0.0
    worst_dd = 0.0

    for ret in completed_returns:
        equity *= 1.0 + float(ret)
        peak = max(peak, equity)
        current_dd = min(float(equity / peak - 1.0), 0.0)
        worst_dd = min(worst_dd, current_dd)

    if open_trade_mtm_return is not None:
        equity_open = equity * (1.0 + float(open_trade_mtm_return))
        peak = max(peak, equity_open)
        current_dd = min(float(equity_open / peak - 1.0), 0.0)
        worst_dd = min(worst_dd, current_dd)

    return current_dd, worst_dd


def generate_hype_eth_rotation_snapshot(
    *,
    hype_close: pd.Series,
    eth_close: pd.Series,
    config: HypeEthRotationConfig,
) -> HypeEthRotationSnapshot:
    idx = hype_close.index.intersection(eth_close.index).sort_values()
    if idx.empty:
        raise ValueError("No overlapping OHLCV dates for HYPE and ETH.")

    hype = hype_close.loc[idx].astype(float)
    eth = eth_close.loc[idx].astype(float)

    if config.start_date is not None:
        start_ts = pd.Timestamp(config.start_date)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        keep = idx >= start_ts
        if not bool(keep.any()):
            raise ValueError("No HYPE/ETH rows available after configured start_date.")
        idx = idx[keep]
        hype = hype.loc[idx]
        eth = eth.loc[idx]

    price_ratio = (hype / eth).rename("hype_eth_price_ratio")

    hype_eq = (hype / float(hype.iloc[0])).rename("hype_equity")
    eth_eq = (eth / float(eth.iloc[0])).rename("eth_equity")
    ratio = (hype_eq / eth_eq).rename("hype_eth_ratio")

    ratio_return_window = ratio.pct_change(config.signal_return_window)
    if config.signal_return_window == 0:
        ratio_return_window = pd.Series(0.0, index=ratio.index, dtype=float)

    ema_fast = ratio.ewm(
        span=config.fast_span,
        adjust=False,
        min_periods=config.fast_span,
    ).mean()
    ema_slow = ratio.ewm(
        span=config.slow_span,
        adjust=False,
        min_periods=config.slow_span,
    ).mean()
    price_ratio_ema_fast = price_ratio.ewm(
        span=config.fast_span,
        adjust=False,
        min_periods=config.fast_span,
    ).mean()
    price_ratio_ema_slow = price_ratio.ewm(
        span=config.slow_span,
        adjust=False,
        min_periods=config.slow_span,
    ).mean()

    desired_side = pd.Series(1.0, index=ratio.index, dtype=float)
    desired_side.loc[ema_fast <= ema_slow] = -1.0

    side_filtered = pd.Series(0.0, index=ratio.index, dtype=float)
    current = 1.0
    for ts in ratio.index:
        desired = float(desired_side.loc[ts])
        rw = ratio_return_window.loc[ts]
        if config.signal_return_window == 0:
            can_flip_long = True
            can_flip_short = True
        else:
            can_flip_long = pd.notna(rw) and float(rw) > 0.0
            can_flip_short = pd.notna(rw) and float(rw) < 0.0
        if desired != current:
            if desired > 0.0 and can_flip_long:
                current = 1.0
            elif desired < 0.0 and can_flip_short:
                current = -1.0
        side_filtered.loc[ts] = current

    rsi = _compute_rsi(ratio, config.rsi_period)
    early_exit_long = (rsi.shift(1) > config.rsi_long_exit_level) & (rsi <= config.rsi_long_exit_level)
    early_exit_short = (rsi.shift(1) < config.rsi_short_exit_level) & (rsi >= config.rsi_short_exit_level)
    signal_flip_event = side_filtered != side_filtered.shift(1)

    if config.use_rsi_early_exit:
        alloc_signal = pd.Series(0.0, index=ratio.index, dtype=float)
        current_pos = 1.0
        for ts in ratio.index:
            if bool(signal_flip_event.loc[ts]):
                current_pos = float(side_filtered.loc[ts])
            else:
                if current_pos == 1.0 and bool(early_exit_long.loc[ts]):
                    current_pos = 0.0
                elif current_pos == -1.0 and bool(early_exit_short.loc[ts]):
                    current_pos = 0.0
            alloc_signal.loc[ts] = current_pos
    else:
        alloc_signal = side_filtered.copy()

    alloc_after_lag = alloc_signal.shift(config.execution_lag_bars).bfill().fillna(1.0)

    as_of = idx[-1]
    prev_as_of = idx[-2] if len(idx) >= 2 else None
    trigger_today = bool(abs(float(alloc_after_lag.loc[as_of] - alloc_after_lag.loc[prev_as_of])) > 1e-12) if prev_as_of is not None else False

    signal_flip_today = bool(signal_flip_event.loc[as_of])
    early_exit_today = bool(early_exit_long.loc[as_of] or early_exit_short.loc[as_of])
    entry_filter_ok_today = True
    rw_now = ratio_return_window.loc[as_of]
    if config.signal_return_window > 0:
        entry_filter_ok_today = bool(pd.notna(rw_now) and float(rw_now) > 0.0)

    signal_pos_now = bool(float(alloc_after_lag.loc[as_of]) > 0.0)
    log_events, log_warnings = load_hype_eth_trade_log(
        config.trade_log_path,
        close_hour=config.close_hour,
    )
    warnings = list(log_warnings)

    completed_returns: list[float] = []
    open_entry_event: HypeEthTradeEvent | None = None
    pending_entry: HypeEthTradeEvent | None = None
    for event in log_events:
        if event.event == "ENTRY":
            pending_entry = event
            continue
        if pending_entry is None:
            continue
        completed_returns.append(float(event.cross_price / pending_entry.cross_price - 1.0))
        pending_entry = None
    if pending_entry is not None:
        open_entry_event = pending_entry

    in_position_from_log = open_entry_event is not None
    if in_position_from_log != signal_pos_now:
        warnings.append(
            "Reconciliation mismatch: "
            f"log_position={'IN TRADE' if in_position_from_log else 'OFF TRADE'} "
            f"vs signal_position={'IN TRADE' if signal_pos_now else 'OFF TRADE'}."
        )

    current_trade_entry: pd.Timestamp | None = None
    current_trade_days: int | None = None
    current_trade_hype_return: float | None = None
    current_trade_eth_return: float | None = None
    current_trade_relative_outperformance: float | None = None
    open_trade_mtm_return: float | None = None
    if open_entry_event is not None:
        current_trade_entry = pd.Timestamp(open_entry_event.date)
        entry_hype = float(open_entry_event.hype_price)
        entry_eth = float(open_entry_event.eth_price)
        now_hype = float(hype.loc[as_of])
        now_eth = float(eth.loc[as_of])
        current_trade_hype_return = now_hype / entry_hype - 1.0
        current_trade_eth_return = now_eth / entry_eth - 1.0
        current_trade_relative_outperformance = (
            (1.0 + current_trade_hype_return) / (1.0 + current_trade_eth_return) - 1.0
        )
        open_trade_mtm_return = float(price_ratio.loc[as_of] / float(open_entry_event.cross_price) - 1.0)
        current_trade_days = int(max((pd.Timestamp(as_of) - current_trade_entry).days, 0))

    current_loss_historical, max_loss_historical = _realized_drawdown_from_trade_returns(
        completed_returns=tuple(completed_returns),
        open_trade_mtm_return=open_trade_mtm_return,
    )
    stop_loss_triggered = bool(
        current_loss_historical is not None and current_loss_historical <= -abs(float(config.stop_loss_threshold))
    )
    strategy_on = not stop_loss_triggered

    return HypeEthRotationSnapshot(
        as_of=pd.Timestamp(as_of),
        hype_close=float(hype.loc[as_of]),
        eth_close=float(eth.loc[as_of]),
        ratio_close=float(ratio.loc[as_of]),
        price_ratio_close=float(price_ratio.loc[as_of]),
        ratio_return_window=float(rw_now) if pd.notna(rw_now) else None,
        ema_fast=float(ema_fast.loc[as_of]),
        ema_slow=float(ema_slow.loc[as_of]),
        ema_fast_prev=float(ema_fast.loc[prev_as_of]) if prev_as_of is not None and pd.notna(ema_fast.loc[prev_as_of]) else None,
        ema_slow_prev=float(ema_slow.loc[prev_as_of]) if prev_as_of is not None and pd.notna(ema_slow.loc[prev_as_of]) else None,
        price_ratio_ema_fast=float(price_ratio_ema_fast.loc[as_of]),
        price_ratio_ema_slow=float(price_ratio_ema_slow.loc[as_of]),
        price_ratio_ema_fast_prev=float(price_ratio_ema_fast.loc[prev_as_of]) if prev_as_of is not None and pd.notna(price_ratio_ema_fast.loc[prev_as_of]) else None,
        price_ratio_ema_slow_prev=float(price_ratio_ema_slow.loc[prev_as_of]) if prev_as_of is not None and pd.notna(price_ratio_ema_slow.loc[prev_as_of]) else None,
        rsi=float(rsi.loc[as_of]),
        rsi_prev=float(rsi.loc[prev_as_of]) if prev_as_of is not None and pd.notna(rsi.loc[prev_as_of]) else None,
        desired_side=float(desired_side.loc[as_of]),
        side_filtered=float(side_filtered.loc[as_of]),
        alloc_signal=float(alloc_signal.loc[as_of]),
        alloc_after_lag=float(alloc_after_lag.loc[as_of]),
        in_position=in_position_from_log,
        trigger_today=trigger_today,
        signal_flip_today=signal_flip_today,
        early_exit_today=early_exit_today,
        entry_filter_ok_today=entry_filter_ok_today,
        current_trade_entry=current_trade_entry,
        current_trade_days=current_trade_days,
        current_trade_hype_return=current_trade_hype_return,
        current_trade_eth_return=current_trade_eth_return,
        current_trade_relative_outperformance=current_trade_relative_outperformance,
        current_loss_historical=current_loss_historical,
        max_loss_historical=max_loss_historical,
        stop_loss_triggered=stop_loss_triggered,
        strategy_on=strategy_on,
        warnings=tuple(warnings),
    )


__all__ = [
    "HypeEthRotationConfig",
    "HypeEthRotationSnapshot",
    "load_hype_eth_rotation_config",
    "generate_hype_eth_rotation_snapshot",
]
