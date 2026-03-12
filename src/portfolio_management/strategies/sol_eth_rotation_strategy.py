from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from portfolio_management.helpers.config import BASE_DIR

from .sol_eth_trade_log import SolEthTradeEvent, load_sol_eth_trade_log


@dataclass(frozen=True)
class SolEthRotationConfig:
    close_hour: int
    start_date: datetime | None
    sol_symbol: str
    eth_symbol: str
    fast_span: int
    slow_span: int
    entry_signal_return_window: int
    exit_signal_return_window: int
    execution_lag_bars: int
    rsi_period: int
    rsi_exit_level: float
    use_rsi_early_exit: bool
    review_streak_x: int
    review_q25_backtest_threshold: float | None
    stop_loss_threshold: float
    trade_log_path: Path


@dataclass(frozen=True)
class SolEthTrade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    hold_days: int
    entry_sol: float
    entry_eth: float
    exit_sol: float
    exit_eth: float
    entry_ratio: float
    exit_ratio: float
    trade_return_abs: float
    completed: bool


@dataclass(frozen=True)
class SolEthReviewStatus:
    streak_x: int
    completed_trade_count: int
    completed_trade_returns: tuple[float, ...]
    q25_tail_threshold: float | None
    last_streak_returns: tuple[float, ...]
    streak_triggered: bool
    cumulative_return: float | None
    current_loss_historical: float | None
    max_loss_historical: float | None
    stop_loss_threshold: float
    stop_loss_triggered: bool


@dataclass(frozen=True)
class SolEthRotationSnapshot:
    as_of: pd.Timestamp
    sol_close: float
    eth_close: float
    ratio_close: float
    ratio_ret_entry_window: float | None
    ratio_ret_exit_window: float | None
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
    signal_flip_today: bool
    early_exit_today: bool
    entry_filter_ok_today: bool
    in_position: bool
    trigger_today: bool
    current_trade: SolEthTrade | None
    previous_completed_trades: tuple[SolEthTrade, ...]
    review: SolEthReviewStatus
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


def _as_optional_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _as_float(value, name=name)


def load_sol_eth_rotation_config(raw: Mapping[str, Any]) -> SolEthRotationConfig:
    pair = raw.get("pair") or {}
    signal = raw.get("signal") or {}
    execution = raw.get("execution") or {}
    rsi = raw.get("rsi") or {}
    review = raw.get("review") or {}
    risk = raw.get("risk") or {}
    trade_log = raw.get("trade_log") or {}
    trade_log_path_raw = trade_log.get("path", "configs/state/sol_eth_trade_log.json")
    trade_log_path = Path(str(trade_log_path_raw))
    if not trade_log_path.is_absolute():
        trade_log_path = BASE_DIR / trade_log_path

    return SolEthRotationConfig(
        close_hour=_as_int(raw.get("close_hour", 12), name="close_hour"),
        start_date=_dt_from_iso(raw.get("start_date")),
        sol_symbol=str(pair.get("sol_symbol", "SOL-USD")),
        eth_symbol=str(pair.get("eth_symbol", "ETH-USD")),
        fast_span=_as_int(signal.get("fast_span", 7), name="signal.fast_span"),
        slow_span=_as_int(signal.get("slow_span", 14), name="signal.slow_span"),
        entry_signal_return_window=_as_int(
            signal.get("entry_return_window", signal.get("entry_pos_return_window", 0)),
            name="signal.entry_return_window",
        ),
        exit_signal_return_window=_as_int(
            signal.get("exit_return_window", 0),
            name="signal.exit_return_window",
        ),
        execution_lag_bars=_as_int(
            execution.get("lag_bars", 1),
            name="execution.lag_bars",
        ),
        rsi_period=_as_int(rsi.get("period", 14), name="rsi.period"),
        rsi_exit_level=_as_float(
            rsi.get("exit_level", 67.5),
            name="rsi.exit_level",
        ),
        use_rsi_early_exit=bool(rsi.get("use_early_exit", True)),
        review_streak_x=_as_int(review.get("streak_x", 3), name="review.streak_x"),
        review_q25_backtest_threshold=_as_optional_float(
            review.get("q25_backtest_threshold"),
            name="review.q25_backtest_threshold",
        ),
        stop_loss_threshold=abs(
            _as_float(risk.get("stop_loss_threshold", 0.30), name="risk.stop_loss_threshold")
        ),
        trade_log_path=trade_log_path,
    )


def _rma_tv(series: pd.Series, period: int) -> pd.Series:
    values = series.astype(float)
    out = pd.Series(index=values.index, dtype="float64")
    valid = values.dropna()
    if len(valid) < period:
        return out

    seed = float(valid.iloc[:period].mean())
    seed_idx = valid.index[period - 1]
    out.loc[seed_idx] = seed

    prev = seed
    for idx, value in valid.iloc[period:].items():
        prev = ((period - 1) * prev + float(value)) / period
        out.loc[idx] = prev
    return out


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _rma_tv(gain, period)
    avg_loss = _rma_tv(loss, period)

    rsi = pd.Series(index=series.index, dtype="float64")
    down_zero = avg_loss == 0
    up_zero = avg_gain == 0
    both_ready = avg_gain.notna() & avg_loss.notna()
    rsi.loc[both_ready & down_zero] = 100.0
    rsi.loc[both_ready & ~down_zero & up_zero] = 0.0
    core_mask = both_ready & ~down_zero & ~up_zero
    rs = avg_gain.loc[core_mask] / avg_loss.loc[core_mask]
    rsi.loc[core_mask] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _as_sol_eth_trade(
    *,
    entry_event: SolEthTradeEvent,
    exit_event: SolEthTradeEvent | None,
    as_of: pd.Timestamp,
    ratio_now: float,
    sol_prices: pd.Series,
    eth_prices: pd.Series,
) -> SolEthTrade:
    def _lookup_price(series: pd.Series, ts: pd.Timestamp, fallback: float) -> float:
        if ts in series.index:
            return float(series.loc[ts])
        if len(series.index) == 0:
            return float(fallback)
        pos = series.index.get_indexer([ts], method="nearest")[0]
        if pos < 0:
            return float(fallback)
        return float(series.iloc[pos])

    entry_ts = pd.Timestamp(entry_event.date)
    entry_ratio = float(entry_event.cross_price)
    entry_sol = _lookup_price(sol_prices, entry_ts, fallback=entry_ratio)
    entry_eth = _lookup_price(eth_prices, entry_ts, fallback=1.0)

    if exit_event is not None:
        exit_ts = pd.Timestamp(exit_event.date)
        exit_ratio = float(exit_event.cross_price)
        exit_sol = _lookup_price(sol_prices, exit_ts, fallback=exit_ratio)
        exit_eth = _lookup_price(eth_prices, exit_ts, fallback=1.0)
        completed = True
    else:
        exit_ts = pd.Timestamp(as_of)
        exit_ratio = float(ratio_now)
        exit_sol = _lookup_price(sol_prices, exit_ts, fallback=exit_ratio)
        exit_eth = _lookup_price(eth_prices, exit_ts, fallback=1.0)
        completed = False

    hold_days = int(max((exit_ts - entry_ts).days, 0))
    return SolEthTrade(
        entry_date=entry_ts,
        exit_date=pd.Timestamp(exit_ts),
        hold_days=hold_days,
        entry_sol=entry_sol,
        entry_eth=entry_eth,
        exit_sol=exit_sol,
        exit_eth=exit_eth,
        entry_ratio=entry_ratio,
        exit_ratio=exit_ratio,
        trade_return_abs=float(exit_ratio / entry_ratio - 1.0),
        completed=completed,
    )


def _build_trades_from_events(
    *,
    events: list[SolEthTradeEvent],
    as_of: pd.Timestamp,
    ratio_now: float,
    sol_prices: pd.Series,
    eth_prices: pd.Series,
) -> list[SolEthTrade]:
    trades: list[SolEthTrade] = []
    open_event: SolEthTradeEvent | None = None
    for event in events:
        if event.event == "ENTRY":
            open_event = event
            continue
        if open_event is None:
            continue
        trades.append(
            _as_sol_eth_trade(
                entry_event=open_event,
                exit_event=event,
                as_of=as_of,
                ratio_now=ratio_now,
                sol_prices=sol_prices,
                eth_prices=eth_prices,
            )
        )
        open_event = None

    if open_event is not None:
        trades.append(
            _as_sol_eth_trade(
                entry_event=open_event,
                exit_event=None,
                as_of=as_of,
                ratio_now=ratio_now,
                sol_prices=sol_prices,
                eth_prices=eth_prices,
            )
        )
    return trades


def _realized_drawdown_from_trade_returns(
    *,
    completed_returns: tuple[float, ...],
    open_trade_mtm_return: float | None,
) -> tuple[float, float]:
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


def _cumulative_return_from_trade_returns(
    *,
    completed_returns: tuple[float, ...],
    open_trade_mtm_return: float | None,
) -> float | None:
    if not completed_returns and open_trade_mtm_return is None:
        return None
    equity = 1.0
    for ret in completed_returns:
        equity *= 1.0 + float(ret)
    if open_trade_mtm_return is not None:
        equity *= 1.0 + float(open_trade_mtm_return)
    return float(equity - 1.0)


def _build_review_status(
    *,
    trades: list[SolEthTrade],
    review_streak_x: int,
    review_q25_backtest_threshold: float | None,
    current_loss_historical: float | None,
    max_loss_historical: float | None,
    cumulative_return: float | None,
    stop_loss_threshold: float,
) -> SolEthReviewStatus:
    completed = [trade for trade in trades if trade.completed]
    completed_vals: tuple[float, ...] = tuple(trade.trade_return_abs for trade in completed)
    q25: float | None = None
    last_vals: tuple[float, ...] = ()
    triggered = False

    if completed:
        vals = np.array(completed_vals, dtype=float)
        if review_q25_backtest_threshold is not None:
            q25 = float(review_q25_backtest_threshold)
        else:
            q25 = float(np.quantile(vals, 0.25))
        last_slice = vals[-review_streak_x:] if len(vals) >= review_streak_x else vals
        last_vals = tuple(float(x) for x in last_slice.tolist())
        if len(vals) >= review_streak_x and q25 is not None:
            triggered = bool(np.all(last_slice < q25))

    stop_triggered = bool(
        current_loss_historical is not None and current_loss_historical <= -abs(float(stop_loss_threshold))
    )

    return SolEthReviewStatus(
        streak_x=review_streak_x,
        completed_trade_count=len(completed_vals),
        completed_trade_returns=completed_vals,
        q25_tail_threshold=q25,
        last_streak_returns=last_vals,
        streak_triggered=triggered,
        cumulative_return=cumulative_return,
        current_loss_historical=current_loss_historical,
        max_loss_historical=max_loss_historical,
        stop_loss_threshold=abs(float(stop_loss_threshold)),
        stop_loss_triggered=stop_triggered,
    )


def generate_sol_eth_rotation_snapshot(
    *,
    sol_close: pd.Series,
    eth_close: pd.Series,
    config: SolEthRotationConfig,
) -> SolEthRotationSnapshot:
    idx = sol_close.index.intersection(eth_close.index).sort_values()
    if idx.empty:
        raise ValueError("No overlapping OHLCV dates for SOL and ETH.")

    sol = sol_close.loc[idx].astype(float)
    eth = eth_close.loc[idx].astype(float)
    if config.start_date is not None:
        start_ts = pd.Timestamp(config.start_date)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        keep = idx >= start_ts
        if not bool(keep.any()):
            raise ValueError("No SOL/ETH rows available after configured start_date.")
        idx = idx[keep]
        sol = sol.loc[idx]
        eth = eth.loc[idx]

    price_ratio = (sol / eth).rename("sol_eth_price_ratio")
    sol_eq = (sol / float(sol.iloc[0])).rename("sol_equity")
    eth_eq = (eth / float(eth.iloc[0])).rename("eth_equity")
    ratio = (sol_eq / eth_eq).rename("sol_eth_ratio")

    if config.entry_signal_return_window == 0:
        ratio_ret_entry_window = pd.Series(0.0, index=ratio.index, dtype=float)
    else:
        ratio_ret_entry_window = ratio.pct_change(config.entry_signal_return_window)

    if config.exit_signal_return_window == 0:
        ratio_ret_exit_window = pd.Series(0.0, index=ratio.index, dtype=float)
    else:
        ratio_ret_exit_window = ratio.pct_change(config.exit_signal_return_window)

    ema_fast = ratio.ewm(span=config.fast_span, adjust=False, min_periods=config.fast_span).mean()
    ema_slow = ratio.ewm(span=config.slow_span, adjust=False, min_periods=config.slow_span).mean()
    price_ratio_ema_fast = price_ratio.ewm(span=config.fast_span, adjust=False, min_periods=config.fast_span).mean()
    price_ratio_ema_slow = price_ratio.ewm(span=config.slow_span, adjust=False, min_periods=config.slow_span).mean()

    base_target_signal = pd.Series(0.0, index=ratio.index, dtype=float)
    current = 1.0
    for ts in ratio.index:
        spread = float(ema_fast.loc[ts] - ema_slow.loc[ts]) if pd.notna(ema_fast.loc[ts]) and pd.notna(ema_slow.loc[ts]) else float("nan")
        rw_entry = ratio_ret_entry_window.loc[ts]
        rw_exit = ratio_ret_exit_window.loc[ts]
        if pd.isna(spread):
            base_target_signal.loc[ts] = current
            continue
        can_enter = config.entry_signal_return_window == 0 or (pd.notna(rw_entry) and float(rw_entry) > 0.0)
        can_exit = config.exit_signal_return_window == 0 or (pd.notna(rw_exit) and float(rw_exit) < 0.0)
        if current <= 0.0 and spread > 0.0 and can_enter:
            current = 1.0
        elif current > 0.0 and spread < 0.0 and can_exit:
            current = 0.0
        base_target_signal.loc[ts] = current

    rsi = _compute_rsi(ratio, config.rsi_period)
    early_exit_target = pd.Series(False, index=ratio.index, dtype=bool)
    rsi_armed = False
    prev_base_target = base_target_signal.shift(1)
    for ts in ratio.index:
        prev_target = prev_base_target.loc[ts]
        rsi_now = rsi.loc[ts]
        if pd.isna(prev_target) or float(prev_target) <= 0.0:
            rsi_armed = False
        if pd.notna(rsi_now) and pd.notna(prev_target) and float(prev_target) > 0.0:
            if float(rsi_now) >= config.rsi_exit_level:
                rsi_armed = True
            if rsi_armed and float(rsi_now) <= config.rsi_exit_level:
                early_exit_target.loc[ts] = True
                rsi_armed = False

    signal_flip_event = base_target_signal != base_target_signal.shift(1)
    if config.use_rsi_early_exit:
        alloc_signal = pd.Series(0.0, index=ratio.index, dtype=float)
        current_pos = 1.0
        for ts in ratio.index:
            if bool(signal_flip_event.loc[ts]):
                current_pos = float(base_target_signal.loc[ts])
            elif current_pos > 0.0 and bool(early_exit_target.loc[ts]):
                current_pos = 0.0
            alloc_signal.loc[ts] = current_pos
    else:
        alloc_signal = base_target_signal.copy()

    alloc_to_sol = alloc_signal.shift(config.execution_lag_bars).bfill().fillna(1.0)
    as_of = idx[-1]
    signal_pos_now = bool(alloc_to_sol.loc[as_of] > 0.5)
    trigger_today = False
    prev_as_of = idx[-2] if len(idx) >= 2 else None
    if len(idx) >= 2:
        trigger_today = bool(abs(float(alloc_to_sol.loc[as_of] - alloc_to_sol.loc[prev_as_of])) > 1e-12)

    log_events, log_warnings = load_sol_eth_trade_log(
        config.trade_log_path,
        close_hour=config.close_hour,
    )
    warnings = list(log_warnings)
    trades = _build_trades_from_events(
        events=log_events,
        as_of=pd.Timestamp(as_of),
        ratio_now=float(price_ratio.loc[as_of]),
        sol_prices=sol,
        eth_prices=eth,
    )
    completed = [trade for trade in trades if trade.completed]
    open_trade = next((trade for trade in reversed(trades) if not trade.completed), None)
    prev_three = tuple(reversed(completed[-3:]))

    pos_from_log = open_trade is not None
    if pos_from_log != signal_pos_now and trigger_today:
        warn = (
            "Reconciliation mismatch: "
            f"log_position={'LONG' if pos_from_log else 'FLAT'} "
            f"vs signal_position={'LONG' if signal_pos_now else 'FLAT'} "
            f"(trigger_today={'YES' if trigger_today else 'NO'})."
        )
        warnings.append(warn)

    completed_returns = tuple(tr.trade_return_abs for tr in completed)
    open_mtm = open_trade.trade_return_abs if open_trade is not None else None
    current_loss, max_loss = _realized_drawdown_from_trade_returns(
        completed_returns=completed_returns,
        open_trade_mtm_return=open_mtm,
    )
    cumulative_return = _cumulative_return_from_trade_returns(
        completed_returns=completed_returns,
        open_trade_mtm_return=open_mtm,
    )
    review = _build_review_status(
        trades=trades,
        review_streak_x=config.review_streak_x,
        review_q25_backtest_threshold=config.review_q25_backtest_threshold,
        current_loss_historical=current_loss,
        max_loss_historical=max_loss,
        cumulative_return=cumulative_return,
        stop_loss_threshold=config.stop_loss_threshold,
    )

    return SolEthRotationSnapshot(
        as_of=pd.Timestamp(as_of),
        sol_close=float(sol.loc[as_of]),
        eth_close=float(eth.loc[as_of]),
        ratio_close=float(ratio.loc[as_of]),
        ratio_ret_entry_window=float(ratio_ret_entry_window.loc[as_of]) if pd.notna(ratio_ret_entry_window.loc[as_of]) else None,
        ratio_ret_exit_window=float(ratio_ret_exit_window.loc[as_of]) if pd.notna(ratio_ret_exit_window.loc[as_of]) else None,
        ema_fast=float(ema_fast.loc[as_of]),
        ema_slow=float(ema_slow.loc[as_of]),
        ema_fast_prev=float(ema_fast.loc[prev_as_of]) if prev_as_of is not None and pd.notna(ema_fast.loc[prev_as_of]) else None,
        ema_slow_prev=float(ema_slow.loc[prev_as_of]) if prev_as_of is not None and pd.notna(ema_slow.loc[prev_as_of]) else None,
        price_ratio_ema_fast=float(price_ratio_ema_fast.loc[as_of]),
        price_ratio_ema_slow=float(price_ratio_ema_slow.loc[as_of]),
        price_ratio_ema_fast_prev=float(price_ratio_ema_fast.loc[prev_as_of]) if prev_as_of is not None and pd.notna(price_ratio_ema_fast.loc[prev_as_of]) else None,
        price_ratio_ema_slow_prev=float(price_ratio_ema_slow.loc[prev_as_of]) if prev_as_of is not None and pd.notna(price_ratio_ema_slow.loc[prev_as_of]) else None,
        rsi=float(rsi.loc[as_of]) if pd.notna(rsi.loc[as_of]) else float("nan"),
        rsi_prev=float(rsi.loc[prev_as_of]) if prev_as_of is not None and pd.notna(rsi.loc[prev_as_of]) else None,
        signal_flip_today=bool(signal_flip_event.loc[as_of]),
        early_exit_today=bool(early_exit_target.loc[as_of]),
        entry_filter_ok_today=bool(
            config.entry_signal_return_window == 0
            or (
                pd.notna(ratio_ret_entry_window.loc[as_of])
                and float(ratio_ret_entry_window.loc[as_of]) > 0.0
            )
        ),
        in_position=pos_from_log,
        trigger_today=trigger_today,
        current_trade=open_trade,
        previous_completed_trades=prev_three,
        review=review,
        warnings=tuple(warnings),
    )


__all__ = [
    "SolEthRotationConfig",
    "SolEthRotationSnapshot",
    "SolEthTrade",
    "SolEthReviewStatus",
    "load_sol_eth_rotation_config",
    "generate_sol_eth_rotation_snapshot",
]
