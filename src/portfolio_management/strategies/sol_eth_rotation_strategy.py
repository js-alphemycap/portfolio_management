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
    entry_pos_return_window: int
    execution_lag_bars: int
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
    ema_fast: float
    ema_slow: float
    bull_entry_today: bool
    bear_cross_today: bool
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
        fast_span=_as_int(signal.get("fast_span", 9), name="signal.fast_span"),
        slow_span=_as_int(signal.get("slow_span", 14), name="signal.slow_span"),
        entry_pos_return_window=_as_int(
            signal.get("entry_pos_return_window", 3),
            name="signal.entry_pos_return_window",
        ),
        execution_lag_bars=_as_int(
            execution.get("lag_bars", 0),
            name="execution.lag_bars",
        ),
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


def _build_signal(
    ratio: pd.Series,
    *,
    fast_span: int,
    slow_span: int,
    entry_pos_return_window: int,
) -> pd.DataFrame:
    ema_fast = ratio.ewm(span=fast_span, adjust=False, min_periods=fast_span).mean()
    ema_slow = ratio.ewm(span=slow_span, adjust=False, min_periods=slow_span).mean()

    above = ema_fast > ema_slow
    bull_cross = above & (~above.shift(1).fillna(False))
    bear_cross = (~above) & (above.shift(1).fillna(False))

    pos_return_ok = pd.Series(True, index=ratio.index)
    if entry_pos_return_window > 0:
        pos_return_ok = (ratio.pct_change(entry_pos_return_window) > 0.0).fillna(False)
    bull_entry = bull_cross & pos_return_ok

    state = pd.Series(0.0, index=ratio.index)
    current = 0.0
    for ts in ratio.index:
        if bool(bull_entry.loc[ts]):
            current = 1.0
        elif bool(bear_cross.loc[ts]):
            current = 0.0
        state.loc[ts] = current

    return pd.DataFrame(
        {
            "ratio": ratio,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "bull_cross": bull_cross.astype(float),
            "bull_entry": bull_entry.astype(float),
            "bear_cross": bear_cross.astype(float),
            "entry_filter_pos_return_ok": pos_return_ok.astype(float),
            "alloc_to_sol_raw": state,
        }
    )


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


def _build_review_status(
    *,
    trades: list[SolEthTrade],
    review_streak_x: int,
    review_q25_backtest_threshold: float | None,
    current_loss_historical: float | None,
    max_loss_historical: float | None,
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
    ratio = (sol / eth).rename("sol_eth_ratio")

    signal_df = _build_signal(
        ratio,
        fast_span=config.fast_span,
        slow_span=config.slow_span,
        entry_pos_return_window=config.entry_pos_return_window,
    )
    alloc_to_sol = signal_df["alloc_to_sol_raw"].shift(config.execution_lag_bars).fillna(0.0)
    as_of = idx[-1]
    signal_pos_now = bool(alloc_to_sol.loc[as_of] > 0.5)
    trigger_today = False
    if len(idx) >= 2:
        prev_as_of = idx[-2]
        trigger_today = bool(abs(float(alloc_to_sol.loc[as_of] - alloc_to_sol.loc[prev_as_of])) > 1e-12)

    log_events, log_warnings = load_sol_eth_trade_log(
        config.trade_log_path,
        close_hour=config.close_hour,
    )
    warnings = list(log_warnings)
    trades = _build_trades_from_events(
        events=log_events,
        as_of=pd.Timestamp(as_of),
        ratio_now=float(ratio.loc[as_of]),
        sol_prices=sol,
        eth_prices=eth,
    )
    completed = [trade for trade in trades if trade.completed]
    open_trade = next((trade for trade in reversed(trades) if not trade.completed), None)
    prev_three = tuple(reversed(completed[-3:]))

    pos_from_log = open_trade is not None
    if pos_from_log != signal_pos_now:
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
    review = _build_review_status(
        trades=trades,
        review_streak_x=config.review_streak_x,
        review_q25_backtest_threshold=config.review_q25_backtest_threshold,
        current_loss_historical=current_loss,
        max_loss_historical=max_loss,
        stop_loss_threshold=config.stop_loss_threshold,
    )

    return SolEthRotationSnapshot(
        as_of=pd.Timestamp(as_of),
        sol_close=float(sol.loc[as_of]),
        eth_close=float(eth.loc[as_of]),
        ratio_close=float(ratio.loc[as_of]),
        ema_fast=float(signal_df.loc[as_of, "ema_fast"]),
        ema_slow=float(signal_df.loc[as_of, "ema_slow"]),
        bull_entry_today=bool(signal_df.loc[as_of, "bull_entry"] > 0),
        bear_cross_today=bool(signal_df.loc[as_of, "bear_cross"] > 0),
        entry_filter_ok_today=bool(signal_df.loc[as_of, "entry_filter_pos_return_ok"] > 0),
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
