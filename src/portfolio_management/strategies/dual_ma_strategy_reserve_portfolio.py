from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from .dual_ma_strategy_core import DualMAParams, dual_ma
from .dual_ma_strategy_telegram import build_dual_ma_strategy_reserve_portfolio_message


@dataclass(frozen=True)
class ReservePortfolioDualMAConfig:
    close_hour: int
    start_date: datetime | None
    btc_symbol: str
    eth_symbol: str
    btc_params: DualMAParams
    eth_params: DualMAParams
    w_ref_btc: float
    w_ref_eth: float
    derisk_btc: float
    derisk_eth: float


def _dt_from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _as_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid float for {name}: {value!r}") from exc


def _as_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid int for {name}: {value!r}") from exc


def load_reserve_portfolio_dual_ma_config(raw: Mapping[str, Any]) -> ReservePortfolioDualMAConfig:
    close_hour = _as_int(raw.get("close_hour", 12), name="close_hour")
    start_date = _dt_from_iso(raw.get("start_date"))

    assets = raw.get("assets") or {}
    btc = assets.get("BTC") or assets.get("btc") or {}
    eth = assets.get("ETH") or assets.get("eth") or {}

    btc_symbol = str(btc.get("symbol", "BTC-USD"))
    eth_symbol = str(eth.get("symbol", "ETH-USD"))

    btc_p = btc.get("params") or {}
    eth_p = eth.get("params") or {}
    btc_params = DualMAParams(
        window_fast=_as_int(btc_p.get("window_fast", 100), name="btc.window_fast"),
        window_slow=_as_int(btc_p.get("window_slow", 365), name="btc.window_slow"),
        atr_win=_as_int(btc_p.get("atr_win", btc_p.get("window_fast", 100)), name="btc.atr_win"),
        atr_buf=_as_float(btc_p.get("atr_buf", 0.5), name="btc.atr_buf"),
    )
    eth_params = DualMAParams(
        window_fast=_as_int(eth_p.get("window_fast", 50), name="eth.window_fast"),
        window_slow=_as_int(eth_p.get("window_slow", 140), name="eth.window_slow"),
        atr_win=_as_int(eth_p.get("atr_win", eth_p.get("window_fast", 50)), name="eth.atr_win"),
        atr_buf=_as_float(eth_p.get("atr_buf", 0.2), name="eth.atr_buf"),
    )

    portfolio = raw.get("portfolio") or {}
    w_ref = portfolio.get("w_ref") or {}
    derisk = portfolio.get("derisk") or {}
    w_ref_btc = _as_float(w_ref.get("BTC", w_ref.get("btc", 0.33)), name="portfolio.w_ref.BTC")
    w_ref_eth = _as_float(w_ref.get("ETH", w_ref.get("eth", 0.67)), name="portfolio.w_ref.ETH")
    derisk_btc = _as_float(derisk.get("BTC", derisk.get("btc", 0.5)), name="portfolio.derisk.BTC")
    derisk_eth = _as_float(derisk.get("ETH", derisk.get("eth", 0.5)), name="portfolio.derisk.ETH")

    return ReservePortfolioDualMAConfig(
        close_hour=close_hour,
        start_date=start_date,
        btc_symbol=btc_symbol,
        eth_symbol=eth_symbol,
        btc_params=btc_params,
        eth_params=eth_params,
        w_ref_btc=w_ref_btc,
        w_ref_eth=w_ref_eth,
        derisk_btc=derisk_btc,
        derisk_eth=derisk_eth,
    )


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def compute_reserve_portfolio_targets(
    *,
    sig_btc: pd.Series,
    sig_eth: pd.Series,
    w_ref_btc: float,
    w_ref_eth: float,
    derisk_btc: float,
    derisk_eth: float,
) -> pd.DataFrame:
    sig_btc_c = _clip01(sig_btc.astype(float))
    sig_eth_c = _clip01(sig_eth.astype(float))
    w_btc = float(w_ref_btc) * ((1.0 - float(derisk_btc)) + float(derisk_btc) * sig_btc_c)
    w_eth = float(w_ref_eth) * ((1.0 - float(derisk_eth)) + float(derisk_eth) * sig_eth_c)
    out = pd.DataFrame({"BTC": w_btc, "ETH": w_eth}).sort_index()

    # Ensure BTC+ETH <= 1.0 (cash is implicit).
    s = out.sum(axis=1)
    scale = pd.Series(1.0, index=out.index)
    scale.loc[s > 1.0] = 1.0 / s.loc[s > 1.0]
    return out.mul(scale, axis=0)


def build_reserve_portfolio_compact_row(
    *,
    btc: pd.Series,
    eth: pd.Series,
    btc_signal: float,
    eth_signal: float,
    btc_rerisk: bool,
    eth_rerisk: bool,
    targets: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.Series, bool]:
    row = pd.Series(dtype="object")

    row["BTC_close"] = float(btc.loc[as_of, "close"])
    row["BTC_atr"] = float(btc.loc[as_of, "atr"])
    row["BTC_ma_fast"] = float(btc.loc[as_of, "ma_fast"])
    row["BTC_ma_slow"] = float(btc.loc[as_of, "ma_slow"])
    row["BTC_ma_fast_lo"] = float(btc.loc[as_of, "ma_fast_lo"])
    row["BTC_ma_fast_hi"] = float(btc.loc[as_of, "ma_fast_hi"])
    row["BTC_ma_slow_lo"] = float(btc.loc[as_of, "ma_slow_lo"])
    row["BTC_ma_slow_hi"] = float(btc.loc[as_of, "ma_slow_hi"])

    row["ETH_close"] = float(eth.loc[as_of, "close"])
    row["ETH_atr"] = float(eth.loc[as_of, "atr"])
    row["ETH_ma_fast"] = float(eth.loc[as_of, "ma_fast"])
    row["ETH_ma_slow"] = float(eth.loc[as_of, "ma_slow"])
    row["ETH_ma_fast_lo"] = float(eth.loc[as_of, "ma_fast_lo"])
    row["ETH_ma_fast_hi"] = float(eth.loc[as_of, "ma_fast_hi"])
    row["ETH_ma_slow_lo"] = float(eth.loc[as_of, "ma_slow_lo"])
    row["ETH_ma_slow_hi"] = float(eth.loc[as_of, "ma_slow_hi"])

    row["BTC_target"] = float(targets.loc[as_of, "BTC"])
    row["ETH_target"] = float(targets.loc[as_of, "ETH"])
    row["CASH_target"] = float(1.0 - targets.loc[as_of, ["BTC", "ETH"]].sum())

    idx = targets.index
    pos = idx.get_loc(as_of)
    trigger_today = True
    if isinstance(pos, int) and pos >= 1:
        prev = idx[pos - 1]
        tol = 1e-10
        trigger_today = bool((targets.loc[as_of] - targets.loc[prev]).abs().max() > tol)

    _ = (btc_signal, eth_signal, btc_rerisk, eth_rerisk)
    return row, trigger_today


def generate_reserve_portfolio_dual_ma_telegram_message(
    *,
    ohlc_btc: pd.DataFrame,
    ohlc_eth: pd.DataFrame,
    config: ReservePortfolioDualMAConfig,
) -> str:
    btc_res = dual_ma(ohlc_btc, config.btc_params, start_date=config.start_date)
    eth_res = dual_ma(ohlc_eth, config.eth_params, start_date=config.start_date)

    btc_sig = btc_res["signal"].dropna().astype(float)
    eth_sig = eth_res["signal"].dropna().astype(float)
    btc_rerisk = btc_res["zone_lock"].fillna(False).astype(bool)
    eth_rerisk = eth_res["zone_lock"].fillna(False).astype(bool)

    idx = btc_sig.index.intersection(eth_sig.index).sort_values()
    if idx.empty:
        raise ValueError("No overlapping valid signal dates between BTC and ETH.")
    as_of = pd.Timestamp(idx[-1])

    targets = compute_reserve_portfolio_targets(
        sig_btc=btc_sig.loc[idx],
        sig_eth=eth_sig.loc[idx],
        w_ref_btc=config.w_ref_btc,
        w_ref_eth=config.w_ref_eth,
        derisk_btc=config.derisk_btc,
        derisk_eth=config.derisk_eth,
    )

    mat_row, trigger_today = build_reserve_portfolio_compact_row(
        btc=btc_res,
        eth=eth_res,
        btc_signal=float(btc_sig.loc[as_of]),
        eth_signal=float(eth_sig.loc[as_of]),
        btc_rerisk=bool(btc_rerisk.loc[as_of]),
        eth_rerisk=bool(eth_rerisk.loc[as_of]),
        targets=targets,
        as_of=as_of,
    )

    return build_dual_ma_strategy_reserve_portfolio_message(
        as_of=as_of,
        row=mat_row,
        btc_sig=float(btc_sig.loc[as_of]),
        eth_sig=float(eth_sig.loc[as_of]),
        btc_rerisk=bool(btc_rerisk.loc[as_of]),
        eth_rerisk=bool(eth_rerisk.loc[as_of]),
        trigger_today=trigger_today,
        btc_fast_days=config.btc_params.window_fast,
        btc_slow_days=config.btc_params.window_slow,
        btc_atr_days=config.btc_params.atr_win,
        eth_fast_days=config.eth_params.window_fast,
        eth_slow_days=config.eth_params.window_slow,
        eth_atr_days=config.eth_params.atr_win,
    )


__all__ = [
    "ReservePortfolioDualMAConfig",
    "load_reserve_portfolio_dual_ma_config",
    "generate_reserve_portfolio_dual_ma_telegram_message",
]
