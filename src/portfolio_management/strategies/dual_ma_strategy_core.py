from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DualMAParams:
    window_fast: int
    window_slow: int
    atr_win: int
    atr_buf: float = 0.0
    atr_buf_on: float | None = None
    atr_buf_off: float | None = None


def _apply_start_date(df: pd.DataFrame, start_date: Optional[datetime | str]) -> pd.DataFrame:
    if start_date is None:
        return df
    sdate = pd.to_datetime(start_date)
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        if sdate.tzinfo is None:
            sdate = sdate.tz_localize(idx.tz)
        else:
            sdate = sdate.tz_convert(idx.tz)
    else:
        if sdate.tzinfo is not None:
            sdate = sdate.tz_convert(None)
    return df[df.index >= sdate]


def _wilder_atr(ohlc: pd.DataFrame, *, win: int) -> pd.Series:
    high = ohlc["high"].astype(float)
    low = ohlc["low"].astype(float)
    close = ohlc["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(win), adjust=False, min_periods=int(win)).mean()


def moving_average_signal_buffered_hysteresis(
    df_close: pd.DataFrame,
    *,
    window: int,
    df_ohlc: pd.DataFrame | None = None,
    atr_win: int = 20,
    atr_buf: float | None = 0.0,
    atr_buf_on: float | None = None,
    atr_buf_off: float | None = None,
) -> pd.DataFrame:
    ma = df_close.rolling(window=int(window), min_periods=int(window)).mean()

    base_buf = 0.0 if atr_buf is None else float(atr_buf)
    if (atr_buf_on is None) and (atr_buf_off is None):
        atr_up_val = base_buf
        atr_dn_val = base_buf
    else:
        atr_up_val = float(atr_buf_on) if atr_buf_on is not None else base_buf
        atr_dn_val = float(atr_buf_off) if atr_buf_off is not None else base_buf

    if (atr_up_val > 0.0) or (atr_dn_val > 0.0):
        if df_ohlc is None:
            raise ValueError("df_ohlc with ['high','low','close'] is required when ATR buffers are used.")
        atr = _wilder_atr(df_ohlc, win=atr_win)
        atr_b = pd.concat([atr] * df_close.shape[1], axis=1)
        atr_b.columns = df_close.columns
        thr_up = ma + atr_up_val * atr_b
        thr_dn = ma - atr_dn_val * atr_b
    else:
        thr_up = ma.copy()
        thr_dn = ma.copy()

    out = pd.DataFrame(pd.NA, index=df_close.index, columns=df_close.columns, dtype="Int64")
    for col in df_close.columns:
        close = df_close[col]
        up = thr_up[col]
        dn = thr_dn[col]

        state: int | None = None
        for t in df_close.index:
            c, u, d = close.loc[t], up.loc[t], dn.loc[t]
            if pd.isna(c) or pd.isna(u) or pd.isna(d):
                continue

            if state is None:
                state = 1 if c >= u else 0
            else:
                if state == 0 and c >= u:
                    state = 1
                elif state == 1 and c <= d:
                    state = 0

            out.at[t, col] = state

    return out


def dual_ma(
    ohlc: pd.DataFrame,
    params: DualMAParams,
    *,
    start_date: Optional[datetime | str] = None,
) -> pd.DataFrame:
    """
    Compute buffered dual-MA zone-lock exposure for a single asset.

    Returns a frame with:
      - close, atr
      - ma_fast/slow and bands (lo/hi)
      - signal (0 / 0.5 / 1)
      - zone_lock (boolean): re-risking entry regime
    """
    ohlc = _apply_start_date(ohlc, start_date).copy()
    required_cols = {"high", "low", "close"}
    missing = required_cols - set(ohlc.columns)
    if missing:
        raise ValueError(f"ohlc missing required columns: {missing}")

    close = ohlc["close"].astype(float)
    df_close = close.to_frame(name="close")

    atr = _wilder_atr(ohlc, win=int(params.atr_win))
    ma_fast = close.rolling(int(params.window_fast), min_periods=int(params.window_fast)).mean()
    ma_slow = close.rolling(int(params.window_slow), min_periods=int(params.window_slow)).mean()

    base_atr_buf = float(params.atr_buf)
    atr_on = float(params.atr_buf_on) if params.atr_buf_on is not None else base_atr_buf
    atr_off = float(params.atr_buf_off) if params.atr_buf_off is not None else base_atr_buf

    ma_fast_hi = ma_fast + atr_on * atr
    ma_fast_lo = ma_fast - atr_off * atr
    ma_slow_hi = ma_slow + atr_on * atr
    ma_slow_lo = ma_slow - atr_off * atr

    sig_fast_buf_df = moving_average_signal_buffered_hysteresis(
        df_close,
        window=int(params.window_fast),
        df_ohlc=ohlc,
        atr_win=int(params.atr_win),
        atr_buf=float(params.atr_buf),
        atr_buf_on=atr_on,
        atr_buf_off=atr_off,
    )
    sig_slow_buf_df = moving_average_signal_buffered_hysteresis(
        df_close,
        window=int(params.window_slow),
        df_ohlc=ohlc,
        atr_win=int(params.atr_win),
        atr_buf=float(params.atr_buf),
        atr_buf_on=atr_on,
        atr_buf_off=atr_off,
    )
    sig_fast_buf = sig_fast_buf_df["close"]
    sig_slow_buf = sig_slow_buf_df["close"]

    valid_idx = sig_fast_buf.dropna().index.intersection(sig_slow_buf.dropna().index).sort_values()

    exposure = pd.Series(np.nan, index=ohlc.index, dtype=float)
    zone_lock = pd.Series(pd.NA, index=ohlc.index, dtype="boolean")

    prev_exp = 0.0
    in_zone = False
    for t in valid_idx:
        a = int(sig_fast_buf.loc[t])
        b = int(sig_slow_buf.loc[t])

        both_off = (a == 0) and (b == 0)
        both_on = (a == 1) and (b == 1)
        any_on = (a == 1) or (b == 1)

        if in_zone:
            if both_off:
                exposure.loc[t] = 0.0
                in_zone = False
            elif both_on:
                exposure.loc[t] = 1.0
                in_zone = False
            else:
                exposure.loc[t] = 1.0
            zone_lock.loc[t] = in_zone
        else:
            if prev_exp != 0.0:
                exposure.loc[t] = 0.5 * (a + b)
                zone_lock.loc[t] = False
            else:
                if any_on:
                    exposure.loc[t] = 1.0
                    in_zone = True
                    zone_lock.loc[t] = True
                else:
                    exposure.loc[t] = 0.0
                    zone_lock.loc[t] = False

        prev_exp = float(exposure.loc[t])

    out = pd.DataFrame(index=ohlc.index)
    out["close"] = close
    out["atr"] = atr
    out["ma_fast"] = ma_fast
    out["ma_slow"] = ma_slow
    out["ma_fast_lo"] = ma_fast_lo
    out["ma_fast_hi"] = ma_fast_hi
    out["ma_slow_lo"] = ma_slow_lo
    out["ma_slow_hi"] = ma_slow_hi
    out["signal"] = exposure
    out["zone_lock"] = zone_lock
    return out


__all__ = ["DualMAParams", "dual_ma"]

