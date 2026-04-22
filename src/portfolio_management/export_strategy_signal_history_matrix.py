from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio_management.export_strategy_signals import (
    DEFAULT_MAPPING_PATH,
    DEFAULT_OUTPUT_ROOT,
    HYPE_FULL_HISTORY_START_DATE,
    SOL_FULL_HISTORY_START_DATE,
)
from portfolio_management.helpers.job_config import load_job_config
from portfolio_management.market_data import load_daily_close, load_daily_ohlc, resolve_db_path
from portfolio_management.strategies.dual_ma_strategy_core import dual_ma
from portfolio_management.strategies.dual_ma_strategy_reserve_portfolio import (
    generate_reserve_portfolio_snapshot,
    load_reserve_portfolio_dual_ma_config,
)
from portfolio_management.strategies.hype_eth_rotation_strategy import (
    _compute_rsi as compute_hype_rsi,
    generate_hype_eth_rotation_snapshot,
    load_hype_eth_rotation_config,
)
from portfolio_management.strategies.sol_eth_rotation_strategy import (
    _compute_rsi as compute_sol_rsi,
    generate_sol_eth_rotation_snapshot,
    load_sol_eth_rotation_config,
)
from portfolio_management.strategy_signal_mapping import load_strategy_signal_mapping


DEFAULT_START_DATE = "2025-12-31"


def _timestamp_slug(timestamp: str) -> str:
    return (
        timestamp.replace(":", "")
        .replace("-", "")
        .replace("+0000", "Z")
        .replace("+00:00", "Z")
        .replace("T", "_")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a daily signal-value matrix for active strategies to the shared formal extract folder."
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=("local", "vm"),
        help="Job profile to use (local or vm).",
    )
    parser.add_argument(
        "--mapping-path",
        default=str(DEFAULT_MAPPING_PATH),
        help="Path to strategy signal mapping YAML.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where formal CSV extracts should be written.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Earliest date to include, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Optional database URL override (e.g. postgresql://...).",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional SQLite path override (relative to repo root or absolute).",
    )
    return parser


def _to_frame(series: pd.Series, column_name: str) -> pd.DataFrame:
    frame = series.astype(float).rename(column_name).to_frame()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.loc[frame.index.notna()].sort_index()
    return frame


def _build_reserve_signal_history(*, db_url: str | None, db_path: Path | None) -> pd.DataFrame:
    reserve_conf = load_reserve_portfolio_dual_ma_config(
        load_job_config("dual_ma_strategy", use_profile=False)
    )
    btc_ohlc = load_daily_ohlc(
        reserve_conf.btc_symbol,
        close_hour=reserve_conf.close_hour,
        start_date=reserve_conf.start_date,
        db_url=db_url,
        db_path=db_path,
    )
    eth_ohlc = load_daily_ohlc(
        reserve_conf.eth_symbol,
        close_hour=reserve_conf.close_hour,
        start_date=reserve_conf.start_date,
        db_url=db_url,
        db_path=db_path,
    )
    btc_signal = 1.0 - dual_ma(btc_ohlc, reserve_conf.btc_params, start_date=reserve_conf.start_date)["signal"].dropna()
    eth_signal = 1.0 - dual_ma(eth_ohlc, reserve_conf.eth_params, start_date=reserve_conf.start_date)["signal"].dropna()
    return pd.concat(
        [
            _to_frame(btc_signal, "ACTIVE_BTC_MA"),
            _to_frame(eth_signal, "ACTIVE_ETH_MA"),
        ],
        axis=1,
        join="outer",
        sort=True,
    )


def _build_sol_signal_history(*, db_url: str | None, db_path: Path | None) -> pd.DataFrame:
    sol_conf = load_sol_eth_rotation_config(load_job_config("sol_eth_rotation_strategy"))
    sol = load_daily_close(
        sol_conf.sol_symbol,
        close_hour=sol_conf.close_hour,
        start_date=SOL_FULL_HISTORY_START_DATE,
        db_url=db_url,
        db_path=db_path,
    )
    eth = load_daily_close(
        sol_conf.eth_symbol,
        close_hour=sol_conf.close_hour,
        start_date=SOL_FULL_HISTORY_START_DATE,
        db_url=db_url,
        db_path=db_path,
    )
    idx = sol.index.intersection(eth.index).sort_values()
    ratio = (sol.loc[idx] / eth.loc[idx]).astype(float)
    price_ratio = ratio.copy()

    if sol_conf.entry_signal_return_window == 0:
        ratio_ret_entry_window = pd.Series(0.0, index=ratio.index, dtype=float)
    else:
        ratio_ret_entry_window = ratio.pct_change(sol_conf.entry_signal_return_window)
    if sol_conf.exit_signal_return_window == 0:
        ratio_ret_exit_window = pd.Series(0.0, index=ratio.index, dtype=float)
    else:
        ratio_ret_exit_window = ratio.pct_change(sol_conf.exit_signal_return_window)

    ema_fast = ratio.ewm(span=sol_conf.fast_span, adjust=False, min_periods=sol_conf.fast_span).mean()
    ema_slow = ratio.ewm(span=sol_conf.slow_span, adjust=False, min_periods=sol_conf.slow_span).mean()

    base_target_signal = pd.Series(0.0, index=ratio.index, dtype=float)
    current = 1.0
    for ts in ratio.index:
        spread = (
            float(ema_fast.loc[ts] - ema_slow.loc[ts])
            if pd.notna(ema_fast.loc[ts]) and pd.notna(ema_slow.loc[ts])
            else float("nan")
        )
        rw_entry = ratio_ret_entry_window.loc[ts]
        rw_exit = ratio_ret_exit_window.loc[ts]
        if pd.isna(spread):
            base_target_signal.loc[ts] = current
            continue
        can_enter = sol_conf.entry_signal_return_window == 0 or (pd.notna(rw_entry) and float(rw_entry) > 0.0)
        can_exit = sol_conf.exit_signal_return_window == 0 or (pd.notna(rw_exit) and float(rw_exit) < 0.0)
        if current <= 0.0 and spread > 0.0 and can_enter:
            current = 1.0
        elif current > 0.0 and spread < 0.0 and can_exit:
            current = 0.0
        base_target_signal.loc[ts] = current

    rsi = compute_sol_rsi(ratio, sol_conf.rsi_period)
    early_exit_target = pd.Series(False, index=ratio.index, dtype=bool)
    rsi_armed = False
    prev_base_target = base_target_signal.shift(1)
    for ts in ratio.index:
        prev_target = prev_base_target.loc[ts]
        rsi_now = rsi.loc[ts]
        if pd.isna(prev_target) or float(prev_target) <= 0.0:
            rsi_armed = False
        if pd.notna(rsi_now) and pd.notna(prev_target) and float(prev_target) > 0.0:
            if float(rsi_now) >= sol_conf.rsi_exit_level:
                rsi_armed = True
            if rsi_armed and float(rsi_now) <= sol_conf.rsi_exit_level:
                early_exit_target.loc[ts] = True
                rsi_armed = False

    signal_flip_event = base_target_signal != base_target_signal.shift(1)
    if sol_conf.use_rsi_early_exit:
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

    return _to_frame(alloc_signal, "ACTIVE_SOL_ETH")


def _build_hype_signal_history(*, db_url: str | None, db_path: Path | None) -> pd.DataFrame:
    hype_conf = load_hype_eth_rotation_config(load_job_config("hype_eth_rotation_strategy"))
    hype = load_daily_close(
        hype_conf.hype_symbol,
        close_hour=hype_conf.close_hour,
        start_date=HYPE_FULL_HISTORY_START_DATE,
        db_url=db_url,
        db_path=db_path,
    )
    eth = load_daily_close(
        hype_conf.eth_symbol,
        close_hour=hype_conf.close_hour,
        start_date=HYPE_FULL_HISTORY_START_DATE,
        db_url=db_url,
        db_path=db_path,
    )
    idx = hype.index.intersection(eth.index).sort_values()
    ratio = (hype.loc[idx] / eth.loc[idx]).astype(float)
    price_ratio = ratio.copy()

    if hype_conf.entry_signal_return_window == 0:
        ratio_ret_entry_window = pd.Series(0.0, index=ratio.index, dtype=float)
    else:
        ratio_ret_entry_window = ratio.pct_change(hype_conf.entry_signal_return_window)
    if hype_conf.exit_signal_return_window == 0:
        ratio_ret_exit_window = pd.Series(0.0, index=ratio.index, dtype=float)
    else:
        ratio_ret_exit_window = ratio.pct_change(hype_conf.exit_signal_return_window)

    ema_fast = ratio.ewm(span=hype_conf.fast_span, adjust=False, min_periods=hype_conf.fast_span).mean()
    ema_slow = ratio.ewm(span=hype_conf.slow_span, adjust=False, min_periods=hype_conf.slow_span).mean()

    base_target_signal = pd.Series(0.0, index=ratio.index, dtype=float)
    current = 1.0
    for ts in ratio.index:
        spread = (
            float(ema_fast.loc[ts] - ema_slow.loc[ts])
            if pd.notna(ema_fast.loc[ts]) and pd.notna(ema_slow.loc[ts])
            else float("nan")
        )
        rw_entry = ratio_ret_entry_window.loc[ts]
        rw_exit = ratio_ret_exit_window.loc[ts]
        if pd.isna(spread):
            base_target_signal.loc[ts] = current
            continue
        can_enter_target = (
            hype_conf.entry_signal_return_window == 0
            or (pd.notna(rw_entry) and float(rw_entry) > 0.0)
        )
        can_exit_target = (
            hype_conf.exit_signal_return_window == 0
            or (pd.notna(rw_exit) and float(rw_exit) < 0.0)
        )
        if current <= 0.0 and spread > 0.0 and can_enter_target:
            current = 1.0
        elif current > 0.0 and spread < 0.0 and can_exit_target:
            current = 0.0
        base_target_signal.loc[ts] = current

    rsi = compute_hype_rsi(ratio, hype_conf.rsi_period)
    early_exit_target = pd.Series(False, index=ratio.index, dtype=bool)
    rsi_armed = False
    prev_base_target = base_target_signal.shift(1)
    for ts in ratio.index:
        prev_target = prev_base_target.loc[ts]
        rsi_now = rsi.loc[ts]
        if pd.isna(prev_target) or float(prev_target) <= 0.0:
            rsi_armed = False
        if pd.notna(rsi_now) and pd.notna(prev_target) and float(prev_target) > 0.0:
            if float(rsi_now) >= hype_conf.rsi_exit_level:
                rsi_armed = True
            if rsi_armed and float(rsi_now) <= hype_conf.rsi_exit_level:
                early_exit_target.loc[ts] = True
                rsi_armed = False

    signal_flip_event = base_target_signal != base_target_signal.shift(1)
    if hype_conf.use_rsi_early_exit:
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

    return _to_frame(alloc_signal, "ACTIVE_HYPE_ETH")


def _build_matrix(*, start_date: str, db_url: str | None, db_path: Path | None) -> pd.DataFrame:
    matrix = pd.concat(
        [
            _build_reserve_signal_history(db_url=db_url, db_path=db_path),
            _build_sol_signal_history(db_url=db_url, db_path=db_path),
            _build_hype_signal_history(db_url=db_url, db_path=db_path),
        ],
        axis=1,
        join="outer",
    ).sort_index()
    start = pd.Timestamp(start_date, tz="UTC")
    matrix = matrix.loc[matrix.index >= start].copy()
    matrix.index.name = "date"
    matrix.reset_index(inplace=True)
    matrix["date"] = pd.to_datetime(matrix["date"], utc=True).dt.strftime("%Y-%m-%d")
    return matrix


def _coerce_value(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    try:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return str(value)


def _build_reserve_context(*, db_url, db_path):
    reserve_conf = load_reserve_portfolio_dual_ma_config(
        load_job_config("dual_ma_strategy", use_profile=False)
    )
    btc_ohlc = load_daily_ohlc(reserve_conf.btc_symbol, close_hour=reserve_conf.close_hour,
        start_date=reserve_conf.start_date, db_url=db_url, db_path=db_path)
    eth_ohlc = load_daily_ohlc(reserve_conf.eth_symbol, close_hour=reserve_conf.close_hour,
        start_date=reserve_conf.start_date, db_url=db_url, db_path=db_path)
    snap = generate_reserve_portfolio_snapshot(ohlc_btc=btc_ohlc, ohlc_eth=eth_ohlc, config=reserve_conf)
    row = {k: _coerce_value(v) for k, v in snap.compact_row.items()}
    btc = {
        "signal": float(snap.btc_signal),
        "rerisk": bool(snap.btc_rerisk),
        "fast_days": int(snap.btc_fast_days),
        "slow_days": int(snap.btc_slow_days),
        "atr_days": int(snap.btc_atr_days),
        "close": row.get("BTC_close"),
        "atr": row.get("BTC_atr"),
        "ma_fast": row.get("BTC_ma_fast"),
        "ma_fast_band": [row.get("BTC_ma_fast_lo"), row.get("BTC_ma_fast_hi")],
        "ma_slow": row.get("BTC_ma_slow"),
        "ma_slow_band": [row.get("BTC_ma_slow_lo"), row.get("BTC_ma_slow_hi")],
        "target_weight": row.get("BTC_target"),
    }
    eth = {
        "signal": float(snap.eth_signal),
        "rerisk": bool(snap.eth_rerisk),
        "fast_days": int(snap.eth_fast_days),
        "slow_days": int(snap.eth_slow_days),
        "atr_days": int(snap.eth_atr_days),
        "close": row.get("ETH_close"),
        "atr": row.get("ETH_atr"),
        "ma_fast": row.get("ETH_ma_fast"),
        "ma_fast_band": [row.get("ETH_ma_fast_lo"), row.get("ETH_ma_fast_hi")],
        "ma_slow": row.get("ETH_ma_slow"),
        "ma_slow_band": [row.get("ETH_ma_slow_lo"), row.get("ETH_ma_slow_hi")],
        "target_weight": row.get("ETH_target"),
    }
    return {"ACTIVE_BTC_MA": btc, "ACTIVE_ETH_MA": eth}


def _build_rotation_context_sol(*, db_url, db_path):
    conf = load_sol_eth_rotation_config(load_job_config("sol_eth_rotation_strategy"))
    sol = load_daily_close(conf.sol_symbol, close_hour=conf.close_hour,
        start_date=SOL_FULL_HISTORY_START_DATE, db_url=db_url, db_path=db_path)
    eth = load_daily_close(conf.eth_symbol, close_hour=conf.close_hour,
        start_date=SOL_FULL_HISTORY_START_DATE, db_url=db_url, db_path=db_path)
    snap = generate_sol_eth_rotation_snapshot(sol_close=sol, eth_close=eth, config=conf)
    return {
        "signal": float(snap.alloc_signal),
        "in_position": bool(snap.in_position),
        "trigger_today": bool(snap.trigger_today),
        "sol_close": _coerce_value(snap.sol_close),
        "eth_close": _coerce_value(snap.eth_close),
        "ratio": _coerce_value(snap.ratio_close),
        "ema_fast": _coerce_value(snap.price_ratio_ema_fast),
        "ema_slow": _coerce_value(snap.price_ratio_ema_slow),
        "ema_fast_prev": _coerce_value(snap.price_ratio_ema_fast_prev),
        "ema_slow_prev": _coerce_value(snap.price_ratio_ema_slow_prev),
        "rsi": _coerce_value(snap.rsi),
        "rsi_prev": _coerce_value(snap.rsi_prev),
        "fast_span": int(conf.fast_span),
        "slow_span": int(conf.slow_span),
        "rsi_period": int(conf.rsi_period),
        "rsi_exit_level": float(conf.rsi_exit_level),
        "entry_filter_ok_today": bool(snap.entry_filter_ok_today),
        "early_exit_today": bool(snap.early_exit_today),
    }


def _build_rotation_context_hype(*, db_url, db_path):
    conf = load_hype_eth_rotation_config(load_job_config("hype_eth_rotation_strategy"))
    hype = load_daily_close(conf.hype_symbol, close_hour=conf.close_hour,
        start_date=HYPE_FULL_HISTORY_START_DATE, db_url=db_url, db_path=db_path)
    eth = load_daily_close(conf.eth_symbol, close_hour=conf.close_hour,
        start_date=HYPE_FULL_HISTORY_START_DATE, db_url=db_url, db_path=db_path)
    snap = generate_hype_eth_rotation_snapshot(hype_close=hype, eth_close=eth, config=conf)
    return {
        "signal": float(snap.alloc_signal),
        "in_position": bool(snap.in_position),
        "trigger_today": bool(snap.trigger_today),
        "hype_close": _coerce_value(snap.hype_close),
        "eth_close": _coerce_value(snap.eth_close),
        "ratio": _coerce_value(snap.ratio_close),
        "ema_fast": _coerce_value(snap.price_ratio_ema_fast),
        "ema_slow": _coerce_value(snap.price_ratio_ema_slow),
        "ema_fast_prev": _coerce_value(snap.price_ratio_ema_fast_prev),
        "ema_slow_prev": _coerce_value(snap.price_ratio_ema_slow_prev),
        "rsi": _coerce_value(snap.rsi),
        "rsi_prev": _coerce_value(snap.rsi_prev),
        "fast_span": int(conf.fast_span),
        "slow_span": int(conf.slow_span),
        "rsi_period": int(conf.rsi_period),
        "rsi_exit_level": float(conf.rsi_exit_level),
        "entry_filter_ok_today": bool(snap.entry_filter_ok_today),
        "early_exit_today": bool(snap.early_exit_today),
    }


def _build_context(*, matrix, db_url, db_path):
    context = {}
    context.update(_build_reserve_context(db_url=db_url, db_path=db_path))
    context["ACTIVE_SOL_ETH"] = _build_rotation_context_sol(db_url=db_url, db_path=db_path)
    context["ACTIVE_HYPE_ETH"] = _build_rotation_context_hype(db_url=db_url, db_path=db_path)
    if len(matrix) >= 2:
        prev = matrix.iloc[-2]
        for strategy_key, details in context.items():
            if strategy_key in matrix.columns:
                details["signal_previous"] = float(prev[strategy_key])
                details["previous_date"] = str(prev["date"])
    return context


def main() -> None:
    args = build_parser().parse_args()
    os.environ["JOB_PROFILE"] = args.profile

    mapping = load_strategy_signal_mapping(args.mapping_path)
    db_url = args.db_url
    db_path = resolve_db_path(args.db_path) if args.db_path is not None else None
    matrix = _build_matrix(start_date=args.start_date, db_url=db_url, db_path=db_path)

    expected_columns = ["date", *mapping.keys()]
    missing = [column for column in expected_columns if column not in matrix.columns]
    if missing:
        raise SystemExit(f"Signal matrix missing expected columns from mapping: {missing}")
    matrix = matrix[expected_columns]

    extracted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    as_of_label = str(matrix["date"].iloc[-1])
    output_path = output_root / f"{as_of_label}_strategy_signal_history_matrix_{_timestamp_slug(extracted_at)}.csv"
    matrix.to_csv(output_path, index=False)

    context = _build_context(matrix=matrix, db_url=db_url, db_path=db_path)
    context_payload = {
        "as_of": as_of_label,
        "generated_at": extracted_at,
        "strategies": context,
    }
    context_path = output_root / f"{as_of_label}_strategy_signal_context_{_timestamp_slug(extracted_at)}.json"
    with open(context_path, "w") as cf:
        json.dump(context_payload, cf, indent=2, default=str)

    print(f"Exported {len(matrix)} rows")
    print(f"Start date: {matrix['date'].iloc[0]}")
    print(f"As of date: {as_of_label}")
    print(f"CSV path: {output_path}")
    print(f"Context JSON path: {context_path}")


if __name__ == "__main__":
    main()
