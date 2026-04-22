from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio_management.helpers.config import BASE_DIR
from portfolio_management.helpers.job_config import load_job_config
from portfolio_management.market_data import load_daily_close, load_daily_ohlc, resolve_db_path
from portfolio_management.strategy_signal_mapping import (
    DEFAULT_MAPPING_PATH,
    load_strategy_signal_mapping,
)
from portfolio_management.strategies.dual_ma_strategy_reserve_portfolio import (
    build_reserve_portfolio_signal_records,
    generate_reserve_portfolio_snapshot,
    load_reserve_portfolio_dual_ma_config,
)
from portfolio_management.strategies.hype_eth_rotation_strategy import (
    build_hype_eth_signal_record,
    generate_hype_eth_rotation_snapshot,
    load_hype_eth_rotation_config,
)
from portfolio_management.strategies.sol_eth_rotation_strategy import (
    build_sol_eth_signal_record,
    generate_sol_eth_rotation_snapshot,
    load_sol_eth_rotation_config,
)


DEFAULT_OUTPUT_ROOT = BASE_DIR.parent / "portfolio_analytics" / "outputs" / "formal_extract"
DEFAULT_SCHEMA_VERSION = "1"
SOL_FULL_HISTORY_START_DATE = datetime(2021, 12, 1, tzinfo=timezone.utc)
HYPE_FULL_HISTORY_START_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc)


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
        description="Export daily strategy signals to the shared formal extract folder."
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


def _records_to_dataframe(records: list[object], extracted_at: str) -> pd.DataFrame:
    rows = []
    for record in records:
        row = asdict(record)
        row["extracted_at_utc"] = extracted_at
        row["as_of_date"] = pd.Timestamp(row["as_of"]).date().isoformat()
        row["as_of_utc"] = pd.Timestamp(row["as_of"]).tz_convert("UTC").isoformat()
        row["source_system"] = "portfolio_management"
        row["schema_version"] = DEFAULT_SCHEMA_VERSION
        row.pop("as_of", None)
        rows.append(row)

    dataframe = pd.DataFrame(rows)
    keep = [
        "extracted_at_utc",
        "as_of_date",
        "as_of_utc",
        "strategy_id",
        "signal_strategy_slug",
        "effective_signal_value",
        "raw_signal_value",
        "target_weight",
        "trigger_today",
        "current_state",
        "source_system",
        "schema_version",
    ]
    dataframe = dataframe[keep].sort_values(by=["strategy_id"], kind="stable").reset_index(drop=True)
    return dataframe


def _validate_mapping(dataframe: pd.DataFrame, mapping: dict[str, str]) -> None:
    actual_strategy_ids = set(dataframe["strategy_id"].tolist())
    expected_strategy_ids = set(mapping.keys())
    if actual_strategy_ids != expected_strategy_ids:
        missing = sorted(expected_strategy_ids - actual_strategy_ids)
        extra = sorted(actual_strategy_ids - expected_strategy_ids)
        raise ValueError(
            "Generated strategy signal set does not match mapping config. "
            f"Missing={missing}, Extra={extra}"
        )

    for row in dataframe.itertuples(index=False):
        expected_slug = mapping[row.strategy_id]
        if row.signal_strategy_slug != expected_slug:
            raise ValueError(
                f"Strategy {row.strategy_id} emitted slug {row.signal_strategy_slug!r}, "
                f"expected {expected_slug!r} from mapping config."
            )


def main() -> None:
    args = build_parser().parse_args()

    import os

    os.environ["JOB_PROFILE"] = args.profile

    mapping = load_strategy_signal_mapping(args.mapping_path)
    db_url = args.db_url
    db_path = resolve_db_path(args.db_path) if args.db_path is not None else None

    reserve_conf = load_reserve_portfolio_dual_ma_config(
        load_job_config("dual_ma_strategy", use_profile=False)
    )
    reserve_snapshot = generate_reserve_portfolio_snapshot(
        ohlc_btc=load_daily_ohlc(
            reserve_conf.btc_symbol,
            close_hour=reserve_conf.close_hour,
            start_date=reserve_conf.start_date,
            db_url=db_url,
            db_path=db_path,
        ),
        ohlc_eth=load_daily_ohlc(
            reserve_conf.eth_symbol,
            close_hour=reserve_conf.close_hour,
            start_date=reserve_conf.start_date,
            db_url=db_url,
            db_path=db_path,
        ),
        config=reserve_conf,
    )
    reserve_records = list(build_reserve_portfolio_signal_records(reserve_snapshot))

    sol_raw = load_job_config("sol_eth_rotation_strategy")
    sol_conf = load_sol_eth_rotation_config(sol_raw)
    sol_snapshot = generate_sol_eth_rotation_snapshot(
        sol_close=load_daily_close(
            sol_conf.sol_symbol,
            close_hour=sol_conf.close_hour,
            start_date=SOL_FULL_HISTORY_START_DATE,
            db_url=db_url,
            db_path=db_path,
        ),
        eth_close=load_daily_close(
            sol_conf.eth_symbol,
            close_hour=sol_conf.close_hour,
            start_date=SOL_FULL_HISTORY_START_DATE,
            db_url=db_url,
            db_path=db_path,
        ),
        config=sol_conf,
    )

    hype_raw = load_job_config("hype_eth_rotation_strategy")
    hype_conf = load_hype_eth_rotation_config(hype_raw)
    hype_snapshot = generate_hype_eth_rotation_snapshot(
        hype_close=load_daily_close(
            hype_conf.hype_symbol,
            close_hour=hype_conf.close_hour,
            start_date=HYPE_FULL_HISTORY_START_DATE,
            db_url=db_url,
            db_path=db_path,
        ),
        eth_close=load_daily_close(
            hype_conf.eth_symbol,
            close_hour=hype_conf.close_hour,
            start_date=HYPE_FULL_HISTORY_START_DATE,
            db_url=db_url,
            db_path=db_path,
        ),
        config=hype_conf,
    )

    records = reserve_records + [
        build_sol_eth_signal_record(sol_snapshot),
        build_hype_eth_signal_record(hype_snapshot),
    ]

    extracted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    dataframe = _records_to_dataframe(records, extracted_at)
    _validate_mapping(dataframe, mapping)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    as_of_label = str(dataframe["as_of_date"].iloc[0])
    output_path = output_root / f"strategy_signals_{as_of_label}_{_timestamp_slug(extracted_at)}.csv"
    dataframe.to_csv(output_path, index=False)

    print(f"Exported {len(dataframe)} rows")
    print(f"As of date: {as_of_label}")
    print(f"CSV path: {output_path}")


if __name__ == "__main__":
    main()
