#!/usr/bin/env python3
"""Generate an end-labelled SMA dashboard and optionally email the summary."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import base64
import os
import traceback
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from price_data_infra.data import fetch_ohlcv
from portfolio_management.helpers.config import BASE_DIR
from portfolio_management.helpers.email import EmailClient
from portfolio_management.helpers.job_config import load_job_config

DEFAULT_ASSET_MA_PAIRS: Sequence[tuple[str, int]] = (
    ("ETH-USD", 50),
    ("ETH-USD", 140),
    ("SOL-USD", 50),
)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_prices(
    pairs: Sequence[tuple[str, int]],
    close_hour: int,
    start: datetime | None,
    *,
    db_url: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    symbols = sorted({symbol for symbol, _ in pairs})
    frames: list[pd.Series] = []
    for symbol in symbols:
        df = fetch_ohlcv(
            symbol,
            frequency="daily",
            close_hour=close_hour,
            db_url=db_url,
            db_path=db_path,
        )
        if start:
            df = df.loc[start:]
        series = df["close"].rename(symbol)
        frames.append(series)
    merged = pd.concat(frames, axis=1).dropna(how="all")
    if not isinstance(merged.index, pd.DatetimeIndex):
        merged.index = pd.to_datetime(merged.index, utc=True)
    else:
        merged.index = merged.index.tz_convert("UTC")
    return merged


def _signal_label(is_above: bool) -> str:
    return "Above MA" if is_above else "Below MA"


def _format_timestamp(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d")


def _compute_signal_metadata(
    prices: pd.Series,
    window: int,
) -> tuple[pd.Series, float, float, bool, dict[str, Any] | None, dict[str, Any] | None]:
    sma = prices.rolling(window).mean()
    latest_price = float(prices.iloc[-1])
    latest_sma = float(sma.iloc[-1])
    above_ma = latest_price > latest_sma
    change_info: dict[str, Any] | None = None
    latest_snapshot: dict[str, Any] | None = None

    aligned = pd.DataFrame({"price": prices, "sma": sma}).dropna()
    if not aligned.empty:
        latest_row = aligned.iloc[-1]
        latest_signal = bool(latest_row["price"] > latest_row["sma"])
        latest_snapshot = {
            "symbol": prices.name,
            "window": window,
            "latest_signal": latest_signal,
            "latest_date": aligned.index[-1],
            "latest_price": float(latest_row["price"]),
            "latest_sma": float(latest_row["sma"]),
        }
        if len(aligned) >= 2:
            previous_row = aligned.iloc[-2]
            previous_signal = bool(previous_row["price"] > previous_row["sma"])
            if latest_signal != previous_signal:
                change_info = {
                    "symbol": prices.name,
                    "window": window,
                    "latest_signal": latest_signal,
                    "previous_signal": previous_signal,
                    "latest_date": aligned.index[-1],
                    "previous_date": aligned.index[-2],
                }

    return sma, latest_price, latest_sma, above_ma, change_info, latest_snapshot


def _build_asset_block(
    prices: pd.Series,
    window: int,
    lookback_days: int,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    sma, latest_price, latest_sma, above_ma, change_info, latest_snapshot = _compute_signal_metadata(
        prices, window
    )

    chart_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    chart_prices = prices.loc[chart_start:]
    chart_sma = sma.loc[chart_start:]
    if chart_prices.empty:
        chart_prices = prices
        chart_sma = sma

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(chart_prices.index, chart_prices, label=f"{prices.name} Price", color="black", linewidth=1.5)
    ax.plot(chart_sma.index, chart_sma, label=f"SMA({window})", color="blue", linewidth=1.2, linestyle="--")
    ax.set_title(f"{prices.name} SMA({window}) Signal", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    chart_html = (
        f'<img src="data:image/png;base64,{img_base64}" '
        'style="max-width:800px; display:block; margin:auto;" />'
    )

    color = "green" if above_ma else "red"
    label = "Above MA" if above_ma else "Below MA"
    light_html = f"""
        <div style="text-align:center; margin-top:15px;">
            <div style="width:40px;height:40px;border-radius:50%;margin:auto;
                        background:{color};box-shadow:0 0 10px {color};"></div>
            <p style="margin-top:8px;font-weight:bold;color:{color};">{label}</p>
            <p>Price: {latest_price:.2f} | SMA({window}): {latest_sma:.2f}</p>
        </div>
    """

    recent = prices.tail(lookback_days + 1)
    recent_sma = sma.tail(lookback_days + 1)
    signals = (recent > recent_sma).astype(int)

    dates = recent.index.strftime("%b-%d").tolist()
    lights = [
        f'<div style="width:20px;height:20px;border-radius:50%;margin:auto;'
        f'background:{"green" if value == 1 else "red"};"></div>'
        for value in signals.values
    ]
    price_values = [f"{value:.2f}" for value in recent.values]
    sma_values = [f"{value:.2f}" for value in recent_sma.values]

    history_table = f"""
    <table style="border-collapse:collapse; text-align:center; margin:auto; margin-top:15px;">
        <thead style="background:#f2f2f2;">
            <tr><th>Date</th>{"".join(f"<th>{day}</th>" for day in dates)}</tr>
        </thead>
        <tbody>
            <tr><td><b>Signal</b></td>{"".join(f"<td>{light}</td>" for light in lights)}</tr>
            <tr><td><b>Price</b></td>{"".join(f"<td>{price}</td>" for price in price_values)}</tr>
            <tr><td><b>SMA</b></td>{"".join(f"<td>{sma_value}</td>" for sma_value in sma_values)}</tr>
        </tbody>
    </table>
    """

    block_html = f"""
    <div style="margin-bottom:50px;">
        <h2 style="text-align:center;">{prices.name} — SMA({window})</h2>
        {chart_html}
        {light_html}
        <div style="text-align:center;margin-top:10px;font-style:italic;">Past {lookback_days} days</div>
        {history_table}
    </div>
    """
    return block_html, change_info, latest_snapshot


def _build_signal_summary(changes: Sequence[dict[str, Any]], prices: pd.DataFrame) -> str | None:
    recent_index = prices.dropna(how="all").index
    latest_date = recent_index[-1] if len(recent_index) >= 1 else None
    previous_date = recent_index[-2] if len(recent_index) >= 2 else None

    if not changes:
        if latest_date and previous_date:
            message = (
                f"No signal changes detected between {_format_timestamp(previous_date)} "
                f"and {_format_timestamp(latest_date)}."
            )
        elif latest_date:
            message = (
                f"Only one closing date available ({_format_timestamp(latest_date)}); "
                "not enough data to compare signals."
            )
        else:
            message = "No price data available to evaluate signal changes."
        return "<br/>".join(["Signal Summary:", f"- {message}"])

    rows = ["Signal Summary:"]
    for change in changes:
        rows.append(
            f"{change['symbol']} SMA({change['window']}): "
            f"{_format_timestamp(change['previous_date'])} {_signal_label(change['previous_signal'])} → "
            f"{_format_timestamp(change['latest_date'])} {_signal_label(change['latest_signal'])}"
        )

    lines = [rows[0]]
    lines.extend(f"- {entry}" for entry in rows[1:])
    return "<br/>".join(lines)


def _build_levels_summary(levels: Sequence[dict[str, Any]]) -> str | None:
    if not levels:
        return "<br/>".join(["SMA Signals:", "- No SMA levels available."])

    sorted_levels = sorted(levels, key=lambda item: (item["symbol"], item["window"]))
    entries = ["SMA Signals:"]
    for item in sorted_levels:
        entries.append(
            f"{item['symbol']} SMA({item['window']}): Price {item['latest_price']:.2f}, "
            f"SMA {item['latest_sma']:.2f} ({_signal_label(item['latest_signal'])})"
        )

    lines = [entries[0]]
    lines.extend(f"- {entry}" for entry in entries[1:])
    return "<br/>".join(lines)


def _build_date_line(prices: pd.DataFrame) -> str | None:
    recent_index = prices.dropna(how="all").index
    latest_date = recent_index[-1] if len(recent_index) >= 1 else None
    if not latest_date:
        return None
    return f"Date: {_format_timestamp(latest_date)}"


def _build_dashboard_html(
    pairs: Sequence[tuple[str, int]],
    prices: pd.DataFrame,
    lookback_days: int,
    close_hour: int,
) -> str:
    blocks: list[str] = []
    changes: list[dict[str, Any]] = []
    latest_levels: list[dict[str, Any]] = []
    for symbol, window in pairs:
        series = prices[symbol].dropna()
        if series.empty:
            continue
        block_html, change_info, latest_snapshot = _build_asset_block(series, window, lookback_days)
        blocks.append(block_html)
        if change_info:
            changes.append(change_info)
        if latest_snapshot:
            latest_levels.append(latest_snapshot)

    date_line = _build_date_line(prices)
    summary_line = _build_signal_summary(changes, prices)
    levels_line = _build_levels_summary(latest_levels)
    summary_lines = [line for line in (date_line, summary_line, levels_line) if line]
    summary_html = "<br/><br/>".join(summary_lines) if summary_lines else ""
    summary_block = f"{summary_html}<br/><br/>" if summary_html else ""

    return f"""
    <html>
    <head>
        <title>SMA Signal Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: auto;
                background: #fafafa;
                color: #333;
            }}
            h1 {{
                text-align: center;
                margin-top: 30px;
            }}
            table, th, td {{
                border: 1px solid #ccc;
                padding: 6px 8px;
            }}
            th {{
                font-size: 12px;
            }}
            td {{
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <h1>SMA Signal (UTC:+{close_hour})</h1>
        {summary_block}
        {''.join(blocks)}
    </body>
    </html>
    """


def run_dashboard(
    asset_ma_pairs: Sequence[tuple[str, int]],
    output_dir: Path,
    lookback_days: int,
    close_hour: int,
    start_date: datetime | None,
    email_recipient: bool,
    email_client: EmailClient | None = None,
    db_url: str | None = None,
    db_path: Path | None = None,
) -> tuple[Path, str]:
    prices = _load_prices(
        asset_ma_pairs,
        close_hour,
        start_date,
        db_url=db_url,
        db_path=db_path,
    )
    if prices.empty:
        raise ValueError("No price data available after applying filters.")

    html = _build_dashboard_html(asset_ma_pairs, prices, lookback_days, close_hour)

    _ensure_directory(output_dir)
    outfile = output_dir / "sma_signal_dashboard.html"
    outfile.write_text(html)

    if email_recipient:
        client = email_client or EmailClient()
        if client.enabled():
            subject = f"SMA Dashboard ({datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC})"
            client.send_html(subject, html)

    return outfile, html


def parse_asset_pairs(raw_values: Iterable[str]) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for item in raw_values:
        symbol, window = item.split(":")
        pairs.append((symbol.strip(), int(window)))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SMA dashboard from stored OHLCV data.")
    parser.add_argument(
        "--asset",
        action="append",
        dest="assets",
        default=[],
        help="Asset/window pair as SYMBOL:WINDOW (e.g. ETH-USD:50). Repeat for multiple.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("html_reports"),
        help="Directory to write the dashboard file.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Number of trailing days to show in the history table.",
    )
    parser.add_argument(
        "--close-hour",
        type=int,
        default=11,
        help="Close hour passed to fetch_ohlcv (0-23, end-labelled).",
    )
    parser.add_argument(
        "--start-date",
        type=lambda value: datetime.fromisoformat(value).replace(tzinfo=timezone.utc),
        default=None,
        help="ISO timestamp (UTC assumed) for the earliest daily bar to include.",
    )
    parser.add_argument(
        "--email",
        action="store_true",
        help="Send a summary email using the configured EmailClient.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=("local", "vm"),
        help="Job profile to use (local or vm).",
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    email_client = EmailClient()
    db_url: str | None = None
    db_path: Path | None = None

    os.environ["JOB_PROFILE"] = args.profile

    try:
        job_conf = load_job_config("market_data_access")
    except FileNotFoundError:
        job_conf = {}

    db_url = job_conf.get("db_url") if isinstance(job_conf, dict) else None
    db_path_value = job_conf.get("db_path") if isinstance(job_conf, dict) else None
    if db_path_value:
        path_candidate = Path(db_path_value)
        if not path_candidate.is_absolute():
            path_candidate = BASE_DIR / path_candidate
        db_path = path_candidate
    if args.db_url is not None:
        db_url = args.db_url
    if args.db_path is not None:
        path_candidate = Path(args.db_path)
        if not path_candidate.is_absolute():
            path_candidate = BASE_DIR / path_candidate
        db_path = path_candidate

    pairs = parse_asset_pairs(args.assets) if args.assets else list(DEFAULT_ASSET_MA_PAIRS)
    try:
        output_path, _ = run_dashboard(
            asset_ma_pairs=pairs,
            output_dir=args.output_dir,
            lookback_days=args.lookback_days,
            close_hour=args.close_hour,
            start_date=args.start_date,
            email_recipient=args.email,
            email_client=email_client,
            db_url=db_url,
            db_path=db_path,
        )
        print(f"✅ Dashboard saved → {output_path}")
    except Exception as exc:
        tb = traceback.format_exc()
        message = f"Dashboard generation failed: {exc}"
        print(f"❌ {message}")
        print(tb)
        if args.email and email_client.enabled():
            subject = f"SMA Dashboard FAILED ({datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC})"
            error_html = (
                "<html><body>"
                "<h2>SMA Dashboard Generation Failed</h2>"
                f"<p>{message}</p>"
                "<pre style=\"background:#f4f4f4;padding:12px;border:1px solid #ccc;\">"
                f"{tb}"
                "</pre>"
                "</body></html>"
            )
            email_client.send_html(subject, error_html)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
