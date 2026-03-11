#!/usr/bin/env python3
"""Email a daily SMA status matrix for configured alt symbols."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import base64
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from price_data_infra.data import fetch_ohlcv
from portfolio_management.helpers.config import BASE_DIR
from portfolio_management.helpers.email import EmailClient
from portfolio_management.helpers.job_config import load_job_config


def _resolve_db_path(db_path_value: str | None) -> Path | None:
    if not db_path_value:
        return None
    path_candidate = Path(db_path_value)
    if not path_candidate.is_absolute():
        path_candidate = BASE_DIR / path_candidate
    return path_candidate


def _load_close_series(
    symbol: str,
    *,
    close_hour: int,
    start_date: datetime | None,
    db_url: str | None,
    db_path: Path | None,
) -> pd.Series:
    df = fetch_ohlcv(
        symbol,
        frequency="daily",
        close_hour=close_hour,
        start=start_date,
        db_url=db_url,
        db_path=db_path,
    )
    if df.empty or "close" not in df.columns:
        raise ValueError(f"No daily close series available for {symbol}.")
    series = df["close"].astype(float).rename(symbol)
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, utc=True)
    elif series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")
    return series.dropna()


def _parse_start_date(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_alts(raw: Iterable[str]) -> list[str]:
    return [str(item).strip() for item in raw if str(item).strip()]


def build_status_frame(
    *,
    alts: list[str],
    window: int,
    close_hour: int,
    start_date: datetime | None,
    db_url: str | None,
    db_path: Path | None,
    btc_symbol: str = "BTC-USD",
    eth_symbol: str = "ETH-USD",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (status, latest_values):
      - status: DataFrame with rows=alts and columns=[Alt/BTC, Alt/ETH, Alt/Cash]
                values in {1 (green), 0 (red), NaN (no data)}
      - latest_values: DataFrame with latest series and SMA values for labeling.
    """
    if window <= 0:
        raise ValueError("window must be > 0")

    btc = _load_close_series(
        btc_symbol, close_hour=close_hour, start_date=start_date, db_url=db_url, db_path=db_path
    )
    eth = _load_close_series(
        eth_symbol, close_hour=close_hour, start_date=start_date, db_url=db_url, db_path=db_path
    )

    columns = ["Alt/BTC", "Alt/ETH", "Alt/Cash"]
    status = pd.DataFrame(index=alts, columns=columns, dtype="float")
    details_rows: list[dict[str, object]] = []

    for alt in alts:
        alt_close = _load_close_series(
            alt, close_hour=close_hour, start_date=start_date, db_url=db_url, db_path=db_path
        )
        idx = alt_close.index.intersection(btc.index).intersection(eth.index).sort_values()
        if idx.empty:
            status.loc[alt] = np.nan
            continue

        alt_close = alt_close.loc[idx]
        btc_close = btc.loc[idx]
        eth_close = eth.loc[idx]

        rel_btc = (alt_close / btc_close).rename("Alt/BTC")
        rel_eth = (alt_close / eth_close).rename("Alt/ETH")
        rel_cash = alt_close.rename("Alt/Cash")

        series_map = {"Alt/BTC": rel_btc, "Alt/ETH": rel_eth, "Alt/Cash": rel_cash}
        latest_date = idx[-1]

        for col, ser in series_map.items():
            sma = ser.rolling(int(window), min_periods=int(window)).mean()
            latest_val = float(ser.loc[latest_date]) if pd.notna(ser.loc[latest_date]) else np.nan
            latest_sma = float(sma.loc[latest_date]) if pd.notna(sma.loc[latest_date]) else np.nan
            if np.isnan(latest_val) or np.isnan(latest_sma):
                status.loc[alt, col] = np.nan
                is_green = None
            else:
                is_green = latest_val >= latest_sma
                status.loc[alt, col] = 1.0 if is_green else 0.0

            details_rows.append(
                {
                    "alt": alt,
                    "metric": col,
                    "date": latest_date,
                    "latest": latest_val,
                    "sma": latest_sma,
                    "green": is_green,
                }
            )

    details = pd.DataFrame(details_rows)
    return status, details


def render_status_matrix_png(status: pd.DataFrame, *, window: int, as_of: pd.Timestamp) -> bytes:
    """
    Render a heatmap-style matrix as PNG bytes.
    Green = above SMA, Red = below SMA, Grey = missing.
    """
    def _display_symbol(symbol: str) -> str:
        # e.g. "LINK-USD" -> "LINK"
        return str(symbol).split("-", maxsplit=1)[0].strip()

    alts = list(status.index)
    metrics = list(status.columns)
    values = status.to_numpy(dtype=float)

    # Map {0,1,nan} -> colormap indices.
    # 0: red, 1: green, nan: grey.
    mapped = np.where(np.isnan(values), 0.5, values)

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#d84a4a", "#aaaaaa", "#2fbf71"])
    # Values: 0.0 -> red, 0.5 -> grey, 1.0 -> green
    bounds = [-0.01, 0.25, 0.75, 1.01]
    from matplotlib.colors import BoundaryNorm

    norm = BoundaryNorm(bounds, cmap.N)

    fig_w = max(6.0, 1.2 + 1.2 * len(metrics))
    fig_h = max(4.0, 1.2 + 0.45 * len(alts))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(mapped, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks(range(len(alts)))
    ax.set_yticklabels([_display_symbol(sym) for sym in alts], fontsize=10)

    ax.set_title(f"Alts SMA Monitoring (window={window}) — {as_of.date()}", fontsize=12)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Draw grid lines.
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(alts), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # No cell annotations; color conveys the signal.

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def build_email_html(image_png: bytes, *, window: int, as_of: pd.Timestamp) -> str:
    img64 = base64.b64encode(image_png).decode("utf-8")
    return f"""
<html>
  <body>
    <h2>Alts SMA Monitoring</h2>
    <p>
      Date (UTC): <b>{as_of:%Y-%m-%d}</b><br/>
      Window: <b>{window}</b><br/>
      Green = above SMA, Red = below SMA, Grey = missing.
    </p>
    <img src="data:image/png;base64,{img64}" style="max-width:100%; height:auto;" />
  </body>
</html>
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Email a daily SMA status matrix for alt symbols."
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=("local", "vm"),
        help="Job profile to use (local or vm) for DB settings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated HTML instead of sending email.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.environ["JOB_PROFILE"] = args.profile
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    job_conf = load_job_config("alts_sma_monitoring", use_profile=False)
    alts = _normalize_alts(job_conf.get("alts", []))
    if not alts:
        raise SystemExit("alts_sma_monitoring config has no alts.")

    window = int(job_conf.get("window", 50))
    close_hour = int(job_conf.get("close_hour", 12))
    start_date = _parse_start_date(job_conf.get("start_date"))

    try:
        db_conf = load_job_config("market_data_access")
    except FileNotFoundError:
        db_conf = {}

    db_url = db_conf.get("db_url") if isinstance(db_conf, dict) else None
    db_path = _resolve_db_path(db_conf.get("db_path") if isinstance(db_conf, dict) else None)

    status, details = build_status_frame(
        alts=alts,
        window=window,
        close_hour=close_hour,
        start_date=start_date,
        db_url=db_url,
        db_path=db_path,
    )

    if details.empty:
        raise SystemExit("No status data produced; check DB contents and symbols.")

    as_of = pd.to_datetime(details["date"].max(), utc=True)
    png = render_status_matrix_png(status, window=window, as_of=as_of)
    html = build_email_html(png, window=window, as_of=as_of)

    subject = f"Alts SMA Monitoring ({as_of:%Y-%m-%d} UTC)"

    if args.dry_run:
        print(html)
        return

    client = EmailClient()
    client.send_html(subject, html)


if __name__ == "__main__":
    main()
