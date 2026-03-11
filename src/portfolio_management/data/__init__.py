"""Portfolio-management data access delegated to price_data_infra."""

from price_data_infra.data import fetch_ohlcv

__all__ = ["fetch_ohlcv"]
