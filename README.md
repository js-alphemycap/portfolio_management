# portfolio_management

Independent portfolio-management repo for:
- generating strategy signals from stored OHLCV data
- producing dashboards and Telegram messages
- depending on `price_data_infra` for market-data access

## Responsibility
`portfolio_management` does not fetch from Coindesk.
It imports `fetch_ohlcv` from the sibling `price_data_infra` repo and reads the database populated by `price_data_infra`.

## Included jobs
- `scripts/send_telegram_dual_ma_strategy.py`
- `scripts/send_telegram_sol_eth_rotation_strategy.py`
- `scripts/send_telegram_hype_eth_rotation_strategy.py`
- `scripts/send_telegram_watchlist.py`
- `scripts/send_telegram_dashboard_summary.py`
- `scripts/send_telegram_alts_sma_monitoring.py`
- `scripts/sma_dashboard.py`
- `scripts/alts_sma_monitoring.py`

## Config
- strategy configs remain under `configs/jobs/`
- market-data DB access lives in `configs/jobs/market_data_access.yaml`
- trade logs remain under `configs/state/`

## Dependency setup
Install from this repo with the sibling dependency in place:
- `pip install -r requirements.txt`

`requirements.txt` includes `-e ../price_data_infra`, so this repo resolves `price_data_infra` from the sibling folder by default.

## Workflow
- `vm`: run against the database created by `price_data_infra` on the VM.
- `local`: sync data from the VM first, then run locally against that synced database.
- No trading logic was changed from the original flow.
