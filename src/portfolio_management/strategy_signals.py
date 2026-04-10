from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategySignalRecord:
    strategy_id: str
    signal_strategy_slug: str
    as_of: pd.Timestamp
    effective_signal_value: float
    target_weight: float | None = None
    trigger_today: bool | None = None
    current_state: str | None = None
    raw_signal_value: float | None = None


__all__ = ["StrategySignalRecord"]
