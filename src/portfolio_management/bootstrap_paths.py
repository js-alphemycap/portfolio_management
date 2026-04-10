from __future__ import annotations

import os
import sys
from pathlib import Path

from portfolio_management.helpers.config import BASE_DIR


def ensure_repo_paths() -> None:
    src = BASE_DIR / "src"
    infra_src = Path(os.environ.get("PRICE_DATA_INFRA_SRC", BASE_DIR.parent / "price_data_infra" / "src"))
    for candidate in (src, infra_src):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


__all__ = ["ensure_repo_paths"]
