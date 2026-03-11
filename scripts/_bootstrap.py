from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
INFRA_SRC = Path(os.environ.get("PRICE_DATA_INFRA_SRC", ROOT.parent / "price_data_infra" / "src"))

for candidate in (SRC, INFRA_SRC):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
