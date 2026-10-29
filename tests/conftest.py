import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def candidates():
    """200 synthetic players across 20 clubs with realistic prices (tenths of GBP m)."""
    rng = np.random.default_rng(0)
    n = 200
    pos = rng.choice(["GK", "DEF", "MID", "FWD"], n, p=[0.1, 0.35, 0.4, 0.15])
    return pd.DataFrame({
        "position": pos,
        "team": [f"Club{i}" for i in rng.integers(0, 20, n)],
        "value": rng.integers(40, 131, n),
        "total_points": rng.poisson(3, n),
        "total_points_r1": rng.poisson(3, n),
        "total_points_r5": rng.uniform(0, 8, n),
        "mu": rng.uniform(0, 8, n),
    })
