import numpy as np

from fplopt.model import cqr_correction
from fplopt.monitor import ewma_chart


def test_cqr_reaches_target_coverage():
    rng = np.random.default_rng(0)
    y_cal, y_test = rng.normal(0, 2, 2000), rng.normal(0, 2, 5000)
    lo, hi = -1.0, 1.0                      # deliberately too narrow quantiles
    q = cqr_correction(y_cal, np.full(2000, lo), np.full(2000, hi), alpha=0.2)
    cover = np.mean((y_test >= lo - q) & (y_test <= hi + q))
    assert abs(cover - 0.8) < 0.03


def test_ewma_flags_a_persistent_shift():
    rng = np.random.default_rng(1)
    e = np.r_[rng.normal(0, 1, 20), rng.normal(2, 1, 15)]
    chart = ewma_chart(e, sigma=1.0)
    assert not chart["signal"].iloc[:15].any()
    assert chart["signal"].iloc[25:].any()
