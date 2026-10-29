"""The data-calibrated robust budget uses only past gameweeks and measures shortfalls the way the
Bertsimas-Sim uncertainty set defines them."""
import numpy as np

from fplopt.backtest import budget_used, calibrated_gamma


def test_budget_used_counts_fractional_shortfalls_capped_at_one():
    mu, lo = np.array([6.0, 6.0, 6.0, 6.0]), np.array([2.0, 2.0, 2.0, 2.0])
    y = np.array([6.0, 4.0, 2.0, -5.0])          # no shortfall, half, full, beyond the bound
    assert budget_used(mu, lo, y) == 0 + 0.5 + 1 + 1


def test_calibrated_gamma_waits_for_history_then_tracks_the_quantile():
    assert calibrated_gamma([], 0.8, 4) == 0
    assert calibrated_gamma([3.0, 5.0, 4.0], 0.8, 4) == 0
    hist = [2.0, 3.0, 4.0, 5.0, 6.0]
    assert calibrated_gamma(hist, 0.8, 4) == float(np.round(np.quantile(hist, 0.8)))
    assert calibrated_gamma(hist + [20.0], 0.8, 4) >= calibrated_gamma(hist, 0.8, 4)
