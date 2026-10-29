"""Statistical process control for the model: an EWMA chart of weekly prediction error.

EWMA_t = lam * e_t + (1 - lam) * EWMA_{t-1};  limits = target +/- L * sigma * sqrt(lam / (2 - lam))
(NIST/SEMATECH e-Handbook, section 6.3.2.4). EWMA charts detect small, persistent drifts, which is
what a scoring-rule change (defensive contributions in 2025-26) looks like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_chart(errors: np.ndarray, lam: float = 0.2, L: float = 3.0, target: float = 0.0,
               sigma: float | None = None) -> pd.DataFrame:
    e = np.asarray(errors, float)
    if sigma is None:
        sigma = float(np.std(e, ddof=1)) if len(e) > 1 else 0.0
    z, prev = np.empty_like(e), target
    for i, v in enumerate(e):
        prev = lam * v + (1 - lam) * prev
        z[i] = prev
    half = L * sigma * np.sqrt(lam / (2 - lam))
    return pd.DataFrame({"error": e, "ewma": z, "lcl": target - half, "ucl": target + half,
                         "signal": (z < target - half) | (z > target + half)})


def weekly_error_by_position(pred: pd.DataFrame) -> pd.DataFrame:
    """Mean (actual - predicted) points per gameweek and position, over players who had a fixture."""
    d = pred.assign(err=pred["total_points"] - pred["mu"])
    return d.groupby(["GW", "position"])["err"].mean().unstack("position").sort_index()
