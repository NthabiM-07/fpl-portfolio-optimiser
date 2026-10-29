"""Baseline strategies. Every baseline goes through the SAME optimiser and constraints as the model;
only the score being maximised changes. (The starter code's baselines broke the budget and position
rules, which made any comparison against them meaningless.)"""
from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_scores(cands: pd.DataFrame, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    out = {
        "top_price": cands["value"].to_numpy(float),                       # pick the most expensive XI
        "last_week": cands["total_points_r1"].fillna(0).to_numpy(float),     # last gameweek's points
        "mean_last5": cands["total_points_r5"].fillna(0).to_numpy(float),    # average of the last 5
    }
    if rng is not None:
        out["random"] = rng.random(len(cands))
    return out
