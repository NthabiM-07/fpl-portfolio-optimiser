"""Leakage forensics on the dataset's `xP` column (FPL's own expected points).

The FPL Historical Dataset README warns that `xP` can contain post-match information. Rather than
take that on trust, this audit tests it two ways on the gameweeks where xP is filled in:

1. Decay. A pre-match forecast captures a player's quality, role and fixtures, which persist, so it
   should predict next week's points almost as well as this week's. Information from *this* match
   (who scored, who got bonus) does not carry over, so a leaky column loses most of its edge one
   week later.
2. Partial correlation. After removing everything the model's pre-match forecast explains (expected
   points, chance of 60+ minutes, 5-week form), does xP still track this week's goals and bonus,
   and does it track next week's?

Neither test is proof; together they show whether the edge looks like knowledge or like hindsight.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import POSITIONS, RAW_DIR

log = logging.getLogger(__name__)


def _raw(season: str, raw_dir: Path) -> pd.DataFrame:
    cols = ["name", "element", "position", "GW", "xP", "goals_scored", "bonus"]
    raw = pd.read_csv(raw_dir / f"merged_gw_{season}.csv", low_memory=False, usecols=cols)
    raw = raw[raw["position"].isin(POSITIONS)]
    # A double gameweek has one row per fixture, each carrying the same xP: take the first xP, sum the rest.
    g = raw.groupby(["name", "element", "GW"], as_index=False).agg(xP=("xP", "first"), goals=("goals_scored", "sum"),
                                                                  bonus=("bonus", "sum"))
    clash = g.groupby("name")["element"].nunique()
    if (clash > 1).any():                      # two players with one name: drop them rather than mis-join
        log.warning("xP audit: dropping %d ambiguous names", int((clash > 1).sum()))
        g = g[g["name"].map(clash) == 1]
    return g.drop(columns="element")


def populated_gws(season: str, raw_dir: Path = RAW_DIR, min_share: float = 0.2) -> list[int]:
    """Gameweeks where xP is actually filled in (in some gameweeks every value is 0)."""
    xp = _raw(season, raw_dir)
    share = xp.assign(nz=xp["xP"] != 0).groupby("GW")["nz"].mean()
    return [int(g) for g in share.index[share > min_share]]


def _resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X1 = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def xp_audit(pred: pd.DataFrame, season: str = "2025-26", raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    gws = populated_gws(season, raw_dir)
    raw = _raw(season, raw_dir)
    p = pred.merge(raw, on=["name", "GW"], how="inner")
    nxt = raw.assign(GW=raw["GW"] - 1)[["name", "GW", "goals", "bonus"]].rename(columns={"goals": "goals_next", "bonus": "bonus_next"})
    pts_next = pred.assign(GW=pred["GW"] - 1)[["key", "GW", "total_points"]].rename(columns={"total_points": "pts_next"})
    p = p.merge(nxt, on=["name", "GW"], how="left").merge(pts_next, on=["key", "GW"], how="left")
    p = p[p["GW"].isin(gws)]
    both = p.dropna(subset=["pts_next", "goals_next", "bonus_next"])      # same rows for every comparison

    rows = []
    for label, f in [("two-stage model (pre-match)", "mu"), ("FPL xP column", "xP"), ("mean of last 5 (pre-match)", "total_points_r5")]:
        x = both[f].fillna(0)
        same, nextw = np.corrcoef(x, both["total_points"])[0, 1], np.corrcoef(x, both["pts_next"])[0, 1]
        rows.append({"forecast": label, "corr_points_this_gw": same, "corr_points_next_gw": nextw,
                     "share_of_edge_kept_next_gw": nextw / same,
                     "rmse_points_this_gw": float(np.sqrt(np.mean((both["total_points"] - x) ** 2)))})
    decay = pd.DataFrame(rows)

    X = both[["mu", "p_play60", "total_points_r5"]].fillna(0).to_numpy(float)
    rx = _resid(both["xP"].to_numpy(float), X)
    partial = []
    for what, col in [("goals", "goals"), ("bonus points", "bonus")]:
        r_same = np.corrcoef(rx, _resid(both[col].to_numpy(float), X))[0, 1]
        r_next = np.corrcoef(rx, _resid(both[f"{col}_next"].to_numpy(float), X))[0, 1]
        partial.append({"xP vs": what, "partial_corr_this_gw": r_same, "partial_corr_next_gw": r_next})
    part = pd.DataFrame(partial)
    decay.attrs["gws"] = gws
    decay.attrs["rows"] = len(both)
    return {"xp_audit": decay, "xp_partial": part}
