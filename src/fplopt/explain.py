"""How much do the model's forecasts and decisions rely on each signal family?

Grouped permutation importance (Breiman 2001; Fisher, Rudin & Dominici 2019), scored two ways.
Each *family* of features is shuffled jointly, within each gameweek (so the weekly distribution is
kept), on a frozen model, and the effect is measured on
  1. forecast RMSE, and
  2. the realised points of the team the optimiser then picks (the decision-focused view).
This measures reliance, not causal worth: a family the model can replace with a correlated one
(points form vs BPS, xG and price) can score near zero and still carry information.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import SEASONS, SETTINGS, TEST_SEASON, Settings
from .features import feature_columns
from .model import TwoStageModel
from .optimise import realised_points, solve_team

log = logging.getLogger(__name__)

FAMILIES = {
    "Playing time (minutes, starts, availability)": ("minutes_r", "starts_r", "share60_", "shareany_", "rows_since_played"),
    "Points form (points, bonus system)": ("total_points_r", "bps_r", "pts_per90_r10"),
    "Underlying stats (xG, xA, xGC, CS, saves, DefCon)": ("expected_goals_r", "expected_assists_r", "expected_goals_conceded_r",
                                                         "clean_sheets_r", "saves_r", "defensive_contribution_r"),
    "Own team strength": ("own_",),
    "Fixture (opponent strength, home, DGW)": ("opp_", "was_home", "n_fixtures", "is_dgw"),
    "Market and crowd (price, ownership, transfers)": ("price", "price_change", "log_selected_lag", "transfers_balance_lag"),
    "Longer history (season, last season)": ("season_ppg", "prev_ppg", "prev_minutes", "prev_xgi"),
}


def family_columns(cols: list[str]) -> dict[str, list[str]]:
    out = {f: [c for c in cols if c.startswith(pref)] for f, pref in FAMILIES.items()}
    return {f: c for f, c in out.items() if c}


def _decision_points(test: pd.DataFrame, mu: np.ndarray, s: Settings) -> float:
    pts = []
    for _, idx in test.groupby("GW").indices.items():
        c = test.iloc[idx].reset_index(drop=True)
        pts.append(realised_points(c, solve_team(c, mu[idx], s.budget_tenths, s.club_cap, s.captain)))
    return float(np.mean(pts))


def signal_value(F: pd.DataFrame, s: Settings = SETTINGS, split_gw: int = 20, repeats: int = 10) -> pd.DataFrame:
    """Train on everything before `split_gw` of the test season, evaluate on the rest."""
    cols = feature_columns(F)
    ti = SEASONS.index(TEST_SEASON)
    train = F[(F["season_idx"] < ti) | ((F["season"] == TEST_SEASON) & (F["GW"] < split_gw))]
    test = F[(F["season"] == TEST_SEASON) & (F["GW"] >= split_gw)].reset_index(drop=True)
    model = TwoStageModel(cols, alpha=s.interval_alpha).fit(train)
    y = test["total_points"].to_numpy(float)
    base_mu = model.predict(test)["mu"].to_numpy()
    base_rmse = float(np.sqrt(np.mean((y - base_mu) ** 2)))
    base_pts = _decision_points(test, base_mu, s)
    rng = np.random.default_rng(s.seed)
    groups = test.groupby("GW").indices
    rows = []
    for fam, fcols in family_columns(model.features).items():
        d_rmse, d_pts = [], []
        for _ in range(repeats):
            Xp = test.copy()
            perm = np.empty(len(test), dtype=int)
            for idx in groups.values():                       # shuffle rows within each gameweek
                perm[idx] = rng.permutation(idx)
            Xp[fcols] = test[fcols].to_numpy()[perm]
            mu = model.predict(Xp)["mu"].to_numpy()
            d_rmse.append(np.sqrt(np.mean((y - mu) ** 2)) - base_rmse)
            d_pts.append(base_pts - _decision_points(test, mu, s))
        rows.append({"family": fam, "n_features": len(fcols),
                     "rmse_increase": float(np.mean(d_rmse)),
                     "xi_points_lost_per_gw": float(np.mean(d_pts)), "xi_points_lost_sd": float(np.std(d_pts, ddof=1))})
        log.info("%s: RMSE %+.3f, XI points %+.1f per GW", fam, rows[-1]["rmse_increase"], -rows[-1]["xi_points_lost_per_gw"])
    out = pd.DataFrame(rows).sort_values("xi_points_lost_per_gw", ascending=False).reset_index(drop=True)
    out.attrs.update(base_rmse=base_rmse, base_pts=base_pts, gws=f"{split_gw}-{int(test['GW'].max())}")
    return out
