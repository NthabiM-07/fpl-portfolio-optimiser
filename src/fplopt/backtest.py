"""Walk-forward backtest on the 2025-26 season.

1. Predict: retrain every `retrain_every` gameweeks on everything before the block (expanding window),
   predict the block, and calibrate the 80% interval with CQR scores from the most recent
   `calib_gws` gameweeks of genuinely out-of-sample predictions (a 2024-25 warm-up seeds the first block).
2. Decide: for every test gameweek, pick an XI + captain with the same MILP for every strategy and
   score it on the points those players actually scored.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .baselines import baseline_scores
from .config import SEASONS, SETTINGS, TEST_SEASON, Settings
from .features import feature_columns
from .model import SingleModel, TwoStageModel, cqr_correction
from .optimise import budget_shadow_price, prefilter, realised_points, solve_team

log = logging.getLogger(__name__)
ID_COLS = ["season", "GW", "key", "name", "position", "team", "value", "total_points", "minutes",
           "total_points_r1", "total_points_r5", "n_fixtures"]


def _conformal_q(records: list[pd.DataFrame], calib_gws: int, alpha: float) -> float:
    if not records:
        return 0.0
    hist = pd.concat(records)
    keys = hist[["season", "GW"]].drop_duplicates().tail(calib_gws)
    recent = hist.merge(keys, on=["season", "GW"])
    return cqr_correction(recent["total_points"], recent["q_lo_raw"], recent["q_hi_raw"], alpha)


def walk_forward_predict(F: pd.DataFrame, s: Settings = SETTINGS, test_season: str = TEST_SEASON) -> pd.DataFrame:
    cols = feature_columns(F)
    test_idx = SEASONS.index(test_season)
    records: list[pd.DataFrame] = []

    # Warm-up: out-of-sample predictions for the last `calib_gws` gameweeks of the previous season.
    prev = SEASONS[test_idx - 1]
    last_gw = int(F.loc[F["season"] == prev, "GW"].max())
    w0 = last_gw - s.calib_gws + 1
    train = F[(F["season_idx"] < test_idx - 1) | ((F["season"] == prev) & (F["GW"] < w0))]
    warm = F[(F["season"] == prev) & (F["GW"] >= w0)]
    m = TwoStageModel(cols, alpha=s.interval_alpha).fit(train)
    records.append(pd.concat([warm[ID_COLS], m.predict(warm)], axis=1))

    out = []
    test = F[F["season"] == test_season]
    gws = sorted(test["GW"].unique())
    for b in range(min(gws), max(gws) + 1, s.retrain_every):
        t0 = time.time()
        block = test[(test["GW"] >= b) & (test["GW"] < b + s.retrain_every)]
        if block.empty:
            continue
        train = F[(F["season_idx"] < test_idx) | ((F["season"] == test_season) & (F["GW"] < b))]
        two = TwoStageModel(cols, alpha=s.interval_alpha).fit(train)
        one = SingleModel(cols).fit(train)
        q = _conformal_q(records, s.calib_gws, s.interval_alpha)
        pr = pd.concat([block[ID_COLS], two.predict(block)], axis=1)
        pr["mu_single"] = one.predict(block)
        pr["cqr_q"] = q
        pr["lo"] = pr["q_lo_raw"] - q
        pr["hi"] = pr["q_hi_raw"] + q
        records.append(pr[ID_COLS + ["q_lo_raw", "q_hi_raw"]])
        out.append(pr)
        log.info("block GW%d-%d: train %d rows, q=%.2f, %.1fs", b, b + s.retrain_every - 1, len(train), q, time.time() - t0)
    return pd.concat(out).reset_index(drop=True)


def decide(pred: pd.DataFrame, s: Settings = SETTINGS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (gw_results, robust_results): realised XI points per gameweek and strategy."""
    rng = np.random.default_rng(s.seed)
    rows, rob = [], []
    for gw, c in pred.groupby("GW"):
        c = c.reset_index(drop=True)
        scores = {"model": c["mu"].to_numpy(), "model_single": c["mu_single"].to_numpy(),
                  **baseline_scores(c), "oracle": c["total_points"].to_numpy(float)}
        rec = {"GW": gw, "candidates": len(c)}
        for name, sc in scores.items():
            team = solve_team(c, sc, s.budget_tenths, s.club_cap, s.captain)
            rec[name] = realised_points(c, team)
            if name == "model":
                rec["model_predicted"] = team.objective
                rec["model_cost"] = c["value"].to_numpy()[team.idx].sum() / 10
        rnd = [realised_points(c, solve_team(c, rng.random(len(c)), s.budget_tenths, s.club_cap, s.captain))
               for _ in range(s.random_baseline_seeds)]
        rec["random"] = float(np.mean(rnd))
        # Sensitivity: value of budget, cost of the club cap.
        rec["budget_shadow_pts_per_m"] = 10 * budget_shadow_price(c, c["mu"].to_numpy(), s.budget_tenths, s.club_cap)
        nocap = solve_team(c, c["mu"].to_numpy(), s.budget_tenths, 11, s.captain)
        rec["club_cap_cost_pred"] = nocap.objective - rec["model_predicted"]
        rows.append(rec)
        # Robust sweep on a prefiltered pool (top 60 per position by mu).
        keep = prefilter(c, c["mu"].to_numpy(), 60)
        cc = c.iloc[keep].reset_index(drop=True)
        d = (cc["mu"] - cc["lo"]).clip(lower=0).to_numpy()
        for g in s.gammas:
            team = solve_team(cc, cc["mu"].to_numpy(), s.budget_tenths, s.club_cap, s.captain, d=d, gamma=g, thetas=32)
            rob.append({"GW": gw, "gamma": g, "realised": realised_points(cc, team),
                        "predicted_mu": float(cc["mu"].to_numpy()[team.idx].sum() + (cc["mu"].to_numpy()[team.captain] if team.captain is not None else 0)),
                        "downside_sum": float(d[team.idx].sum())})
        log.info("GW%d done", gw)
    return pd.DataFrame(rows), pd.DataFrame(rob)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    bs = np.random.default_rng(seed).choice(d, (n_boot, len(d))).mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def strategy_table(gw: pd.DataFrame, gw_from: int = 1) -> pd.DataFrame:
    g = gw[gw["GW"] >= gw_from]
    strategies = ["model", "model_single", "mean_last5", "last_week", "top_price", "random", "oracle"]
    rows = []
    for st in strategies:
        r = {"strategy": st, "mean_pts_per_gw": g[st].mean(), "total_pts": g[st].sum()}
        for base in ["last_week", "mean_last5", "top_price", "random"]:
            r[f"uplift_vs_{base}_%"] = 100 * (g[st].mean() / g[base].mean() - 1)
        rows.append(r)
    return pd.DataFrame(rows)


def significance_table(gw: pd.DataFrame, gw_from: int = 1) -> pd.DataFrame:
    g = gw[gw["GW"] >= gw_from]
    rows = []
    for base in ["last_week", "mean_last5", "top_price", "random", "model_single"]:
        m, lo, hi = paired_bootstrap(g["model"], g[base])
        rows.append({"model_vs": base, "mean_diff_pts_per_gw": m, "ci95_low": lo, "ci95_high": hi,
                     "model_better_gws": int((g["model"] > g[base]).sum()), "gws": len(g)})
    return pd.DataFrame(rows)


def forecast_table(pred: pd.DataFrame) -> pd.DataFrame:
    y = pred["total_points"].to_numpy(float)
    forecasts = {"two_stage": pred["mu"], "single_gbm": pred["mu_single"],
                 "mean_last5": pred["total_points_r5"].fillna(0), "last_week": pred["total_points_r1"].fillna(0)}
    band = pd.cut(pred["total_points"], [-np.inf, 2, 4, np.inf], labels=["0-2 pts", "3-4 pts", "5+ pts"])
    rows = []
    for name, f in forecasts.items():
        f = f.to_numpy(float)
        r = {"forecast": name, "rmse": np.sqrt(np.mean((y - f) ** 2)), "mae": np.mean(np.abs(y - f))}
        for p in ["GK", "DEF", "MID", "FWD"]:
            m = (pred["position"] == p).to_numpy()
            r[f"rmse_{p}"] = np.sqrt(np.mean((y[m] - f[m]) ** 2))
        played = (pred["minutes"] > 0).to_numpy()
        r["rmse_played"] = np.sqrt(np.mean((y[played] - f[played]) ** 2))
        for bnd in band.cat.categories:
            m = (band == bnd).to_numpy()
            r[f"rmse_{bnd}"] = np.sqrt(np.mean((y[m] - f[m]) ** 2))
        rows.append(r)
    return pd.DataFrame(rows)


def interval_table(pred: pd.DataFrame) -> pd.DataFrame:
    inside = (pred["total_points"] >= pred["lo"]) & (pred["total_points"] <= pred["hi"])
    raw = (pred["total_points"] >= pred["q_lo_raw"]) & (pred["total_points"] <= pred["q_hi_raw"])
    t = pred.assign(inside=inside, raw=raw, width=pred["hi"] - pred["lo"])
    by = t.groupby("position").agg(coverage=("inside", "mean"), raw_coverage=("raw", "mean"), mean_width=("width", "mean"))
    allr = pd.DataFrame({"coverage": [inside.mean()], "raw_coverage": [raw.mean()], "mean_width": [t["width"].mean()]}, index=["ALL"])
    played = t[t["minutes"] > 0]
    pl = pd.DataFrame({"coverage": [played["inside"].mean()], "raw_coverage": [played["raw"].mean()],
                       "mean_width": [played["width"].mean()]}, index=["played only"])
    return pd.concat([allr, by, pl])
