"""Walk-forward backtest on the 2025-26 season.

1. Predict: retrain every `retrain_every` gameweeks on everything before the block (expanding window),
   predict the block, and calibrate the 80% interval with CQR scores from the most recent
   `calib_gws` gameweeks of genuinely out-of-sample predictions (a 2024-25 warm-up seeds the first block).
   Calibration is Mondrian (group-conditional): likely starters (P(60+ min) >= 0.5, known before the
   deadline) and everyone else get separate corrections. Pooled, the ~60% of zero-minute rows score
   exactly 0 and pin the correction at 0, which says nothing about the players the optimiser picks.
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


STARTER_P = 0.5      # Mondrian group: predicted P(60+ min) at or above this = "likely starter"


def is_likely_starter(p_play60) -> np.ndarray:
    return np.asarray(p_play60, float) >= STARTER_P


def _conformal_q(records: list[pd.DataFrame], calib_gws: int, alpha: float) -> dict[bool, float]:
    """CQR corrections per Mondrian group, from the last `calib_gws` gameweeks of out-of-sample rows."""
    if not records:
        return {True: 0.0, False: 0.0}
    hist = pd.concat(records)
    keys = hist[["season", "GW"]].drop_duplicates().tail(calib_gws)
    recent = hist.merge(keys, on=["season", "GW"])
    g = is_likely_starter(recent["p_play60"])
    return {k: cqr_correction(recent.loc[g == k, "total_points"], recent.loc[g == k, "q_lo_raw"],
                              recent.loc[g == k, "q_hi_raw"], alpha) for k in (True, False)}


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
        qd = _conformal_q(records, s.calib_gws, s.interval_alpha)
        pr = pd.concat([block[ID_COLS], two.predict(block)], axis=1)
        pr["mu_single"] = one.predict(block)
        pr["cqr_q"] = np.where(is_likely_starter(pr["p_play60"]), qd[True], qd[False])
        pr["lo"] = pr["q_lo_raw"] - pr["cqr_q"]
        pr["hi"] = pr["q_hi_raw"] + pr["cqr_q"]
        records.append(pr[ID_COLS + ["p_play60", "q_lo_raw", "q_hi_raw"]])
        out.append(pr)
        log.info("block GW%d-%d: train %d rows, q starters=%.2f others=%.2f, %.1fs", b, b + s.retrain_every - 1,
                 len(train), qd[True], qd[False], time.time() - t0)
    return pd.concat(out).reset_index(drop=True)


def budget_used(mu: np.ndarray, lo: np.ndarray, y: np.ndarray) -> float:
    """How much of the Bertsimas-Sim uncertainty budget a team actually used: each player's shortfall
    as a fraction of his allowed downside d = mu - lo, capped at 1, summed over the XI."""
    d = np.clip(mu - lo, 1e-9, None)
    return float(np.clip((mu - y) / d, 0, 1).sum())


def calibrated_gamma(history: list[float], quantile: float, min_history: int) -> float:
    """Choose the robust budget from data instead of by hand: the `quantile` of the budget the
    risk-neutral XI used in PAST gameweeks (0 until `min_history` weeks exist), rounded to a whole
    player. In the spirit of Zhou & Zhu (2025, arXiv:2510.07750), who calibrate robustness levels
    from data; this is the simplest leakage-free version, using only gameweeks already played."""
    if len(history) < min_history:
        return 0.0
    return float(np.round(np.quantile(history, quantile)))


def decide(pred: pd.DataFrame, s: Settings = SETTINGS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (gw_results, robust_results): realised XI points per gameweek and strategy."""
    rng = np.random.default_rng(s.seed)
    rows, rob = [], []
    shortfalls: list[int] = []
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
                model_team = team
        rnd = [realised_points(c, solve_team(c, rng.random(len(c)), s.budget_tenths, s.club_cap, s.captain))
               for _ in range(s.random_baseline_seeds)]
        rec["random"] = float(np.mean(rnd))
        # Sensitivity: value of budget, cost of the club cap.
        rec["budget_shadow_pts_per_m"] = 10 * budget_shadow_price(c, c["mu"].to_numpy(), s.budget_tenths, s.club_cap)
        nocap = solve_team(c, c["mu"].to_numpy(), s.budget_tenths, 11, s.captain)
        rec["club_cap_cost_pred"] = nocap.objective - rec["model_predicted"]
        rec["nocap_max_per_club"] = int(pd.Series(c["team"].to_numpy()[nocap.idx]).value_counts().max())
        rows.append(rec)
        # Robust sweep on a prefiltered pool (top 60 per position by mu).
        keep = prefilter(c, c["mu"].to_numpy(), 60)
        cc = c.iloc[keep].reset_index(drop=True)
        d = (cc["mu"] - cc["lo"]).clip(lower=0).to_numpy()
        for g in s.gammas:
            team = solve_team(cc, cc["mu"].to_numpy(), s.budget_tenths, s.club_cap, s.captain, d=d, gamma=g, thetas=32)
            mu_cc = cc["mu"].to_numpy()
            captain_mu = mu_cc[team.captain] if team.captain is not None else 0.0
            rob.append({"GW": gw, "gamma": g, "realised": realised_points(cc, team),
                        "predicted_mu": float(mu_cc[team.idx].sum() + captain_mu),
                        "downside_sum": float(d[team.idx].sum())})
        # Calibrated Gamma: chosen from past weeks only, then scored on this week.
        g_cal = calibrated_gamma(shortfalls, s.gamma_quantile, s.gamma_min_history)
        if g_cal > 0:
            t_cal = solve_team(cc, cc["mu"].to_numpy(), s.budget_tenths, s.club_cap, s.captain, d=d, gamma=g_cal, thetas=32)
            rec["model_robust_cal"] = realised_points(cc, t_cal)
            rec["robust_floor"] = t_cal.objective          # the worst case the robust team was built to beat
        else:
            rec["model_robust_cal"] = rec["model"]
            rec["robust_floor"] = np.nan
        rec["gamma_cal"] = g_cal
        idx = model_team.idx
        used = budget_used(c["mu"].to_numpy()[idx], c["lo"].to_numpy()[idx], c["total_points"].to_numpy(float)[idx])
        rec["budget_used_model_xi"] = used
        shortfalls.append(used)                  # revealed after the gameweek; used from the next one on
        log.info("GW%d done", gw)
    return pd.DataFrame(rows), pd.DataFrame(rob)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    bs = np.random.default_rng(seed).choice(d, (n_boot, len(d))).mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def strategy_table(gw: pd.DataFrame, gw_from: int = 1) -> pd.DataFrame:
    g = gw[gw["GW"] >= gw_from]
    strategies = ["model", "model_robust_cal", "model_single", "mean_last5", "last_week", "top_price", "random", "oracle"]
    strategies = [st for st in strategies if st in g.columns]
    rows = []
    for st in strategies:
        r = {"strategy": st, "mean_pts_per_gw": g[st].mean(), "total_pts": g[st].sum()}
        for base in ["last_week", "mean_last5", "top_price", "random"]:
            r[f"uplift_vs_{base}_%"] = 100 * (g[st].mean() / g[base].mean() - 1)
        rows.append(r)
    return pd.DataFrame(rows)


def robust_calibrated_table(gw: pd.DataFrame) -> pd.DataFrame:
    """Risk-neutral XI vs the XI with the data-calibrated Gamma, over the same 38 gameweeks."""
    rows = []
    for st, label in [("model", "risk-neutral (Gamma = 0)"), ("model_robust_cal", "calibrated Gamma (80th pct of past budget used)")]:
        x = gw[st]
        rows.append({"strategy": label, "mean": x.mean(), "p10": x.quantile(0.1), "worst": x.min(), "sd": x.std()})
    out = pd.DataFrame(rows)
    m, lo, hi = paired_bootstrap(gw["model_robust_cal"], gw["model"])
    on = gw["robust_floor"].notna()
    out.attrs.update(diff=m, ci_low=lo, ci_high=hi, gammas=gw["gamma_cal"].value_counts().sort_index().to_dict(),
                     budget_used_median=float(gw["budget_used_model_xi"].median()),
                     budget_used_q80=float(gw["budget_used_model_xi"].quantile(0.8)),
                     floor_beaten_share=float((gw.loc[on, "model_robust_cal"] >= gw.loc[on, "robust_floor"]).mean()),
                     floor_weeks=int(on.sum()))
    return out


def significance_table(gw: pd.DataFrame, gw_from: int = 1) -> pd.DataFrame:
    g = gw[gw["GW"] >= gw_from]
    rows = []
    for base in ["last_week", "mean_last5", "top_price", "random", "model_single"]:
        m, lo, hi = paired_bootstrap(g["model"], g[base])
        rows.append({"model_vs": base, "mean_diff_pts_per_gw": m, "ci95_low": lo, "ci95_high": hi,
                     "model_better_gws": int((g["model"] > g[base]).sum()), "gws": len(g)})
    return pd.DataFrame(rows)


def _forecasts(pred: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"two_stage": pred["mu"].to_numpy(float), "single_gbm": pred["mu_single"].to_numpy(float),
            "mean_last5": pred["total_points_r5"].fillna(0).to_numpy(float),
            "last_week": pred["total_points_r1"].fillna(0).to_numpy(float)}


def outcome_bucket(pred: pd.DataFrame) -> pd.Series:
    """OpenFPL's evaluation buckets (Groos 2025): Zeros = did not play, Blanks <= 2, Tickers 3-4, Haulers >= 5."""
    pts, played = pred["total_points"], pred["minutes"] > 0
    return pd.Series(np.select([~played, pts <= 2, pts <= 4], ["zeros", "blanks", "tickers"], "haulers"), index=pred.index)


def forecast_table(pred: pd.DataFrame) -> pd.DataFrame:
    y = pred["total_points"].to_numpy(float)
    bucket = outcome_bucket(pred).to_numpy()
    rows = []
    for name, f in _forecasts(pred).items():
        r = {"forecast": name, "rmse": np.sqrt(np.mean((y - f) ** 2)), "mae": np.mean(np.abs(y - f))}
        for p in ["GK", "DEF", "MID", "FWD"]:
            m = (pred["position"] == p).to_numpy()
            r[f"rmse_{p}"] = np.sqrt(np.mean((y[m] - f[m]) ** 2))
        played = (pred["minutes"] > 0).to_numpy()
        r["rmse_played"] = np.sqrt(np.mean((y[played] - f[played]) ** 2))
        for b in ["zeros", "blanks", "tickers", "haulers"]:
            m = bucket == b
            r[f"rmse_{b}"] = np.sqrt(np.mean((y[m] - f[m]) ** 2))
        rows.append(r)
    return pd.DataFrame(rows)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 3) -> tuple[float, float]:
    """Diebold-Mariano (1995) test of equal expected loss for forecasts A and B. Inputs are per-period
    losses; a positive statistic means A has the lower loss. The variance is Newey-West (Bartlett
    kernel) with `lag` lags (default 3 ~ T^(1/3) for T = 38, a guard against serial correlation in
    weekly losses); the statistic is scaled by sqrt((T-1)/T), the Harvey, Leybourne & Newbold (1997)
    factor for one-step forecasts, and compared with t(T-1). Returns (statistic, two-sided p)."""
    from scipy import stats

    d = np.asarray(loss_b, float) - np.asarray(loss_a, float)        # > 0 when A is better
    T = len(d)
    dc = d - d.mean()
    v = dc @ dc / T
    for k in range(1, min(lag, T - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T
    if v <= 0:                                                       # identical losses: no evidence either way
        return 0.0, 1.0
    dm = d.mean() / np.sqrt(v / T)
    dm *= np.sqrt((T - 1) / T)                                       # HLN correction, h = 1
    return float(dm), float(2 * stats.t.sf(abs(dm), df=T - 1))


def forecast_dm_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Is the two-stage model's forecast error significantly lower? Losses are averaged within each
    gameweek first (players in the same gameweek share shocks), then tested across the 38 gameweeks."""
    y = pred["total_points"].to_numpy(float)
    fc = _forecasts(pred)
    gw = pred["GW"].to_numpy()
    def per_gw(f: np.ndarray, kind: str) -> np.ndarray:
        loss = (y - f) ** 2 if kind == "se" else np.abs(y - f)
        return pd.Series(loss).groupby(gw).mean().to_numpy()

    rows = []
    for base in ["single_gbm", "mean_last5", "last_week"]:
        r = {"model_vs": base}
        for kind in ("se", "ae"):
            a, b = per_gw(fc["two_stage"], kind), per_gw(fc[base], kind)
            stat, p = diebold_mariano(a, b)
            r[f"dm_{kind}"], r[f"p_{kind}"] = stat, p
            r[f"model_better_gws_{kind}"] = int((a < b).sum())
        r["gws"] = len(np.unique(gw))
        rows.append(r)
    return pd.DataFrame(rows)


def interval_table(pred: pd.DataFrame) -> pd.DataFrame:
    inside = (pred["total_points"] >= pred["lo"]) & (pred["total_points"] <= pred["hi"])
    raw = (pred["total_points"] >= pred["q_lo_raw"]) & (pred["total_points"] <= pred["q_hi_raw"])
    t = pred.assign(inside=inside, raw=raw, width=pred["hi"] - pred["lo"])
    agg = dict(coverage=("inside", "mean"), raw_coverage=("raw", "mean"), mean_width=("width", "mean"),
               mean_cqr_q=("cqr_q", "mean"), rows=("inside", "size"))
    parts = [t.assign(g="ALL").groupby("g").agg(**agg),
             t.assign(g=np.where(is_likely_starter(t["p_play60"]), "likely starters (P60 >= 0.5)", "others (P60 < 0.5)")).groupby("g").agg(**agg),
             t.groupby("position").agg(**agg),
             t[t["minutes"] > 0].assign(g="played (known only afterwards)").groupby("g").agg(**agg)]
    return pd.concat(parts)


def cqr_blocks(pred: pd.DataFrame, retrain_every: int = SETTINGS.retrain_every) -> pd.DataFrame:
    """The conformal correction used in each retraining block, per Mondrian group."""
    t = pred.assign(block=(pred["GW"] - 1) // retrain_every * retrain_every + 1,
                    group=np.where(is_likely_starter(pred["p_play60"]), "likely_starters", "others"))
    out = t.groupby(["block", "group"])["cqr_q"].first().unstack()
    out.index = [f"GW{b}-{min(b + retrain_every - 1, int(pred['GW'].max()))}" for b in out.index]
    return out.reset_index().rename(columns={"index": "block"})
