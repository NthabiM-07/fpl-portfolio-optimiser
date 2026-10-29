"""Side experiment: TabICL, an open tabular foundation model (BSD-3), as a challenger to gradient boosting.

    pip install tabicl            # pulls PyTorch; the checkpoint downloads from Hugging Face on first use
    python scripts/tabicl_challenger.py --gw-from 20 --context 4000

Design, fixed before running:
- Same walk-forward blocks as the main backtest (refit every 4 gameweeks), test gameweeks
  `gw_from`..38 of 2025-26, same 63 as-of features, target = total points (one stage, like the
  single-GBM ablation).
- TabICL learns *in context*: its training set is passed through the network at prediction time,
  so on a 2-core CPU it gets only the most recent `context` rows before each block. The main
  model trains on all ~80k earlier rows. To separate "better model" from "more data", a single
  gradient-boosting model is also fitted on exactly the same rows ("GBM, same rows").
- Scored two ways on the same player-gameweeks: forecast error, and the realised points of the
  XI that the same MILP picks from each forecast (GBP 83.5m, captain doubled), with a paired
  bootstrap against the main two-stage model.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fplopt.backtest import paired_bootstrap  # noqa: E402
from fplopt.config import REPORTS_DIR, SETTINGS, TEST_SEASON  # noqa: E402
from fplopt.features import feature_columns  # noqa: E402
from fplopt.model import SingleModel  # noqa: E402
from fplopt.optimise import realised_points, solve_team  # noqa: E402

log = logging.getLogger("tabicl_challenger")
KEYS = ["season", "GW", "key"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw-from", type=int, default=20)
    ap.add_argument("--context", type=int, default=4000, help="most recent training rows given to TabICL")
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=40.0, help="stop after this many minutes (time box)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from tabicl import TabICLRegressor

    F = pd.read_pickle(ROOT / "data" / "processed" / "features.pkl")
    pred = pd.read_csv(REPORTS_DIR / "predictions_2025-26.csv.gz")
    cols = feature_columns(F)
    test_idx = int(F.loc[F["season"] == TEST_SEASON, "season_idx"].iloc[0])
    t_start, out = time.time(), []
    for b in range(a.gw_from, 39, SETTINGS.retrain_every):
        if (time.time() - t_start) / 60 > a.max_minutes:
            log.info("time box reached; stopping before GW%d", b)
            break
        block = F[(F["season"] == TEST_SEASON) & (F["GW"] >= b) & (F["GW"] < b + SETTINGS.retrain_every)]
        train = F[(F["season_idx"] < test_idx) | ((F["season"] == TEST_SEASON) & (F["GW"] < b))]
        ctx = train.sort_values(["season_idx", "GW"]).tail(a.context)
        use = [c for c in cols if ctx[c].nunique(dropna=True) > 1]
        fill = ctx[use].median(numeric_only=True)
        t0 = time.time()
        reg = TabICLRegressor(n_estimators=a.n_estimators, random_state=SETTINGS.seed)
        reg.fit(ctx[use].fillna(fill).to_numpy(np.float32), ctx["total_points"].to_numpy(np.float32))
        mu_icl = reg.predict(block[use].fillna(fill).to_numpy(np.float32))
        gbm = SingleModel(use).fit(ctx)
        r = block[KEYS].copy()
        r["mu_tabicl"] = mu_icl
        r["mu_gbm_same_rows"] = gbm.predict(block)
        out.append(r)
        log.info("block GW%d-%d: context %d rows, %d features, TabICL %.0fs", b, min(b + SETTINGS.retrain_every - 1, 38),
                 len(ctx), len(use), time.time() - t0)
    if not out:
        raise SystemExit("no block finished inside the time box")
    P = pred.merge(pd.concat(out), on=KEYS, how="inner")
    forecasts = {"Two-stage model (all rows)": "mu", "Single GBM (all rows)": "mu_single",
                 f"TabICL ({a.context} recent rows)": "mu_tabicl", f"GBM, same {a.context} rows": "mu_gbm_same_rows"}
    gw_rows = []
    for gw, c in P.groupby("GW"):
        c = c.reset_index(drop=True)
        rec = {"GW": gw}
        for name, col in forecasts.items():
            team = solve_team(c, c[col].to_numpy(), SETTINGS.budget_tenths, SETTINGS.club_cap, SETTINGS.captain)
            rec[name] = realised_points(c, team)
        gw_rows.append(rec)
    G = pd.DataFrame(gw_rows)
    y = P["total_points"].to_numpy(float)
    summary = []
    for name, col in forecasts.items():
        e = P[col].to_numpy(float) - y
        row = {"forecast": name, "gameweeks": len(G), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
               "xi_pts_per_gw": float(G[name].mean())}
        if col != "mu":
            m, lo, hi = paired_bootstrap(G["Two-stage model (all rows)"], G[name])
            row.update(model_minus_this=m, ci95_low=lo, ci95_high=hi)
        summary.append(row)
    S = pd.DataFrame(summary)
    G.to_csv(REPORTS_DIR / "challenger_tabicl_gw.csv", index=False)
    S.to_csv(REPORTS_DIR / "challenger_tabicl.csv", index=False)
    print(S.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
