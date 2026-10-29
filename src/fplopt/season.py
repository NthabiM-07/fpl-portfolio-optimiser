"""Season backtest for the 15-man squad with transfers (`python main.py season`, about 4 minutes).

Design, fixed before looking at the second half of the season:
1. Tune on gameweeks 1-19: try planning horizons H in {1, 3} and transfer hurdles in {4, 6, 8, 12}
   points per extra transfer (FPL deducts 4; the hurdle is what the planner *charges itself*).
2. Pick the setting with the most points over gameweeks 1-19.
3. Judge every strategy on gameweeks 20-38, which played no part in the choice.
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import paired_bootstrap  # noqa: E402
from .config import FIGURES_DIR, REPORTS_DIR  # noqa: E402
from .planner import PlanConfig, simulate_season  # noqa: E402

log = logging.getLogger(__name__)
HORIZONS, HURDLES, SPLIT_GW = (1, 3), (4, 6, 8, 12), 20


def run(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tuning, runs = [], {}
    for h in HORIZONS:
        for hc in HURDLES:
            r = simulate_season(pred, "mu", PlanConfig(horizon=h, hit_cost=hc))
            runs[(h, hc)] = r
            tuning.append({"horizon": h, "hurdle": hc, "pts_gw1_19": r.loc[r.GW < SPLIT_GW, "points"].sum(),
                           "pts_gw20_38": r.loc[r.GW >= SPLIT_GW, "points"].sum(), "pts_season": r["points"].sum(),
                           "transfers": int(r["transfers"].sum()), "hit_points": r["hits"].sum()})
            log.info("H=%d hurdle=%d: %s", h, hc, tuning[-1])
    tuning = pd.DataFrame(tuning)
    best = tuning.sort_values(["pts_gw1_19", "pts_gw20_38"], ascending=[False, True]).iloc[0]   # ties broken pessimistically
    bh, bc = int(best["horizon"]), int(best["hurdle"])
    tuning["selected"] = (tuning["horizon"] == bh) & (tuning["hurdle"] == bc)
    final = {
        "Model, plan ahead (selected setting)": runs[(bh, bc)],
        "Model, plan ahead, hurdle = real cost (4)": runs[(3, 4)],
        "Model, one week at a time, hurdle 4": runs[(1, 4)],
        "Model, keep the GW1 squad (no transfers)": simulate_season(pred, "mu", PlanConfig(horizon=bh, hit_cost=bc), hold=True),
        "Mean of last 5, plan ahead (selected setting)": simulate_season(pred, "total_points_r5", PlanConfig(horizon=bh, hit_cost=bc)),
    }
    gw = pd.DataFrame({"GW": runs[(bh, bc)]["GW"]})
    rows = []
    ref = "Model, plan ahead (selected setting)"
    for name, r in final.items():
        gw[name] = r["points"].to_numpy()
        second = r["GW"] >= SPLIT_GW
        row = {"strategy": name, "pts_gw20_38": r.loc[second, "points"].sum(), "pts_season": r["points"].sum(),
               "mean_per_gw": r["points"].mean(), "transfers": int(r["transfers"].sum()), "hit_points": r["hits"].sum(),
               "autosubs": int(r["autosubs"].sum())}
        if name != ref:
            m, lo, hi = paired_bootstrap(final[ref].loc[second, "points"], r.loc[second, "points"])
            row.update(selected_minus_this_gw20_38=m, ci95_low=lo, ci95_high=hi)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.attrs.update(horizon=bh, hurdle=bc)
    return {"season_tuning": tuning, "season_gw": gw, "season_summary": summary,
            "season_selected_detail": runs[(bh, bc)]}


def write(tabs: dict[str, pd.DataFrame]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, t in tabs.items():
        t.to_csv(REPORTS_DIR / f"{name}.csv", index=False)
    gw = tabs["season_gw"]
    colors = ["#1E4D3A", "#B5473A", "#4C72B0", "#8C6D31", "#9AA19D"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = [c for c in gw.columns if c != "GW"]
    for col, color in zip(names, colors):
        ax.plot(gw["GW"], gw[col].cumsum(), color=color, lw=2.4 if col.endswith("(selected setting)") and col.startswith("Model") else 1.5,
                label=f"{col} ({gw[col].sum():.0f})")
    ax.axvline(SPLIT_GW - 0.5, color="#3A403C", ls=":", lw=1)
    ax.text(SPLIT_GW - 0.3, ax.get_ylim()[1] * 0.04, "settings chosen on GW 1-19 | judged on GW 20-38", fontsize=8, color="#3A403C")
    ax.set_title("Full FPL rules, 2025-26: 15-man squad, GBP 100m, 1 free transfer a week, -4 per extra",
                 loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("Gameweek"); ax.set_ylabel("Cumulative points (after hits)")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E3E6E1")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "season_cumulative.png", dpi=150); plt.close(fig)
    log.info("season reports written; selected H=%s hurdle=%s", tabs["season_summary"].attrs.get("horizon"),
             tabs["season_summary"].attrs.get("hurdle"))


def headline(tabs: dict[str, pd.DataFrame]) -> str:
    s = tabs["season_summary"]
    return s[["strategy", "pts_gw20_38", "pts_season", "transfers", "hit_points"]].round(1).to_string(index=False)

