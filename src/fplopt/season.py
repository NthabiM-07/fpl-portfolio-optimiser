"""Season backtest for the 15-man squad with transfers (`python main.py season`, about 3 minutes).

Design, fixed before looking at the second half of the season:
1. Tune on gameweeks 1-19: try planning horizons H in {1, 3} and transfer hurdles in {4, 6, 8, 12}
   points per extra transfer (FPL deducts 4; the hurdle is what the planner *charges itself*).
2. Pick the setting with the most points over gameweeks 1-19 (ties: fewer transfers, then the
   shorter horizon; nothing from gameweeks 20-38).
3. Judge every strategy on gameweeks 20-38, which played no part in the choice. The naive
   forecast (mean of the last 5) gets its own tuning on gameweeks 1-19, so the comparison is fair.
4. Because one season is one path (an early transfer changes every later squad), also compare the
   two forecasts like for like: same horizon and hurdle, all eight settings (`season_matched`),
   and report how well the first half's ranking of settings predicts the second half's.
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


def _grid(pred: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, dict, tuple[int, int]]:
    tuning, runs = [], {}
    for h in HORIZONS:
        for hc in HURDLES:
            r = simulate_season(pred, score_col, PlanConfig(horizon=h, hit_cost=hc))
            runs[(h, hc)] = r
            tuning.append({"forecast": score_col, "horizon": h, "hurdle": hc,
                           "pts_gw1_19": r.loc[r.GW < SPLIT_GW, "points"].sum(),
                           "pts_gw20_38": r.loc[r.GW >= SPLIT_GW, "points"].sum(), "pts_season": r["points"].sum(),
                           "transfers": int(r["transfers"].sum()), "hit_points": r["hits"].sum()})
            log.info("%s H=%d hurdle=%d: %s", score_col, h, hc, tuning[-1])
    tuning = pd.DataFrame(tuning)
    best = tuning.sort_values(["pts_gw1_19", "transfers", "horizon"], ascending=[False, True, True]).iloc[0]
    sel = (int(best["horizon"]), int(best["hurdle"]))
    tuning["selected"] = (tuning["horizon"] == sel[0]) & (tuning["hurdle"] == sel[1])
    return tuning, runs, sel


def run(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tuning, runs, (bh, bc) = _grid(pred, "mu")
    tuning_naive, runs_naive, (nh, nc) = _grid(pred, "total_points_r5")
    final = {
        "Model, selected setting": runs[(bh, bc)],
        f"Model, H={4 - bh}, hurdle {bc} (other horizon)": runs[(4 - bh, bc)],
        "Model, hurdle = real cost (4), H=3": runs[(3, 4)],
        "Model, hurdle = real cost (4), H=1": runs[(1, 4)],
        "Model, keep the GW1 squad (no transfers)": simulate_season(pred, "mu", PlanConfig(horizon=bh, hit_cost=bc), hold=True),
        f"Mean of last 5, its own selected setting (H={nh}, hurdle {nc})": runs_naive[(nh, nc)],
    }
    tuning = pd.concat([tuning, tuning_naive], ignore_index=True)
    gw = pd.DataFrame({"GW": runs[(bh, bc)]["GW"]})
    rows = []
    ref = "Model, selected setting"
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
    summary.insert(1, "setting", [f"H={bh}, hurdle {bc}"] + [""] * (len(summary) - 1))
    return {"season_tuning": tuning, "season_gw": gw, "season_summary": summary,
            "season_selected_detail": runs[(bh, bc)], "season_matched": matched_table(tuning)}


def matched_table(tuning: pd.DataFrame) -> pd.DataFrame:
    """Model vs mean-of-last-5 forecasts at the same horizon and hurdle, plus the stability of the
    tuning: Spearman correlation between the gameweek 1-19 and 20-38 rankings of the settings."""
    a = tuning[tuning["forecast"] == "mu"].set_index(["horizon", "hurdle"])
    z = tuning[tuning["forecast"] == "total_points_r5"].set_index(["horizon", "hurdle"])
    m = pd.DataFrame({"model_season": a["pts_season"], "naive_season": z["pts_season"],
                      "model_gw20_38": a["pts_gw20_38"], "naive_gw20_38": z["pts_gw20_38"],
                      "model_hit_points": a["hit_points"], "naive_hit_points": z["hit_points"]}).reset_index()
    m["model_minus_naive_season"] = m["model_season"] - m["naive_season"]
    m["model_minus_naive_gw20_38"] = m["model_gw20_38"] - m["naive_gw20_38"]
    m.attrs["rank_corr_model"] = float(a["pts_gw1_19"].corr(a["pts_gw20_38"], method="spearman"))
    m.attrs["rank_corr_naive"] = float(z["pts_gw1_19"].corr(z["pts_gw20_38"], method="spearman"))
    return m


def write(tabs: dict[str, pd.DataFrame]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, t in tabs.items():
        t.to_csv(REPORTS_DIR / f"{name}.csv", index=False)
    gw = tabs["season_gw"]
    colors = ["#1E4D3A", "#6E9B4E", "#B5473A", "#4C72B0", "#8C6D31", "#9AA19D"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = [c for c in gw.columns if c != "GW"]
    for col, color in zip(names, colors, strict=False):
        ax.plot(gw["GW"], gw[col].cumsum(), color=color, lw=2.4 if col == "Model, selected setting" else 1.5,
                label=f"{col} ({gw[col].sum():.0f})")
    ax.axvline(SPLIT_GW - 0.5, color="#3A403C", ls=":", lw=1)
    ax.text(SPLIT_GW - 0.3, ax.get_ylim()[1] * 0.04, "settings chosen on GW 1-19 | judged on GW 20-38", fontsize=8, color="#3A403C")
    ax.set_title("Full FPL rules, 2025-26: 15-man squad, GBP 100m, 1 free transfer a week, -4 per extra",
                 loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative points (after hits)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E6E1")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "season_cumulative.png", dpi=150)
    plt.close(fig)
    log.info("season reports written; selected H=%s hurdle=%s", tabs["season_summary"].attrs.get("horizon"),
             tabs["season_summary"].attrs.get("hurdle"))


def headline(tabs: dict[str, pd.DataFrame]) -> str:
    s = tabs["season_summary"]
    m = tabs["season_matched"]
    wins_season = int((m["model_minus_naive_season"] > 0).sum())
    wins_second = int((m["model_minus_naive_gw20_38"] > 0).sum())
    return (s[["strategy", "pts_gw20_38", "pts_season", "transfers", "hit_points"]].round(1).to_string(index=False)
            + f"\n\nSame settings, model vs mean of last 5: model ahead in {wins_season}/{len(m)} on the season, "
              f"{wins_second}/{len(m)} on GW20-38 (mean gap {m['model_minus_naive_season'].mean():.0f} season points)."
              f"\nRank correlation of settings, GW1-19 vs GW20-38: model {m.attrs['rank_corr_model']:.2f}, "
              f"naive {m.attrs['rank_corr_naive']:.2f}")

