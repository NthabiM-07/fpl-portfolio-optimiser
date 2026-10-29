"""Write result tables (CSV + Markdown) and figures for the README and the demo."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import forecast_table, interval_table, significance_table, strategy_table  # noqa: E402
from .config import FIGURES_DIR, REPORTS_DIR  # noqa: E402
from .monitor import ewma_chart, weekly_error_by_position  # noqa: E402

COLORS = {"model": "#1E4D3A", "model_single": "#6E9B4E", "mean_last5": "#4C72B0", "last_week": "#B5473A",
          "top_price": "#8C6D31", "random": "#9AA19D", "oracle": "#C9CCC8"}
LABELS = {"model": "Two-stage model + MILP", "model_single": "Single GBM + MILP", "mean_last5": "Mean of last 5",
          "last_week": "Last week's points", "top_price": "Most expensive XI", "random": "Random valid XI", "oracle": "Hindsight oracle"}


def to_markdown(df: pd.DataFrame) -> str:
    """Small Markdown table writer (avoids an extra dependency)."""
    cols = list(df.columns)
    fmt = lambda v: f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E6E1", linewidth=0.8)


def write_all(pred: pd.DataFrame, gw: pd.DataFrame, rob: pd.DataFrame) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["season", "GW", "key", "name", "position", "team", "value", "total_points", "minutes", "n_fixtures",
            "total_points_r1", "total_points_r5", "p_play0", "p_play1_59", "p_play60", "pts_if_1_59", "pts_if_60",
            "mu", "mu_single", "q_lo_raw", "q_hi_raw", "cqr_q", "lo", "hi"]
    pred[keep].to_csv(REPORTS_DIR / "predictions_2025-26.csv.gz", index=False, compression="gzip")
    gw.to_csv(REPORTS_DIR / "gw_results.csv", index=False)
    rob.to_csv(REPORTS_DIR / "robust_results.csv", index=False)

    tabs = {
        "strategies_all_gws": strategy_table(gw, 1),
        "strategies_gw6_38": strategy_table(gw, 6),
        "significance_all_gws": significance_table(gw, 1),
        "forecast_accuracy": forecast_table(pred),
        "interval_coverage": interval_table(pred).reset_index().rename(columns={"index": "group"}),
    }
    rsum = (rob.groupby("gamma")["realised"].agg(mean="mean", p10=lambda s: s.quantile(0.1), worst="min", sd="std")
               .reset_index())
    tabs["robust_frontier"] = rsum
    sens = pd.DataFrame({
        "metric": ["budget shadow price, predicted pts per extra GBP 1m (median)", "gameweeks where budget binds",
                   "club-cap cost, predicted pts (mean)", "gameweeks where club cap binds", "mean XI cost (GBP m)"],
        "value": [gw["budget_shadow_pts_per_m"].median(), int((gw["budget_shadow_pts_per_m"] > 1e-6).sum()),
                  gw["club_cap_cost_pred"].mean(), int((gw["club_cap_cost_pred"] > 1e-6).sum()), gw["model_cost"].mean()],
    })
    tabs["sensitivity"] = sens
    err = weekly_error_by_position(pred)
    ew = {p: ewma_chart(err[p].to_numpy()) for p in err.columns}
    tabs["ewma_signals"] = pd.DataFrame({"position": list(ew), "first_signal_gw": [
        (int(err.index[np.flatnonzero(e["signal"].to_numpy())[0]]) if e["signal"].any() else None) for e in ew.values()],
        "mean_error": [float(err[p].mean()) for p in ew]})

    md = []
    for name, t in tabs.items():
        t.to_csv(REPORTS_DIR / f"{name}.csv", index=False)
        md.append(f"### {name}\n\n" + to_markdown(t) + "\n")
    (REPORTS_DIR / "results.md").write_text("\n".join(md), encoding="utf-8")

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(9, 4.8))
    lines = ["model", "mean_last5", "last_week", "top_price", "random"]
    ends = sorted(((gw[st].sum(), st) for st in lines), reverse=True)
    min_gap = 0.045 * max(e for e, _ in ends)          # keep end labels at least ~4.5% of the range apart
    label_y, last = {}, None
    for e, st in ends:
        y = e if last is None else min(e, last - min_gap)
        label_y[st], last = y, y
    for st in lines:
        ax.plot(gw["GW"], gw[st].cumsum(), color=COLORS[st], lw=2.4 if st == "model" else 1.5, label=LABELS[st])
        ax.annotate(f"{LABELS[st]} ({gw[st].sum():.0f})", (gw["GW"].iloc[-1], label_y[st]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=COLORS[st], va="center")
    _style(ax, "Cumulative XI points, 2025-26 (same optimiser and rules for every strategy)", "Gameweek", "Cumulative points")
    ax.set_xlim(1, gw["GW"].max() + 9)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "cumulative_points.png", dpi=150); plt.close(fig)

    st = tabs["strategies_all_gws"].set_index("strategy").drop(index="oracle")
    sig = tabs["significance_all_gws"].set_index("model_vs")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = ["model", "model_single", "mean_last5", "last_week", "top_price", "random"]
    vals = st.loc[order, "mean_pts_per_gw"]
    ax.barh([LABELS[o] for o in order][::-1], vals[::-1], color=[COLORS[o] for o in order][::-1])
    for i, (o, v) in enumerate(zip(order[::-1], vals[::-1])):
        txt = f"{v:.1f}"
        if o in sig.index:
            r = sig.loc[o]
            txt += f"   (model +{r.mean_diff_pts_per_gw:.1f}, 95% CI {r.ci95_low:+.1f} to {r.ci95_high:+.1f})"
        ax.text(v + 0.5, i, txt, va="center", fontsize=8.5)
    _style(ax, "Mean realised XI points per gameweek, 2025-26 (all 38 GWs, captain doubled)", "Points per gameweek", "")
    ax.grid(axis="x", color="#E3E6E1"); ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlim(0, vals.max() * 1.75)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "mean_points.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    base = rsum.set_index("gamma").loc[0]
    x = np.arange(len(rsum))
    dm, dp = rsum["mean"] - base["mean"], rsum["p10"] - base["p10"]
    ax.bar(x - 0.2, dp, 0.4, color=COLORS["model"], label="Change in 10th-percentile week (downside)")
    ax.bar(x + 0.2, dm, 0.4, color=COLORS["random"], label="Change in mean week")
    for xi, a, b in zip(x, dp, dm):
        ax.text(xi - 0.2, a + (0.1 if a >= 0 else -0.1), f"{a:+.1f}", ha="center", va="bottom" if a >= 0 else "top", fontsize=8)
        ax.text(xi + 0.2, b + (0.1 if b >= 0 else -0.1), f"{b:+.1f}", ha="center", va="bottom" if b >= 0 else "top", fontsize=8)
    ax.axhline(0, color="#3A403C", lw=0.8)
    ax.set_ylim(min(dm.min(), dp.min(), 0) - 0.7, max(dm.max(), dp.max(), 0) + 0.7)
    ax.set_xticks(x, [f"Γ={int(g)}" for g in rsum["gamma"]])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "Robust dial vs Γ=0: downside protection bought per point of mean given up", "Budget of uncertainty Γ",
           "Points per gameweek vs Γ=0")
    ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "robust_frontier.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharex=True)
    for ax, p in zip(axes.ravel(), ["GK", "DEF", "MID", "FWD"]):
        e = ew[p]
        ax.plot(err.index, e["error"], color="#C9CCC8", lw=1, label="weekly mean error")
        ax.plot(err.index, e["ewma"], color=COLORS["model"], lw=2, label="EWMA (λ=0.2)")
        ax.axhline(e["ucl"].iat[0], color=COLORS["last_week"], ls="--", lw=1)
        ax.axhline(e["lcl"].iat[0], color=COLORS["last_week"], ls="--", lw=1)
        sig_gw = err.index[e["signal"].to_numpy()]
        ax.scatter(sig_gw, e.loc[e["signal"], "ewma"], color=COLORS["last_week"], zorder=3, s=14)
        _style(ax, p, "Gameweek" if p in ("MID", "FWD") else "", "Actual − predicted")
    fig.suptitle("Model monitoring: EWMA control chart of weekly prediction error, 2025-26", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "ewma_monitor.png", dpi=150); plt.close(fig)

    summary = {k: v.round(4).to_dict(orient="records") for k, v in tabs.items()}
    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return tabs
