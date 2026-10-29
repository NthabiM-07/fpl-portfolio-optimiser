"""Interactive demo: `streamlit run src/fplopt/app.py` (or `python main.py app`).

Runs offline from the files in reports/ written by `python main.py backtest`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from fplopt.baselines import baseline_scores  # noqa: E402
from fplopt.config import REPORTS_DIR, SETTINGS  # noqa: E402
from fplopt.monitor import ewma_chart, weekly_error_by_position  # noqa: E402
from fplopt.optimise import InfeasibleTeamError, realised_points, solve_team  # noqa: E402

st.set_page_config(page_title="FPL portfolio optimiser", layout="wide")
GREEN, RED, GREY, BLUE = "#1E4D3A", "#B5473A", "#9AA19D", "#4C72B0"
NAMES = {"model": "Two-stage model", "model_single": "Single GBM", "mean_last5": "Mean of last 5",
         "last_week": "Last week", "top_price": "Most expensive", "random": "Random valid", "oracle": "Hindsight oracle"}


@st.cache_data
def load():
    pred = pd.read_csv(REPORTS_DIR / "predictions_2025-26.csv.gz")
    gw = pd.read_csv(REPORTS_DIR / "gw_results.csv")
    rob = pd.read_csv(REPORTS_DIR / "robust_results.csv")
    tabs = {n: pd.read_csv(REPORTS_DIR / f"{n}.csv") for n in
            ["strategies_all_gws", "strategies_gw6_38", "significance_all_gws", "forecast_accuracy", "interval_coverage", "sensitivity"]}
    for n in ["forecast_dm", "xp_audit", "xp_partial", "signal_value", "cqr_blocks"]:          # optional extras
        f = REPORTS_DIR / f"{n}.csv"
        tabs[n] = pd.read_csv(f) if f.exists() else None
    return pred, gw, rob, tabs


if not (REPORTS_DIR / "predictions_2025-26.csv.gz").exists():
    st.error("No results yet. Run `python main.py backtest` first.")
    st.stop()
pred, gwres, rob, tabs = load()

st.title("FPL team selection as constrained portfolio optimisation")
st.caption("Two-stage expected-points model (will he play? x how much if he does?) feeding an exact integer programme. "
           "Walk-forward backtest on every 2025-26 gameweek; every strategy uses the same optimiser and rules.")

s = tabs["strategies_all_gws"].set_index("strategy")
sig = tabs["significance_all_gws"].set_index("model_vs")
k = st.columns(4)
k[0].metric("Model, mean XI points / GW", f"{s.loc['model', 'mean_pts_per_gw']:.1f}")
k[1].metric("vs last week's points", f"{s.loc['model', 'uplift_vs_last_week_%']:+.1f}%",
            f"95% CI {sig.loc['last_week', 'ci95_low']:+.1f} to {sig.loc['last_week', 'ci95_high']:+.1f} pts/GW")
k[2].metric("vs mean of last 5", f"{s.loc['model', 'uplift_vs_mean_last5_%']:+.1f}%",
            f"95% CI {sig.loc['mean_last5', 'ci95_low']:+.1f} to {sig.loc['mean_last5', 'ci95_high']:+.1f} pts/GW")
k[3].metric("vs most expensive XI", f"{s.loc['model', 'uplift_vs_top_price_%']:+.1f}%")

with st.sidebar:
    st.header("Pick a team")
    gw = st.slider("Gameweek (2025-26)", int(pred["GW"].min()), int(pred["GW"].max()), 20)
    budget = st.slider("XI budget (GBP m)", 70.0, 100.0, SETTINGS.budget_tenths / 10, 0.5)
    gamma = st.select_slider("Risk dial Γ (players allowed to hit their downside)", options=list(range(0, 9)), value=0)
    show_actual = st.checkbox("Show what actually happened", value=True)
    c = pred[pred["GW"] == gw].reset_index(drop=True)
    labels = (c["name"] + " (" + c["position"] + ", " + c["team"] + ")").tolist()
    lock = st.multiselect("Must include", labels)
    excl = st.multiselect("Exclude", labels)

tab_pick, tab_bt, tab_risk, tab_model, tab_sens = st.tabs(["Team", "Backtest", "Risk dial", "Model quality", "Sensitivity"])

with tab_pick:
    d = (c["mu"] - c["lo"]).clip(lower=0).to_numpy()
    try:
        team = solve_team(c, c["mu"].to_numpy(), int(round(budget * 10)), SETTINGS.club_cap, True, d=d, gamma=gamma, thetas=16,
                          lock=[labels.index(x) for x in lock], exclude=[labels.index(x) for x in excl])
    except InfeasibleTeamError as e:
        st.error(f"No valid team: {e}. Loosen the budget, locks or exclusions.")
        st.stop()
    xi = c.iloc[team.idx].copy()
    xi["C"] = np.where(xi.index == team.captain, "C", "")
    xi["price"] = xi["value"] / 10
    xi["pts per GBP m"] = xi["mu"] / xi["price"]
    xi["80% interval"] = xi.apply(lambda r: f"{max(r.lo, 0):.1f} to {r.hi:.1f}", axis=1)
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi = xi.sort_values(["position", "mu"], key=lambda col: col.map(order) if col.name == "position" else -col)
    cols = ["C", "name", "position", "team", "price", "mu", "80% interval", "p_play60", "pts per GBP m"] + (["total_points"] if show_actual else [])
    st.subheader(f"Gameweek {gw}: optimal XI at GBP {budget:.1f}m, Γ = {gamma}")
    st.dataframe(xi[cols].rename(columns={"mu": "expected pts", "p_play60": "P(60+ min)", "total_points": "actual pts"}).round(2),
                 hide_index=True, use_container_width=True, height=36 * 12 + 4)
    m = st.columns(4)
    m[0].metric("Expected points (captain doubled)", f"{c['mu'].to_numpy()[team.idx].sum() + c['mu'].to_numpy()[team.captain]:.1f}")
    m[1].metric("Cost", f"GBP {xi['price'].sum():.1f}m")
    if show_actual:
        m[2].metric("Actual points", f"{realised_points(c, team):.0f}")
        base = {n: realised_points(c, solve_team(c, sc, int(round(budget * 10)), SETTINGS.club_cap, True))
                for n, sc in baseline_scores(c).items()}
        m[3].metric("Best baseline this GW", f"{max(base.values()):.0f}", max(base, key=base.get).replace("_", " "))

    with st.expander("Why these players? (points lost if each one is removed and the team re-optimised)"):
        if st.button("Explain picks"):
            rows = []
            for i in team.idx:
                try:
                    t2 = solve_team(c, c["mu"].to_numpy(), int(round(budget * 10)), SETTINGS.club_cap, True, d=d, gamma=gamma, thetas=16,
                                    lock=[labels.index(x) for x in lock if labels.index(x) != i],
                                    exclude=[labels.index(x) for x in excl] + [i])
                    rows.append({"player": c.at[i, "name"], "expected pts lost without him": team.objective - t2.objective})
                except InfeasibleTeamError:
                    rows.append({"player": c.at[i, "name"], "expected pts lost without him": np.nan})
            st.dataframe(pd.DataFrame(rows).sort_values("expected pts lost without him", ascending=False).round(2), hide_index=True)
            st.caption("This is the player's marginal value to the portfolio, not just his own score: a cheap, reliable "
                       "pick can matter more than a star because he frees budget for others.")

with tab_bt:
    long = gwres.melt(id_vars="GW", value_vars=["model", "mean_last5", "last_week", "top_price", "random"], var_name="strategy", value_name="points")
    long["cumulative"] = long.groupby("strategy")["points"].cumsum()
    long["strategy"] = long["strategy"].map(NAMES)
    fig = px.line(long, x="GW", y="cumulative", color="strategy",
                  color_discrete_map={NAMES["model"]: GREEN, NAMES["mean_last5"]: BLUE, NAMES["last_week"]: RED,
                                      NAMES["top_price"]: "#8C6D31", NAMES["random"]: GREY},
                  title="Cumulative realised XI points, 2025-26")
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    a.markdown("**Mean points per gameweek (all 38 GWs)**")
    a.dataframe(tabs["strategies_all_gws"].assign(strategy=lambda t: t.strategy.map(NAMES)).round(1), hide_index=True)
    b.markdown("**Model minus baseline, paired bootstrap 95% CI (points per GW)**")
    b.dataframe(tabs["significance_all_gws"].assign(model_vs=lambda t: t.model_vs.map(NAMES)).round(2), hide_index=True)
    st.markdown("**Gameweeks 6-38 only** (pre-declared secondary view)")
    st.dataframe(tabs["strategies_gw6_38"].assign(strategy=lambda t: t.strategy.map(NAMES)).round(1), hide_index=True)

with tab_risk:
    fr = rob.groupby("gamma")["realised"].agg(mean="mean", p10=lambda x: x.quantile(0.1), worst="min").reset_index()
    b0 = fr.set_index("gamma").loc[0]
    fr["change in 10th-percentile week"] = fr["p10"] - b0["p10"]
    fr["change in mean week"] = fr["mean"] - b0["mean"]
    fig = go.Figure()
    fig.add_bar(x=[f"\u0393={g}" for g in fr["gamma"]], y=fr["change in 10th-percentile week"], name="10th-percentile week (downside)",
                marker_color=GREEN, text=fr["change in 10th-percentile week"].round(1), textposition="outside")
    fig.add_bar(x=[f"\u0393={g}" for g in fr["gamma"]], y=fr["change in mean week"], name="Mean week",
                marker_color=GREY, text=fr["change in mean week"].round(1), textposition="outside")
    fig.update_layout(barmode="group", title="Robust dial vs \u0393=0: downside protection bought per point of mean given up",
                      yaxis_title="Points per gameweek vs \u0393=0")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fr.round(1), hide_index=True)
    st.caption("Γ = how many players may fall to the bottom of their 80% interval at once. The MILP protects against the worst Γ of them. "
               "Robust picks are solved on the top 60 candidates per position by expected points, on a 32-point θ grid. "
               "The effect is not monotone in Γ and 38 weeks is a small sample: read it as suggestive, not proven.")

with tab_model:
    st.markdown("**Forecast accuracy, 2025-26 (out of sample, walk-forward).** Buckets follow OpenFPL (Groos 2025): "
                "zeros = did not play, blanks ≤ 2, tickers 3-4, haulers ≥ 5.")
    st.dataframe(tabs["forecast_accuracy"].round(3), hide_index=True)
    if tabs["forecast_dm"] is not None:
        st.markdown("**Is the forecast error significantly lower? Diebold-Mariano test across the 38 gameweeks** "
                    "(losses averaged within each gameweek; Newey-West variance; HLN correction)")
        fmt_p = lambda v: "<0.0001" if v < 1e-4 else f"{v:.4f}"
        st.dataframe(tabs["forecast_dm"].assign(p_se=lambda t: t.p_se.map(fmt_p), p_ae=lambda t: t.p_ae.map(fmt_p)).round(2),
                     hide_index=True)
        st.caption("Against the single GBM the gain is not significant on squared error (significant on absolute error), yet "
                   "its *decisions* are clearly worse (Backtest tab). Forecast accuracy and decision quality are different things.")
    if tabs["signal_value"] is not None:
        st.markdown("**How much do forecasts and decisions rely on each signal family?** Each family is shuffled within each "
                    "gameweek (10 times, frozen model); the effect is measured on forecast error and on the realised points of "
                    "the team the optimiser then picks. Families overlap, so a low score can mean 'replaceable', not 'useless'.")
        sv = tabs["signal_value"].sort_values("xi_points_lost_per_gw")
        fig = go.Figure()
        fig.add_bar(y=sv["family"], x=sv["xi_points_lost_per_gw"], orientation="h", name="XI points lost per GW",
                    marker_color=GREEN, error_x=dict(type="data", array=sv["xi_points_lost_sd"]))
        fig.update_layout(title="XI points lost per gameweek when the family is scrambled (GW 20-38, ±1 sd over 10 shuffles)",
                          xaxis_title="Points per gameweek", height=380, margin=dict(l=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tabs["signal_value"].round(3), hide_index=True)
    if tabs["xp_audit"] is not None:
        with st.expander("Leakage forensics: why FPL's own xP column is excluded"):
            st.dataframe(tabs["xp_audit"].round(3), hide_index=True)
            if tabs["xp_partial"] is not None:
                st.dataframe(tabs["xp_partial"].round(3), hide_index=True)
            st.caption("On the gameweeks where xP is filled in, it beats every pre-match forecast on this week's points but keeps only "
                       "about two-thirds of that edge a week later, when it is worse than the model. After removing what the model's "
                       "pre-match forecast explains, xP still tracks this week's goals and bonus, but not next week's. That strongly "
                       "suggests post-match information, as the dataset warns.")
    st.markdown("**80% prediction interval coverage.** Mondrian CQR: likely starters (P(60+ min) ≥ 0.5, known before the "
                "deadline) and others are calibrated separately, from the last 8 gameweeks of out-of-sample errors.")
    st.dataframe(tabs["interval_coverage"].round(3), hide_index=True)
    if tabs["cqr_blocks"] is not None:
        with st.expander("Conformal correction used in each retraining block"):
            st.dataframe(tabs["cqr_blocks"].round(3), hide_index=True)
    err = weekly_error_by_position(pred)
    pos = st.radio("Position", list(err.columns), horizontal=True, index=list(err.columns).index("DEF") if "DEF" in err else 0)
    e = ewma_chart(err[pos].to_numpy())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=err.index, y=e["error"], name="weekly mean error", line=dict(color=GREY)))
    fig.add_trace(go.Scatter(x=err.index, y=e["ewma"], name="EWMA (λ=0.2)", line=dict(color=GREEN, width=3)))
    for lim in ("ucl", "lcl"):
        fig.add_hline(y=float(e[lim].iat[0]), line_dash="dash", line_color=RED)
    fig.update_layout(title=f"EWMA control chart of prediction error: {pos}", xaxis_title="Gameweek", yaxis_title="Actual − predicted points")
    st.plotly_chart(fig, use_container_width=True)

with tab_sens:
    st.dataframe(tabs["sensitivity"].round(3), hide_index=True)
    fig = px.bar(gwres, x="GW", y="club_cap_cost_pred", title="Expected points the 3-per-club rule costs each gameweek",
                 color_discrete_sequence=[GREEN])
    st.plotly_chart(fig, use_container_width=True)
    fig = px.bar(gwres, x="GW", y="budget_shadow_pts_per_m", title="Budget shadow price: expected points per extra GBP 1m (LP relaxation)",
                 color_discrete_sequence=[BLUE])
    st.plotly_chart(fig, use_container_width=True)
