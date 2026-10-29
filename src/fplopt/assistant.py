"""Tools an AI assistant can call (exposed over MCP by `mcp_server.py`).

Design rule: the language model only interprets the request and chooses a tool. Every number it
reports comes from the saved walk-forward forecasts and an exact MILP solve, so it cannot invent
an expected-points figure or a team that breaks the rules. The same pattern fits a credit copilot:
the LLM is the interface, the governed model and the optimiser are the engine.

Everything here is a plain function returning JSON-ready dicts, so it is unit-tested without MCP.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, SETTINGS
from .optimise import InfeasibleTeamError, Team, realised_points, solve_team

ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


@lru_cache(maxsize=1)
def _predictions() -> pd.DataFrame:
    path = REPORTS_DIR / "predictions_2025-26.csv.gz"
    if not path.exists():
        raise FileNotFoundError("No saved forecasts. Run `python main.py backtest` first.")
    return pd.read_csv(path)


def _gameweek(gameweek: int) -> pd.DataFrame:
    p = _predictions()
    c = p[p["GW"] == gameweek].reset_index(drop=True)
    if c.empty:
        raise ValueError(f"Gameweek {gameweek} is not in the 2025-26 backtest (valid: {int(p.GW.min())}-{int(p.GW.max())}).")
    return c


# Letters that Unicode NFKD does not split into ASCII + accent (so "Odegaard" finds "Ødegaard").
_SPECIAL = str.maketrans({"ø": "o", "Ø": "O", "ß": "ss", "ı": "i", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
                          "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "þ": "th", "ð": "d"})


def _fold(s: str) -> str:
    s = s.translate(_SPECIAL)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def _resolve(c: pd.DataFrame, names: list[str] | None) -> list[int]:
    """Map player names to row positions: any case, accents optional, and a part of the name is fine
    as long as it starts a word ("salah", "Odegaard", "Haal"). Unknown or ambiguous names raise."""
    idx = []
    folded = c["name"].map(_fold)
    for n in names or []:
        q = _fold(n)
        exact = np.flatnonzero(folded == q)
        word = folded.str.contains(r"(?:^|[\s\-'])" + re.escape(q), regex=True)
        hits = exact if len(exact) else np.flatnonzero(word)
        if len(hits) == 0:
            raise ValueError(f"No player matching '{n}' in this gameweek.")
        if len(hits) > 1:
            opts = ", ".join(c.loc[hits[:6], "name"])
            raise ValueError(f"'{n}' matches several players ({opts}). Use a fuller name.")
        idx.append(int(hits[0]))
    return idx


def _solve(c: pd.DataFrame, budget_m: float, gamma: int, lock: list[int], exclude: list[int]) -> Team:
    d = (c["mu"] - c["lo"]).clip(lower=0).to_numpy()
    return solve_team(c, c["mu"].to_numpy(), int(round(budget_m * 10)), SETTINGS.club_cap, True,
                      d=d, gamma=gamma, thetas=16, lock=lock, exclude=exclude)


def _expected(c: pd.DataFrame, t: Team) -> float:
    mu = c["mu"].to_numpy()
    return float(mu[t.idx].sum() + mu[t.captain])


def pick_team(gameweek: int, budget_m: float = 83.5, risk_gamma: int = 0, must_include: list[str] | None = None,
              exclude: list[str] | None = None, reveal_actual: bool = False) -> dict:
    """Pick the provably optimal starting XI and captain for a 2025-26 gameweek.

    Args:
        gameweek: 1-38.
        budget_m: budget for the eleven starters in GBP millions (default 83.5, the XI share of 100m).
        risk_gamma: 0 = maximise expected points; higher = protect against that many players
            falling to the bottom of their 80% interval at once (Bertsimas-Sim robust optimisation).
        must_include: player names that must be in the XI.
        exclude: player names that must not be picked.
        reveal_actual: also return the points the team really scored (hindsight).
    """
    c = _gameweek(gameweek)
    try:
        t = _solve(c, budget_m, risk_gamma, _resolve(c, must_include), _resolve(c, exclude))
    except InfeasibleTeamError as e:
        return {"error": f"No valid team: {e}. Loosen the budget, locks or exclusions."}
    xi = c.iloc[t.idx].assign(_o=lambda x: x["position"].map(ORDER)).sort_values(["_o", "mu"], ascending=[True, False])
    players = [{"name": r.name, "position": r.position, "club": r.team, "price_m": r.value / 10,
                "expected_points": round(r.mu, 2), "interval_80": [round(max(r.lo, 0), 1), round(r.hi, 1)],
                "p_60_minutes": round(r.p_play60, 2), "captain": int(i) == t.captain,
                **({"actual_points": int(r.total_points)} if reveal_actual else {})}
               for i, r in zip(xi.index, xi.itertuples(), strict=True)]
    out = {"gameweek": gameweek, "budget_m": budget_m, "risk_gamma": risk_gamma, "players": players,
           "expected_points_captain_doubled": round(_expected(c, t), 1),
           "cost_m": round(float(c["value"].to_numpy()[t.idx].sum()) / 10, 1),
           "solver": ("exact MILP (HiGHS, solved to proven optimality)" if risk_gamma == 0 else
                      "robust MILP decomposition on a 16-point theta grid (a valid robust team; at worst slightly conservative)")}
    if risk_gamma > 0:
        out["worst_case_points"] = round(t.objective, 1)
    if reveal_actual:
        out["actual_points"] = realised_points(c, t)
    return out


def explain_player(gameweek: int, name: str, budget_m: float = 83.5) -> dict:
    """Explain one player's forecast and what he is worth to the optimal team.

    Returns the two-stage forecast (chance of playing, points if he plays, 80% interval) and his
    marginal value: expected points lost if he is removed and the team re-optimised (if he was
    picked), or the cost of forcing him in (if he was not).
    """
    c = _gameweek(gameweek)
    (i,) = _resolve(c, [name])
    r = c.iloc[i]
    try:
        base = _solve(c, budget_m, 0, [], [])
    except InfeasibleTeamError as e:
        return {"error": f"No valid team at this budget: {e}"}
    picked = i in set(base.idx)
    try:
        alt = _solve(c, budget_m, 0, [] if picked else [i], [i] if picked else [])
        delta = round(_expected(c, base) - _expected(c, alt), 2)
    except InfeasibleTeamError:
        delta = None
    return {
        "name": r["name"], "position": r["position"], "club": r["team"], "price_m": r["value"] / 10,
        "p_did_not_play": round(r["p_play0"], 2), "p_1_59_minutes": round(r["p_play1_59"], 2), "p_60_minutes": round(r["p_play60"], 2),
        "points_if_60_minutes": round(r["pts_if_60"], 2), "expected_points": round(r["mu"], 2),
        "interval_80": [round(max(r["lo"], 0), 1), round(r["hi"], 1)],
        "points_last_week": None if pd.isna(r["total_points_r1"]) else float(r["total_points_r1"]),
        "mean_points_last_5": None if pd.isna(r["total_points_r5"]) else round(float(r["total_points_r5"]), 2),
        "in_optimal_team": picked,
        ("expected_points_lost_without_him" if picked else "expected_points_cost_of_forcing_him_in"): delta,
    }


def compare_risk(gameweek: int, budget_m: float = 83.5, gammas: list[int] | None = None) -> list[dict]:
    """Show the risk-return trade-off for a gameweek: for each risk level gamma, the expected points,
    the guaranteed worst case if gamma players hit their downside, and which players change."""
    c = _gameweek(gameweek)
    out, base = [], None
    for g in gammas or [0, 2, 4, 6]:
        try:
            t = _solve(c, budget_m, g, [], [])
        except InfeasibleTeamError as e:
            return [{"error": f"No valid team at this budget: {e}"}]
        names = set(c.loc[t.idx, "name"])
        base = names if base is None else base
        d = (c["mu"] - c["lo"]).clip(lower=0).to_numpy()
        worst = _expected(c, t) - np.sort(d[t.idx])[::-1][:g].sum()
        out.append({"risk_gamma": g, "expected_points": round(_expected(c, t), 1), "worst_case_points": round(float(worst), 1),
                    "players_in": sorted(names - base), "players_out": sorted(base - names)})
    return out


def backtest_summary() -> dict:
    """Out-of-sample results on all 38 gameweeks of 2025-26: mean XI points per strategy (same optimiser
    and rules for all) and the model's paired-bootstrap advantage over each baseline."""
    s = pd.read_csv(REPORTS_DIR / "strategies_all_gws.csv")[["strategy", "mean_pts_per_gw", "total_pts"]]
    g = pd.read_csv(REPORTS_DIR / "significance_all_gws.csv")
    return {"strategies": s.round(1).to_dict(orient="records"), "model_vs_baselines": g.round(2).to_dict(orient="records"),
            "note": "Walk-forward: the model never saw a gameweek before forecasting it."}
