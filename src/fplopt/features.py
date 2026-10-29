"""As-of feature engineering.

Rule: a feature on the row for gameweek t may only use information available before the
gameweek-t deadline. Every rolling statistic is therefore built from *earlier* rows
(`shift(1)` inside each player or team), and team form is aggregated per gameweek first, so
a double gameweek's second match never leaks into the first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLL_COLS = ["total_points", "minutes", "starts", "expected_goals", "expected_assists",
             "expected_goals_conceded", "bps", "clean_sheets", "saves", "defensive_contribution"]
WINDOWS = (1, 3, 5, 10)


def _rolling_prev(df: pd.DataFrame, group: list[str], col: str, w: int) -> pd.Series:
    prev = df.groupby(group, sort=False)[col].shift(1)
    return prev.groupby([df[g] for g in group], sort=False).transform(lambda s: s.rolling(w, min_periods=1).mean())


def team_form(teams: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Per (season, GW, team): rolling per-match goals and xG for/against over the previous `window` gameweeks.

    Team history runs across seasons (by team name), so a team's first gameweek uses last season's form.
    """
    t = teams.copy()
    order = {s: i for i, s in enumerate(sorted(t["season"].unique()))}
    t["season_idx"] = t["season"].map(order)
    t = t.sort_values(["team", "season_idx", "GW"]).reset_index(drop=True)
    out = t[["season", "GW", "team"]].copy()
    for c in ["goals_for", "goals_against", "xg_for", "xg_against"]:
        num = t.groupby("team")[c].shift(1).groupby(t["team"]).transform(lambda s: s.rolling(window, min_periods=1).sum())
        den = t.groupby("team")["matches"].shift(1).groupby(t["team"]).transform(lambda s: s.rolling(window, min_periods=1).sum())
        out[f"{c}_pm"] = num / den
    return out


def build_features(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Return the modelling table: one row per (season, GW, player) with as-of features and targets."""
    p = players.sort_values(["key", "season_idx", "GW"]).reset_index(drop=True).copy()
    g = ["key"]

    # Player form across seasons (by key).
    for c in ROLL_COLS:
        for w in WINDOWS:
            if c in ("saves", "clean_sheets", "defensive_contribution") and w in (1, 3):
                continue
            p[f"{c}_r{w}"] = _rolling_prev(p, g, c, w)
    p["played60"] = (p["minutes"] >= 60).astype(float)
    p["played_any"] = (p["minutes"] > 0).astype(float)
    for w in (3, 5, 10):
        p[f"share60_r{w}"] = _rolling_prev(p, g, "played60", w)
        p[f"shareany_r{w}"] = _rolling_prev(p, g, "played_any", w)

    # Points per 90 over the last 10 appearances-weighted window.
    pts10 = p.groupby(g)["total_points"].shift(1).groupby(p["key"]).transform(lambda s: s.rolling(10, min_periods=1).sum())
    min10 = p.groupby(g)["minutes"].shift(1).groupby(p["key"]).transform(lambda s: s.rolling(10, min_periods=1).sum())
    p["pts_per90_r10"] = np.where(min10 > 0, pts10 / min10 * 90, np.nan)

    # Gameweeks since the player last played (as-of), counted over his previous rows.
    last_played = p["played_any"].where(p["played_any"] > 0).groupby(p["key"]).cumcount()
    idx = p.groupby(g).cumcount()
    marker = pd.Series(np.where(p["played_any"] > 0, idx, np.nan), index=p.index)
    last_idx = marker.groupby(p["key"]).transform(lambda s: s.shift(1).ffill())
    p["rows_since_played"] = idx - last_idx
    del last_played

    # Season-to-date points per game (same season only, as-of).
    p["season_ppg"] = (p.groupby(["key", "season"])["total_points"]
                         .transform(lambda s: s.shift(1).expanding().mean()))
    # Previous season summary, joined by key.
    prev = (p.groupby(["key", "season_idx"]).agg(prev_ppg=("total_points", "mean"), prev_minutes=("minutes", "mean"),
                                                 prev_xgi=("expected_goal_involvements", "mean")).reset_index())
    prev["season_idx"] = prev["season_idx"] + 1
    p = p.merge(prev, on=["key", "season_idx"], how="left")

    # Price and crowd signals. Price is set before the deadline; crowd signals are lagged one gameweek.
    p["price"] = p["value"] / 10.0
    p["price_change"] = p["value"] - p.groupby("key")["value"].shift(1)
    p["log_selected_lag"] = np.log1p(p.groupby("key")["selected"].shift(1))
    p["transfers_balance_lag"] = p.groupby("key")["transfers_balance"].shift(1) / 1e5

    # Fixture context (known before the deadline).
    p["is_dgw"] = (p["n_fixtures"] > 1).astype(int)
    tf = team_form(teams)
    own = tf.rename(columns={c: f"own_{c}" for c in tf.columns if c.endswith("_pm")})
    p = p.merge(own, on=["season", "GW", "team"], how="left")
    opp_long = p[["season", "GW", "key", "opponents"]].explode("opponents").rename(columns={"opponents": "team"})
    opp_long = opp_long.merge(tf, on=["season", "GW", "team"], how="left")
    opp_feat = (opp_long.groupby(["season", "GW", "key"])[[c for c in tf.columns if c.endswith("_pm")]].mean()
                        .add_prefix("opp_").reset_index())
    p = p.merge(opp_feat, on=["season", "GW", "key"], how="left")

    p["pos_code"] = p["position"].map({"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}).astype(int)
    p["minutes_bucket"] = np.select([p["minutes"] <= 0, p["minutes"] < 60], [0, 1], default=2).astype(int)
    return p.sort_values(["season_idx", "GW", "key"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns the models may use. Everything here is as-of; targets and identifiers are excluded."""
    base = [c for c in df.columns if any(c.startswith(f"{r}_r") for r in ROLL_COLS)]
    base += [c for c in df.columns if c.startswith(("share60_", "shareany_", "own_", "opp_"))]
    base += ["pts_per90_r10", "rows_since_played", "season_ppg", "prev_ppg", "prev_minutes", "prev_xgi",
             "price", "price_change", "log_selected_lag", "transfers_balance_lag",
             "n_fixtures", "is_dgw", "was_home", "pos_code", "scoring_era"]
    return [c for c in dict.fromkeys(base) if c in df.columns]
