"""Load and clean the FPL Historical Dataset (Vaastav Anand, MIT licence).

Output grain: one row per (season, gameweek, player) with fixture-level stats summed,
plus a team table with one row per (season, gameweek, team).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import POSITIONS, RAW_DIR, RAW_URL, SCORING_ERA, SEASONS

log = logging.getLogger(__name__)

SUM_COLS = [
    "total_points", "minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "bonus", "bps", "saves", "yellow_cards", "red_cards", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded", "defensive_contribution",
]
REQUIRED = {"name", "position", "team", "element", "GW", "fixture", "minutes", "total_points", "value", "was_home",
            "team_h_score", "team_a_score", "kickoff_time"}


def raw_path(season: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"merged_gw_{season}.csv"


def download(seasons: list[str] = SEASONS, raw_dir: Path = RAW_DIR) -> None:
    """Fetch the per-season merged gameweek files (only needed to refresh the committed copies)."""
    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)
    for s in seasons:
        r = requests.get(RAW_URL.format(season=s), timeout=60)
        r.raise_for_status()
        raw_path(s, raw_dir).write_bytes(r.content)
        log.info("downloaded %s (%d bytes)", s, len(r.content))


def load_fixture_rows(season: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per player per fixture, cleaned. Drops assistant managers and validates the schema."""
    df = pd.read_csv(raw_path(season, raw_dir), low_memory=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{season}: missing columns {sorted(missing)}")
    df = df[df["position"].isin(POSITIONS)].copy()
    for c in SUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["season"] = season
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    df["was_home"] = df["was_home"].astype(str).str.lower().isin(["true", "1"])
    df["team_goals"] = np.where(df["was_home"], df["team_h_score"], df["team_a_score"])
    df["opp_goals"] = np.where(df["was_home"], df["team_a_score"], df["team_h_score"])
    # Opponent *name* from the fixture: the other team that has rows in the same fixture.
    teams = df.groupby(["fixture", "was_home"])["team"].agg(lambda s: s.mode().iat[0]).unstack()
    teams.columns = ["away", "home"] if list(teams.columns) == [False, True] else teams.columns
    df = df.join(teams, on="fixture")
    df["opponent"] = np.where(df["was_home"], df["away"], df["home"])
    return df.drop(columns=["home", "away"])


def _player_keys(df: pd.DataFrame) -> pd.Series:
    """Stable cross-season key: the player's name, disambiguated by FPL id if two players share a name."""
    dup = df.groupby("name")["element"].transform("nunique") > 1
    return np.where(dup, df["name"] + "#" + df["element"].astype(str), df["name"])


def build_tables(seasons: list[str] = SEASONS, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (players, teams). players: one row per (season, GW, key). teams: one row per (season, GW, team)."""
    frames = []
    for s in seasons:
        f = load_fixture_rows(s, raw_dir)
        f["key"] = _player_keys(f)
        frames.append(f)
    fx = pd.concat(frames, ignore_index=True)

    # ---- team table (per GW): goals and xG for/against, number of matches ----
    tf = (fx.groupby(["season", "GW", "fixture", "team", "opponent"], as_index=False)
            .agg(goals_for=("team_goals", "first"), goals_against=("opp_goals", "first"),
                 xg_for=("expected_goals", "sum")))
    opp_xg = tf[["season", "fixture", "team", "xg_for"]].rename(columns={"team": "opponent", "xg_for": "xg_against"})
    tf = tf.merge(opp_xg, on=["season", "fixture", "opponent"], how="left")
    teams = (tf.groupby(["season", "GW", "team"], as_index=False)
               .agg(matches=("fixture", "nunique"), goals_for=("goals_for", "sum"),
                    goals_against=("goals_against", "sum"), xg_for=("xg_for", "sum"), xg_against=("xg_against", "sum")))

    # ---- player table (per GW) ----
    agg = {c: "sum" for c in SUM_COLS}
    agg.update({"value": "first", "position": "first", "team": "first", "name": "first", "element": "first",
                "was_home": "mean", "fixture": "nunique", "selected": "first", "transfers_balance": "first",
                "kickoff_time": "min"})
    players = fx.groupby(["season", "GW", "key"], as_index=False).agg(agg)
    players = players.rename(columns={"fixture": "n_fixtures"})
    # minutes of 0 on every row of a season means the column was all-NaN (defensive_contribution pre-2025-26)
    for c in ["defensive_contribution"]:
        has = fx.groupby("season")[c].apply(lambda s: s.notna().any())
        players.loc[players["season"].map(has).eq(False), c] = np.nan
    players["scoring_era"] = players["season"].map(SCORING_ERA).astype(int)
    opps = fx.groupby(["season", "GW", "key"])["opponent"].agg(list).rename("opponents")
    players = players.join(opps, on=["season", "GW", "key"])
    players["season_idx"] = players["season"].map({s: i for i, s in enumerate(seasons)})
    players = players.sort_values(["key", "season_idx", "GW"]).reset_index(drop=True)
    log.info("players table %s, teams table %s", players.shape, teams.shape)
    return players, teams
