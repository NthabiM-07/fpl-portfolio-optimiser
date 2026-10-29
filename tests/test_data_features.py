import numpy as np
import pandas as pd
import pytest

from fplopt.config import RAW_DIR
from fplopt.data import build_tables
from fplopt.features import build_features, feature_columns

pytestmark = pytest.mark.skipif(not (RAW_DIR / "merged_gw_2024-25.csv").exists(), reason="raw data not present")


@pytest.fixture(scope="module")
def tables():
    return build_tables(["2024-25"])


def test_one_row_per_player_gameweek(tables):
    players, _ = tables
    assert not players.duplicated(["season", "GW", "key"]).any()
    assert set(players["position"]) <= {"GK", "DEF", "MID", "FWD"}
    assert (players["n_fixtures"] >= 1).all()


def test_features_do_not_see_the_future(tables):
    """Overwrite every outcome from gameweek 20 onward; features up to and including GW 20 must not change."""
    players, teams = tables
    base = build_features(players, teams)
    p2, t2 = players.copy(), teams.copy()
    future = p2["GW"] >= 20
    for c in ["total_points", "minutes", "starts", "expected_goals", "expected_assists", "bps", "selected", "transfers_balance"]:
        p2.loc[future, c] = 999
    t2.loc[t2["GW"] >= 20, ["goals_for", "goals_against", "xg_for", "xg_against"]] = 999
    pert = build_features(p2, t2)
    cols = feature_columns(base)
    a = base[base["GW"] <= 20].set_index(["season", "GW", "key"])[cols].sort_index()
    b = pert[pert["GW"] <= 20].set_index(["season", "GW", "key"])[cols].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_rolling_mean_uses_only_earlier_rows(tables):
    players, teams = tables
    f = build_features(players, teams).sort_values(["key", "GW"])
    k = f["key"].value_counts().index[0]
    g = f[f["key"] == k].reset_index(drop=True)
    expect = g["total_points"].shift(1).rolling(5, min_periods=1).mean()
    assert np.allclose(g["total_points_r5"].fillna(-1), expect.fillna(-1))
