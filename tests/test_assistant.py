"""The AI-assistant tools run on the committed forecasts and must never return an illegal team."""
import pytest

from fplopt import assistant
from fplopt.optimise import XI_LIMITS


def _legal(team: dict, budget_m: float) -> None:
    ps = team["players"]
    assert len(ps) == 11
    assert sum(p["price_m"] for p in ps) <= budget_m + 1e-9
    for pos, (lo, hi) in XI_LIMITS.items():
        assert lo <= sum(p["position"] == pos for p in ps) <= hi
    clubs = [p["club"] for p in ps]
    assert max(clubs.count(c) for c in set(clubs)) <= 3
    assert sum(p["captain"] for p in ps) == 1


def test_pick_team_is_legal_and_honours_locks():
    base = assistant.pick_team(20)
    _legal(base, 83.5)
    star = max(base["players"], key=lambda p: p["expected_points"])["name"]
    out = assistant.pick_team(20, budget_m=78, exclude=[star])
    _legal(out, 78)
    assert star not in {p["name"] for p in out["players"]}


def test_robust_team_reports_a_worst_case_below_its_expectation():
    t = assistant.pick_team(20, risk_gamma=3)
    assert t["worst_case_points"] <= t["expected_points_captain_doubled"]


def test_unknown_or_ambiguous_names_fail_loudly():
    with pytest.raises(ValueError, match="No player"):
        assistant.pick_team(20, must_include=["Not A Real Footballer"])
    with pytest.raises(ValueError, match="several players"):
        assistant.pick_team(20, must_include=["james"])
    with pytest.raises(ValueError, match="Gameweek"):
        assistant.pick_team(99)


@pytest.mark.parametrize("typed", ["Odegaard", "ødegaard", "Martin Odegaard", "salah", "HAAL"])
def test_names_match_without_special_letters_and_by_word_start(typed):
    c = assistant._gameweek(20)
    (i,) = assistant._resolve(c, [typed])
    assert assistant._fold(typed) in assistant._fold(c.at[i, "name"])


def test_locked_player_is_always_picked():
    player = assistant.explain_player(20, "Odegaard")
    locked = assistant.pick_team(20, must_include=["Odegaard"])
    _legal(locked, 83.5)
    assert player["name"] in {p["name"] for p in locked["players"]}


def test_explain_player_and_risk_tradeoff():
    name = assistant.pick_team(20)["players"][0]["name"]
    e = assistant.explain_player(20, name)
    assert e["in_optimal_team"] and e["expected_points_lost_without_him"] >= 0
    assert abs(e["p_did_not_play"] + e["p_1_59_minutes"] + e["p_60_minutes"] - 1) < 0.02
    risk = assistant.compare_risk(20, gammas=[0, 4])
    assert risk[1]["expected_points"] <= risk[0]["expected_points"] + 1e-9


def test_mcp_server_exposes_the_tools():
    pytest.importorskip("mcp")
    from fplopt.mcp_server import TOOLS, build_server

    assert build_server() is not None
    assert {f.__name__ for f in TOOLS} == {"pick_team", "explain_player", "compare_risk", "backtest_summary"}
