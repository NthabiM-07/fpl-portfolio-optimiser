"""The multi-week planner must keep every FPL squad rule and count free transfers and hits correctly."""
from pathlib import Path

import numpy as np
import pandas as pd

from fplopt.config import XI_LIMITS
from fplopt.planner import SQUAD, PlanConfig, autosub_points, plan_transfers, simulate_season


def _pool(seed=0, n=60, clubs=12):
    rng = np.random.default_rng(seed)
    pos = np.array(["GK"] * 6 + ["DEF"] * 18 + ["MID"] * 22 + ["FWD"] * 14)[:n]
    return (rng.uniform(1, 7, (3, n)), rng.integers(40, 120, n).astype(float), pos,
            np.array([f"C{i % clubs}" for i in rng.permutation(n)]))


def _legal_squad(squad, xi, captain, price, pos, club, budget):
    assert len(squad) == 15 and price[squad].sum() <= budget + 1e-6
    for p, k in SQUAD.items():
        assert (pos[squad] == p).sum() == k
    assert pd.Series(club[squad]).value_counts().max() <= 3
    assert set(xi) <= set(squad) and len(xi) == 11 and captain in set(xi)
    for p, (lo, hi) in XI_LIMITS.items():
        assert lo <= (pos[xi] == p).sum() <= hi


def test_first_week_builds_a_legal_squad_and_xi():
    mu, price, pos, club = _pool()
    p = plan_transfers(mu, price, pos, club, np.zeros(len(pos)), 15, 1000.0, PlanConfig())
    _legal_squad(p.squad, p.xi, p.captain, price, pos, club, 1000.0)


def test_extra_transfers_are_charged_and_a_high_hurdle_stops_them():
    mu, price, pos, club = _pool(1)
    first = plan_transfers(mu, price, pos, club, np.zeros(len(pos)), 15, 1000.0, PlanConfig())
    current = np.zeros(len(pos))
    current[first.squad] = 1
    mu2 = mu.copy()
    outfield = [i for i in first.xi if pos[i] != "GK"][:2]
    mu2[:, outfield] = 0.0                        # two starters suddenly worthless: worth replacing both
    budget = 1000.0
    cheap = plan_transfers(mu2, price, pos, club, current, 1, budget, PlanConfig(hit_cost=0.5))
    assert len(cheap.buys) >= 2 and cheap.paid == len(cheap.buys) - 1
    _legal_squad(cheap.squad, cheap.xi, cheap.captain, price, pos, club, budget)
    strict = plan_transfers(mu2, price, pos, club, current, 1, budget, PlanConfig(hit_cost=100))
    assert strict.paid == 0 and len(strict.buys) <= 1


def test_autosubs_keep_a_legal_formation_and_vice_captain_steps_in():
    pos = {"g1": "GK", "g2": "GK", **{f"d{i}": "DEF" for i in range(5)}, **{f"m{i}": "MID" for i in range(5)},
           **{f"f{i}": "FWD" for i in range(3)}}
    squad = list(pos)
    xi = ["g1", "d0", "d1", "d2", "m0", "m1", "m2", "m3", "f0", "f1", "f2"]
    mins = {k: 90 for k in squad}
    mins.update({"d0": 0, "m0": 0})               # a defender and the captain do not play
    pts = {k: 2 for k in squad}
    pts.update({"d3": 6, "m4": 5, "m1": 10})
    mu = {k: 1.0 for k in squad}
    mu.update({"d3": 3.0, "m4": 2.0})
    total, subs = autosub_points(squad, xi, "m0", "m1", pts, mins, pos, mu)
    assert subs == 2                              # d3 for d0 (keeps 3 DEF), then m4 for m0
    # final XI: g1, d1, d2, m2, m3, f0, f1, f2 (8 x 2) + d3 (6) + m4 (5) + m1 (10, doubled as vice-captain)
    assert total == 8 * 2 + 6 + 5 + 10 + 10


def test_one_short_season_keeps_the_rules_every_week():
    pred = pd.read_csv(Path(__file__).resolve().parents[1] / "reports" / "predictions_2025-26.csv.gz")
    r = simulate_season(pred[pred["GW"] <= 3], "mu", PlanConfig(horizon=2))
    assert len(r) == 3
    assert (r["bank_m"] >= -1e-9).all()               # never overspends (squad value may grow with price rises)
    assert (r["free_transfers_next"].between(1, 5)).all()
    assert (r["hits"] % 4 == 0).all() and (r["hits"] >= 0).all() and r.loc[r["GW"] == 1, "hits"].eq(0).all()


def test_banked_free_transfers_cover_several_moves_and_hits_are_never_bought():
    mu, price, pos, club = _pool(2)
    first = plan_transfers(mu, price, pos, club, np.zeros(len(pos)), 15, 1000.0, PlanConfig())
    current = np.zeros(len(pos))
    current[first.squad] = 1
    mu2 = mu.copy()
    mu2[:, [i for i in first.xi if pos[i] != "GK"][:3]] = 0.0
    banked = plan_transfers(mu2, price, pos, club, current, 3, 1000.0, PlanConfig(hit_cost=4))
    assert banked.paid == max(0, len(banked.buys) - 3)          # up to 3 moves are free
    eager = plan_transfers(mu2, price, pos, club, current, 1, 1000.0, PlanConfig(hit_cost=0.5, decay=1.0, ft_value=3.0))
    assert eager.paid == max(0, len(eager.buys) - 1)              # never pays a hit while a free transfer is unused


def test_goalkeeper_is_only_replaced_by_the_bench_goalkeeper_and_a_blank_captain_passes_the_armband():
    pos = {"g1": "GK", "g2": "GK", **{f"d{i}": "DEF" for i in range(5)}, **{f"m{i}": "MID" for i in range(5)},
           **{f"f{i}": "FWD" for i in range(3)}}
    squad = list(pos)
    xi = ["g1", "d0", "d1", "d2", "m0", "m1", "m2", "m3", "f0", "f1", "f2"]
    mins = {k: 90 for k in squad}
    mins["g1"] = 0
    pts = {k: 1 for k in squad}
    pts.update({"g2": 7, "f0": 9})
    mu = {k: 1.0 for k in squad}
    total, subs = autosub_points(squad, xi, "f0", "m0", pts, mins, pos, mu)
    assert subs == 1 and total == 7 + 9 * 2 + 9 * 1          # g2 in for g1; captain f0 played, so doubled
    mins["f0"] = 0
    total, subs = autosub_points(squad, xi, "f0", "m0", pts, mins, pos, mu)
    assert total == 7 + 10 * 1 + 1                            # an outfielder replaces f0; vice-captain m0 doubles
