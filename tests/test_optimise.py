import numpy as np
import pytest

from fplopt.baselines import baseline_scores
from fplopt.optimise import InfeasibleTeamError, check_team, solve_team


def test_team_obeys_every_rule(candidates):
    team = solve_team(candidates, candidates["mu"].to_numpy(), budget_tenths=835)
    assert check_team(candidates, team, 835) == []


def test_every_baseline_obeys_every_rule(candidates):
    for name, score in baseline_scores(candidates, np.random.default_rng(1)).items():
        team = solve_team(candidates, score, budget_tenths=835)
        assert check_team(candidates, team, 835) == [], name


def test_infeasible_budget_raises(candidates):
    with pytest.raises(InfeasibleTeamError):
        solve_team(candidates, candidates["mu"].to_numpy(), budget_tenths=300)


def test_gamma_zero_equals_risk_neutral(candidates):
    mu = candidates["mu"].to_numpy()
    d = np.full(len(mu), 2.0)
    a = solve_team(candidates, mu, 835)
    b = solve_team(candidates, mu, 835, d=d, gamma=0)
    assert set(a.idx) == set(b.idx)


def test_robust_team_is_valid_and_less_exposed(candidates):
    mu = candidates["mu"].to_numpy()
    d = np.random.default_rng(2).uniform(0, 4, len(mu))
    neutral = solve_team(candidates, mu, 835)
    robust = solve_team(candidates, mu, 835, d=d, gamma=4)
    assert check_team(candidates, robust, 835) == []
    assert np.sort(d[robust.idx])[-4:].sum() <= np.sort(d[neutral.idx])[-4:].sum() + 1e-9


def test_lock_and_exclude(candidates):
    mu = candidates["mu"].to_numpy()
    best = solve_team(candidates, mu, 835)
    out = int(best.idx[0])
    cheap_gk = int(candidates[candidates.position == "GK"].sort_values("value").index[0])
    t = solve_team(candidates, mu, 835, lock=[cheap_gk], exclude=[out])
    assert cheap_gk in set(t.idx) and out not in set(t.idx)


def test_decomposition_matches_direct_robust_milp(candidates):
    """Bertsimas-Sim decomposition (exact theta set) gives the same robust value as the one-shot MILP."""
    from fplopt.optimise import robust_value, solve_team_robust_milp
    small = candidates.sample(60, random_state=3).reset_index(drop=True)
    mu = small["mu"].to_numpy()
    d = np.random.default_rng(4).uniform(0, 3, len(mu))
    for g in (1, 2, 3):
        a = solve_team(small, mu, 835, d=d, gamma=g, thetas="exact")
        b = solve_team_robust_milp(small, mu, d, g, 835)
        assert abs(a.objective - robust_value(mu, d, b, g)) < 1e-6
