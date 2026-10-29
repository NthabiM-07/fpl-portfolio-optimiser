"""Property-based tests (Hypothesis): the optimiser's guarantees must hold on *any* player pool,
not just the hand-made fixtures. Hypothesis generates the pools and shrinks any failure to a
minimal counter-example."""
import numpy as np
import pandas as pd
import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from fplopt.backtest import diebold_mariano  # noqa: E402
from fplopt.model import cqr_correction  # noqa: E402
from fplopt.optimise import InfeasibleTeamError, check_team, robust_value, solve_team, solve_team_robust_milp  # noqa: E402

SETTINGS = settings(max_examples=25, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow])


def _pool(seed: int, n: int, clubs: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos = np.r_[["GK"] * 3, ["DEF"] * 6, ["MID"] * 6, ["FWD"] * 4, rng.choice(["GK", "DEF", "MID", "FWD"], n - 19)]
    return pd.DataFrame({"position": pos, "team": [f"C{i}" for i in rng.integers(0, clubs, n)],
                         "value": rng.integers(40, 131, n), "mu": rng.gamma(2.0, 1.5, n),
                         "d": rng.uniform(0, 4, n), "total_points": rng.poisson(3, n)})


@SETTINGS
@given(seed=st.integers(0, 10**6), n=st.integers(30, 120), clubs=st.integers(6, 20), budget=st.integers(700, 1000))
def test_any_solved_team_obeys_every_rule(seed, n, clubs, budget):
    c = _pool(seed, n, clubs)
    try:
        t = solve_team(c, c["mu"].to_numpy(), budget, 3, True)
    except InfeasibleTeamError:
        return                                            # allowed: an honest "no valid team"
    assert check_team(c, t, budget, 3) == []


@SETTINGS
@given(seed=st.integers(0, 10**6), budget=st.integers(750, 1000))
def test_milp_beats_every_random_legal_team(seed, budget):
    c = _pool(seed, 80, 12)
    mu = c["mu"].to_numpy()
    try:
        best = solve_team(c, mu, budget, 3, True)
    except InfeasibleTeamError:
        return
    rng = np.random.default_rng(seed)
    for _ in range(5):                                    # random legal teams via random objectives
        other = solve_team(c, rng.random(len(c)), budget, 3, True)
        assert mu[other.idx].sum() + mu[other.idx].max() <= best.objective + 1e-6


@SETTINGS
@given(seed=st.integers(0, 10**6))
def test_more_protection_never_raises_the_worst_case(seed):
    c = _pool(seed, 60, 12)
    mu, d = c["mu"].to_numpy(), c["d"].to_numpy()
    vals = [solve_team(c, mu, 900, 3, True, d=d, gamma=g).objective for g in (0, 1, 2, 4)]
    assert all(a >= b - 1e-6 for a, b in zip(vals, vals[1:]))


@settings(max_examples=10, deadline=None, derandomize=True)
@given(seed=st.integers(0, 10**6), gamma=st.sampled_from([1, 2, 3]))
def test_decomposition_matches_the_direct_robust_milp(seed, gamma):
    c = _pool(seed, 40, 10)
    mu, d = c["mu"].to_numpy(), c["d"].to_numpy()
    fast = solve_team(c, mu, 900, 3, True, d=d, gamma=gamma)
    direct = solve_team_robust_milp(c, mu, d, gamma, 900, 3, True)
    assert abs(fast.objective - direct.objective) < 1e-5
    assert abs(robust_value(mu, d, fast, gamma) - fast.objective) < 1e-9


@SETTINGS
@given(seed=st.integers(0, 10**6), alpha=st.sampled_from([0.1, 0.2, 0.3]))
def test_cqr_reaches_its_target_coverage_on_exchangeable_data(seed, alpha):
    rng = np.random.default_rng(seed)
    y = rng.standard_t(4, 4000)
    lo, hi = np.full(4000, -0.5), np.full(4000, 0.5)      # deliberately too narrow
    q = cqr_correction(y[:2000], lo[:2000], hi[:2000], alpha)
    cover = np.mean((y[2000:] >= lo[2000:] - q) & (y[2000:] <= hi[2000:] + q))
    assert cover >= 1 - alpha - 0.05


@SETTINGS
@given(seed=st.integers(0, 10**6))
def test_diebold_mariano_is_antisymmetric_and_detects_a_clear_winner(seed):
    rng = np.random.default_rng(seed)
    a = rng.gamma(2, 1, 38)
    b = a + rng.normal(1.0, 0.5, 38)                      # B is worse by ~1 every period
    s_ab, p_ab = diebold_mariano(a, b)
    s_ba, p_ba = diebold_mariano(b, a)
    assert np.isclose(s_ab, -s_ba) and np.isclose(p_ab, p_ba)
    assert s_ab > 0 and p_ab < 0.05
