"""Team selection as a mixed-integer linear programme (portfolio construction under constraints).

Nominal problem (risk-neutral):

    max   sum_i mu_i x_i + sum_i mu_i c_i                       (captain scores double)
    s.t.  sum_i x_i = 11,   sum_i price_i x_i <= B,   sum_i c_i = 1,   c_i <= x_i
          formation: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD;  at most 3 players per club

Robust problem (Bertsimas & Sim): each player can fall short of mu_i by up to d_i >= 0 (the distance
to the bottom of his 80% interval); protect against the worst Gamma shortfalls in the team:

    max_x  sum_i mu_i x_i (+ captain)  -  max_{|S| <= Gamma} sum_{i in S} d_i x_i

Bertsimas & Sim (2003, Math. Programming 98:49-71) show this equals

    max_theta  [ -Gamma * theta + nominal problem with scores mu_i - max(d_i - theta, 0) ],

with theta ranging over {0, d_1, ..., d_n}; Lee & Kwon (2014) show about half of those values suffice.
So the robust team is found by solving a series of nominal MILPs (`thetas="exact"`), or, for speed,
a quantile grid of theta values (`thetas=<int>`). On the full ~790-player pool the exact set is no
faster than the direct MILP (both take tens of seconds at Gamma = 4); the 16-point grid takes about
1.5 s. Every candidate team is scored by its true worst case, so a grid answer is always a valid
robust team, at worst slightly conservative. `solve_team_robust_milp` is the direct single-MILP
formulation, kept to test that the exact decomposition agrees with it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .config import XI_LIMITS

MILP_OPTIONS = {"mip_rel_gap": 0.0}   # solve to proven optimality, not HiGHS's default 0.01% gap


class InfeasibleTeamError(ValueError):
    """No team satisfies the constraints (for example, the budget is too small)."""


@dataclass
class Team:
    idx: np.ndarray            # row positions in the candidate frame
    captain: int | None        # row position of the captain
    objective: float           # predicted points incl. captain (net of the robust protection when Gamma > 0)
    theta: float | None = None


def _validate(cands: pd.DataFrame) -> None:
    missing = {"value", "position", "team"} - set(cands.columns)
    if missing:
        raise ValueError(f"candidate frame missing columns {sorted(missing)}")
    if len(cands) < 11:
        raise InfeasibleTeamError(f"only {len(cands)} candidates; need at least 11")


def _constraints(cands: pd.DataFrame, n_extra: int, captain: bool, budget_tenths: int, club_cap: int):
    """Sparse rows shared by every formulation. Variables: x (n), c (n if captain), then n_extra others."""
    n = len(cands)
    nc = n if captain else 0
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(cidx, v, l, h):
        nonlocal r
        rows.extend([r] * len(cidx)); cols.extend(cidx); vals.extend(v); lo.append(l); hi.append(h); r += 1

    ix = np.arange(n)
    add(ix, np.ones(n), 11, 11)
    add(ix, cands["value"].to_numpy(float), 0, budget_tenths)
    pos = cands["position"].to_numpy()
    for p, (a, b) in XI_LIMITS.items():
        m = np.flatnonzero(pos == p)
        add(m, np.ones(len(m)), a, b)
    for _, m in pd.Series(ix).groupby(cands["team"].to_numpy()):
        add(m.to_numpy(), np.ones(len(m)), 0, club_cap)
    if captain:
        add(n + ix, np.ones(n), 1, 1)
        for i in range(n):
            add([n + i, i], [1.0, -1.0], -np.inf, 0)
    return rows, cols, vals, lo, hi, r, n + nc + n_extra


def _solve_nominal(cands, x_score, c_score, budget_tenths, club_cap, captain, lock, exclude) -> Team:
    n = len(cands)
    rows, cols, vals, lo, hi, r, N = _constraints(cands, 0, captain, budget_tenths, club_cap)
    c = np.zeros(N)
    c[:n] = -x_score
    if captain:
        c[n:2 * n] = -c_score
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, N))
    lb, ub = np.zeros(N), np.ones(N)
    for i in (lock or []):
        lb[i] = 1
    for i in (exclude or []):
        ub[i] = 0
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(N), bounds=Bounds(lb, ub), options=MILP_OPTIONS)
    if res.status != 0 or res.x is None:
        raise InfeasibleTeamError(f"no feasible team: {res.message}")
    x = np.flatnonzero(res.x[:n] > 0.5)
    cap = int(np.argmax(res.x[n:2 * n])) if captain else None
    return Team(idx=x, captain=cap, objective=float(-res.fun))


def robust_value(mu: np.ndarray, d: np.ndarray, team: Team, gamma: float) -> float:
    """Worst-case predicted points of a team when its `gamma` largest shortfalls all happen."""
    base = mu[team.idx].sum() + (mu[team.captain] if team.captain is not None else 0.0)
    worst = np.sort(d[team.idx])[::-1][: int(np.floor(gamma))].sum()
    frac = gamma - np.floor(gamma)
    if frac > 0 and int(gamma) < len(team.idx):
        worst += frac * np.sort(d[team.idx])[::-1][int(gamma)]
    return float(base - worst)


def _theta_values(d: np.ndarray, gamma: float, thetas) -> np.ndarray:
    if thetas == "exact":
        ds = np.sort(d)[::-1]                        # descending, d_1 >= ... >= d_n, plus d_{n+1} = 0
        n, g = len(ds), int(np.ceil(gamma))
        idx = list(range(g + 1, n + 1, 2)) + [n + 1]   # Lee & Kwon (2014) index set (1-based)
        vals = [ds[i - 1] if i <= n else 0.0 for i in idx]
        return np.unique(np.r_[vals, 0.0])
    return np.unique(np.r_[np.quantile(d, np.linspace(0, 1, int(thetas))), 0.0])


def solve_team(cands: pd.DataFrame, mu: np.ndarray, budget_tenths: int = 835, club_cap: int = 3,
               captain: bool = True, d: np.ndarray | None = None, gamma: float = 0.0,
               lock: list[int] | None = None, exclude: list[int] | None = None, thetas="exact") -> Team:
    """Best XI (and captain) for scores `mu`; robust to `gamma` shortfalls of size `d` when gamma > 0.
    `lock` / `exclude` are row positions to force in or out."""
    _validate(cands)
    mu = np.nan_to_num(np.asarray(mu, float))
    if gamma <= 0 or d is None:
        return _solve_nominal(cands, mu, mu, budget_tenths, club_cap, captain, lock, exclude)
    d = np.nan_to_num(np.asarray(d, float)).clip(min=0)
    # Each theta gives a candidate team; keep the one with the best TRUE worst case. With the exact
    # (Lee-Kwon) theta set this is the robust optimum; on a grid it is a feasible, slightly
    # conservative approximation that can only improve on ranking teams by the theta-bound.
    best: Team | None = None
    for th in _theta_values(d, gamma, thetas):
        t = _solve_nominal(cands, mu - np.maximum(d - th, 0), mu, budget_tenths, club_cap, captain, lock, exclude)
        val = robust_value(mu, d, t, gamma)
        if best is None or val > best.objective + 1e-9:
            best = Team(t.idx, t.captain, val, th)
    return best


def solve_team_robust_milp(cands, mu, d, gamma, budget_tenths=835, club_cap=3, captain=True) -> Team:
    """Direct one-MILP robust counterpart (slow for large gamma); used to validate `solve_team`."""
    n = len(cands)
    mu = np.asarray(mu, float); d = np.asarray(d, float).clip(min=0)
    rows, cols, vals, lo, hi, r, N = _constraints(cands, 1 + n, captain, budget_tenths, club_cap)
    nxc = n * (2 if captain else 1)
    z = nxc
    for i in range(n):                                   # z + q_i - d_i x_i >= 0
        rows.extend([r] * 3); cols.extend([z, z + 1 + i, i]); vals.extend([1.0, 1.0, -d[i]]); lo.append(0); hi.append(np.inf); r += 1
    c = np.zeros(N); c[:n] = -mu
    if captain:
        c[n:2 * n] = -mu
    c[z] = gamma; c[z + 1:] = 1.0
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, N))
    integ = np.r_[np.ones(nxc), np.zeros(1 + n)]
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(0, np.r_[np.ones(nxc), np.full(1 + n, np.inf)]), options=MILP_OPTIONS)
    if res.status != 0:
        raise InfeasibleTeamError(res.message)
    x = np.flatnonzero(res.x[:n] > 0.5)
    cap = int(np.argmax(res.x[n:2 * n])) if captain else None
    return Team(x, cap, float(-res.fun))


def realised_points(cands: pd.DataFrame, team: Team, col: str = "total_points") -> float:
    pts = cands[col].to_numpy(float)
    total = pts[team.idx].sum()
    if team.captain is not None:
        total += pts[team.captain]
    return float(total)


def check_team(cands: pd.DataFrame, team: Team, budget_tenths: int, club_cap: int = 3) -> list[str]:
    """Return a list of violated rules (empty list = valid)."""
    t = cands.iloc[team.idx]
    errs = []
    if len(team.idx) != 11:
        errs.append(f"{len(team.idx)} players")
    if t["value"].sum() > budget_tenths + 1e-6:
        errs.append(f"cost {t['value'].sum()} > {budget_tenths}")
    vc = t["position"].value_counts()
    for p, (a, b) in XI_LIMITS.items():
        if not a <= vc.get(p, 0) <= b:
            errs.append(f"{p}={vc.get(p, 0)}")
    if t["team"].value_counts().max() > club_cap:
        errs.append("club cap")
    if team.captain is not None and team.captain not in set(team.idx):
        errs.append("captain not in XI")
    return errs


def budget_shadow_price(cands: pd.DataFrame, mu: np.ndarray, budget_tenths: int = 835, club_cap: int = 3) -> float:
    """Dual value of the budget in the LP relaxation (no captain): predicted points per extra GBP 0.1m."""
    n = len(cands)
    A, b = [cands["value"].to_numpy(float)], [budget_tenths]
    pos = cands["position"].to_numpy()
    for p, (lo, hi) in XI_LIMITS.items():
        m = (pos == p).astype(float)
        A += [m, -m]; b += [hi, -lo]
    for t in pd.unique(cands["team"]):
        A.append((cands["team"].to_numpy() == t).astype(float)); b.append(club_cap)
    res = linprog(-np.nan_to_num(np.asarray(mu, float)), A_ub=np.array(A), b_ub=b,
                  A_eq=np.ones((1, n)), b_eq=[11], bounds=(0, 1), method="highs")
    return float(-res.ineqlin.marginals[0]) if res.status == 0 else float("nan")


def prefilter(cands: pd.DataFrame, mu: np.ndarray, per_position: int = 60) -> np.ndarray:
    """Row positions of the top-`per_position` candidates by mu in each position (speeds up robust sweeps)."""
    s = pd.Series(np.nan_to_num(mu), index=np.arange(len(cands)))
    keep = s.groupby(cands["position"].to_numpy()).nlargest(per_position).index.get_level_values(-1)
    return np.sort(keep.to_numpy())
