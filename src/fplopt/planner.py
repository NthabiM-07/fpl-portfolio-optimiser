"""Multi-week planning: a 15-man squad managed through the season with transfers.

This is the full FPL game rather than one free team per week. Each gameweek a rolling-horizon MILP
plans the next H weeks and only the first week's decisions are carried out (model predictive
control, as in production planning):

    max  sum_h w_h * [ sum_i mu_ih (x_ih + c_ih) ]  -  4 * sum_h w_h * paid_h  +  v * ft_H
    s.t. squad:      sum_i s_ih = 15; 2 GK, 5 DEF, 5 MID, 3 FWD; <= 3 per club; value <= budget
         flow:       s_ih = s_i,h-1 + buy_ih - sell_ih                (s_i,-1 = current squad)
         starting XI: x_ih <= s_ih, 11 players in a legal formation; one captain c_ih <= x_ih
         transfers:  paid_h >= sum_i buy_ih - ft_h                      (each extra costs 4 points)
                     ft_h+1 <= ft_h - (sum_i buy_ih - paid_h) + 1,  ft <= 5   (bank up to 5)

w_h = decay^h discounts later weeks, whose forecasts are weaker; v values a banked transfer at the
end of the horizon so the plan doesn't burn transfers for nothing.

Leakage: at gameweek t only the forecast made for t exists. Later weeks reuse each player's
per-fixture rate (mu_t / fixtures_t) times his club's fixture count in that week, which the
fixture list shows in advance. Nothing about later results, injuries or transfers is used.

Simplifications (stated in the README): players are bought and sold at their current price (FPL
keeps half of any rise when you sell); no chips; bench order is by forecast.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .config import XI_LIMITS

log = logging.getLogger(__name__)
SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
HIT = 4.0                              # what FPL actually deducts per extra transfer


@dataclass
class PlanConfig:
    horizon: int = 3
    budget_tenths: int = 1000          # GBP 100m for the 15-man squad
    hit_cost: float = 4.0              # planning penalty per extra transfer (FPL deducts 4; a higher
                                       # "hurdle" guards against chasing noisy forecasts)
    max_free_transfers: int = 5        # FPL lets you bank up to 5 (since 2024-25)
    decay: float = 0.85                # weight on week h is decay**h
    ft_value: float = 1.0              # value of a banked free transfer at the horizon end, in points
    pool_per_position: int = 40        # candidates per position (plus the current squad)
    club_cap: int = 3
    time_limit: float = 60.0
    mip_rel_gap: float = 1e-4


@dataclass
class Plan:
    squad: np.ndarray                  # indices into the pool, week 0
    xi: np.ndarray
    captain: int
    buys: np.ndarray
    sells: np.ndarray
    paid: int
    objective: float
    status: str = ""
    extra: dict = field(default_factory=dict)


def plan_transfers(mu: np.ndarray, price: np.ndarray, position: np.ndarray, club: np.ndarray,
                   current: np.ndarray, free_transfers: int, budget: float, cfg: PlanConfig) -> Plan:
    """Solve the H-week plan. mu is (H, n); `current` is a 0/1 vector of the squad held now (all 0
    for the first week, where the whole squad is picked for free)."""
    H, n = mu.shape
    S, X, C, B, D = 0, 1, 2, 3, 4
    nv_week = 5 * n
    PAID = H * nv_week                              # paid_h, h = 0..H-1
    FT = PAID + H                                   # ft_h for h = 1..H  (ft_0 is a constant)
    N = FT + H

    def v(h, kind, i):
        return h * nv_week + kind * n + i

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(idx, coef, l, u):
        nonlocal r
        rows.extend([r] * len(idx)); cols.extend(idx); vals.extend(coef); lo.append(l); hi.append(u); r += 1

    ix = np.arange(n)
    for h in range(H):
        s_idx = [v(h, S, i) for i in ix]
        add(s_idx, [1.0] * n, 15, 15)
        for p, k in SQUAD.items():
            m = np.flatnonzero(position == p)
            add([v(h, S, i) for i in m], [1.0] * len(m), k, k)
        for cl in np.unique(club):
            m = np.flatnonzero(club == cl)
            add([v(h, S, i) for i in m], [1.0] * len(m), 0, cfg.club_cap)
        add(s_idx, list(price.astype(float)), 0, budget)
        for i in ix:                                # flow: s_h - s_{h-1} - buy + sell = 0
            if h == 0:
                add([v(0, S, i), v(0, B, i), v(0, D, i)], [1, -1, 1], float(current[i]), float(current[i]))
            else:
                add([v(h, S, i), v(h - 1, S, i), v(h, B, i), v(h, D, i)], [1, -1, -1, 1], 0, 0)
            add([v(h, X, i), v(h, S, i)], [1, -1], -np.inf, 0)          # x <= s
            add([v(h, C, i), v(h, X, i)], [1, -1], -np.inf, 0)          # c <= x
        add([v(h, X, i) for i in ix], [1.0] * n, 11, 11)
        for p, (a, b_) in XI_LIMITS.items():
            m = np.flatnonzero(position == p)
            add([v(h, X, i) for i in m], [1.0] * len(m), a, b_)
        add([v(h, C, i) for i in ix], [1.0] * n, 1, 1)
        buys = [v(h, B, i) for i in ix]
        # paid_h - sum(buy) + ft_h >= 0
        if h == 0:
            add([PAID] + buys, [1.0] + [-1.0] * n, -float(free_transfers), np.inf)
        else:
            add([PAID + h] + buys + [FT + h - 1], [1.0] + [-1.0] * n + [1.0], 0, np.inf)
        # ft_{h+1} - ft_h + sum(buy) - paid_h <= 1
        if h == 0:
            add([FT] + buys + [PAID], [1.0] + [1.0] * n + [-1.0], -np.inf, 1.0 + free_transfers)
        else:
            add([FT + h] + [FT + h - 1] + buys + [PAID + h], [1.0, -1.0] + [1.0] * n + [-1.0], -np.inf, 1.0)

    c = np.zeros(N)
    for h in range(H):
        w = cfg.decay ** h
        c[h * nv_week + X * n: h * nv_week + X * n + n] = -w * mu[h]
        c[h * nv_week + C * n: h * nv_week + C * n + n] = -w * mu[h]
        c[PAID + h] = w * cfg.hit_cost
    c[FT + H - 1] = -cfg.ft_value
    lb, ub = np.zeros(N), np.ones(N)
    ub[PAID:PAID + H] = 15
    ub[FT:FT + H] = cfg.max_free_transfers
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, N))
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(N), bounds=Bounds(lb, ub),
               options={"time_limit": cfg.time_limit, "mip_rel_gap": cfg.mip_rel_gap})
    if res.x is None:
        raise RuntimeError(f"planner found no feasible plan: {res.message}")
    x = res.x
    get = lambda kind: np.flatnonzero(x[kind * n: kind * n + n] > 0.5)
    return Plan(squad=get(S), xi=get(X), captain=int(get(C)[0]), buys=get(B), sells=get(D),
                paid=int(round(x[PAID])), objective=float(-res.fun), status=res.message)


def autosub_points(squad: list, xi: list, captain, vice, pts: dict, mins: dict, pos: dict, mu: dict) -> tuple[float, int]:
    """Realised FPL score with automatic substitutions: a starter who didn't play is replaced by the
    first bench player (bench ordered by forecast) who did, keeping a legal formation; the GK only by
    the bench GK. The vice-captain doubles if the captain didn't play. Returns (points, subs made)."""
    xi = list(xi)
    bench = sorted([k for k in squad if k not in xi], key=lambda k: -mu.get(k, 0.0))
    bench_gk = [k for k in bench if pos[k] == "GK"]
    bench_out = [k for k in bench if pos[k] != "GK"]
    subs = 0
    for k in [k for k in xi if mins.get(k, 0) <= 0]:
        if pos[k] == "GK":
            if bench_gk and mins.get(bench_gk[0], 0) > 0:
                xi[xi.index(k)] = bench_gk.pop(0); subs += 1
            continue
        for b in bench_out:
            if mins.get(b, 0) <= 0:
                continue
            trial = [b if q == k else q for q in xi]
            counts = pd.Series([pos[q] for q in trial]).value_counts()
            if all(lo_ <= counts.get(p, 0) <= hi_ for p, (lo_, hi_) in XI_LIMITS.items()):
                xi = trial; bench_out.remove(b); subs += 1
                break
    total = float(sum(pts.get(k, 0) for k in xi))
    cap = captain if mins.get(captain, 0) > 0 else (vice if vice in xi and mins.get(vice, 0) > 0 else None)
    if cap is not None:
        total += pts.get(cap, 0)
    return total, subs


def simulate_season(pred: pd.DataFrame, score_col: str = "mu", cfg: PlanConfig | None = None,
                    hold: bool = False) -> pd.DataFrame:
    """Run a whole season week by week. `score_col` is the forecast to plan with; `hold=True` picks
    the squad in week 1 and never transfers again (only the XI and captain change)."""
    cfg = cfg or PlanConfig()
    gws = sorted(pred["GW"].unique())
    by_gw = {g: d.set_index("key") for g, d in pred.groupby("GW")}
    team_fx = pred.groupby(["GW", "team"])["n_fixtures"].max()       # club fixture counts (known in advance)
    last_price: dict = {}
    last_rate: dict = {}
    pos_of: dict = dict(zip(pred["key"], pred["position"]))
    club_of: dict = {}
    squad: list = []
    bank = float(cfg.budget_tenths)
    ft = 15                                                          # week 1: pick the whole squad free
    rows = []
    for t in gws:
        d = by_gw[t]
        last_price.update(d["value"].to_dict())
        club_of.update(d["team"].to_dict())
        rate_now = (d[score_col].fillna(0) / d["n_fixtures"].clip(lower=1)).to_dict()
        last_rate.update(rate_now)
        future = [g for g in gws if g >= t][: (1 if hold and t > gws[0] else cfg.horizon)]

        def fx(k, g):
            n_fx = team_fx.get((g, club_of.get(k)))
            return 0.0 if n_fx is None or pd.isna(n_fx) else float(n_fx)

        # candidate pool: best per position by discounted forecast, plus the current squad
        weights = np.array([cfg.decay ** h for h in range(len(future))])
        val = {k: last_rate.get(k, 0.0) * float(np.dot(weights, [fx(k, g) for g in future])) for k in d.index}
        cand = pd.Series(val).groupby(pd.Series({k: pos_of[k] for k in val})).nlargest(cfg.pool_per_position)
        keys = list(dict.fromkeys(list(cand.index.get_level_values(-1)) + squad))
        n = len(keys)
        mu = np.zeros((len(future), n))
        for h, g in enumerate(future):
            for j, k in enumerate(keys):
                if h == 0:
                    mu[h, j] = float(d[score_col].get(k, 0.0)) if k in d.index else 0.0
                else:
                    mu[h, j] = last_rate.get(k, 0.0) * fx(k, g)
        mu = np.nan_to_num(mu)
        price = np.array([last_price[k] for k in keys], float)
        position = np.array([pos_of[k] for k in keys])
        club = np.array([club_of[k] for k in keys])
        current = np.array([1.0 if k in squad else 0.0 for k in keys])
        budget = bank + float(price[current > 0].sum())
        if hold and squad:
            sub = np.flatnonzero(current > 0)
            p = plan_transfers(mu[:1, sub], price[sub], position[sub], club[sub], np.ones(len(sub)), 0, 1e9,
                               PlanConfig(**{**cfg.__dict__, "horizon": 1, "ft_value": 0.0}))
            p.squad, p.xi, p.captain, p.buys, p.sells = sub[p.squad], sub[p.xi], int(sub[p.captain]), np.array([], int), np.array([], int)
        else:
            p = plan_transfers(mu, price, position, club, current, min(ft, 15), budget, cfg)
        new_squad = [keys[j] for j in p.squad]
        n_buys = len(p.buys)
        if squad:                                                    # bank and free transfers after this week
            bank += float(price[p.sells].sum() - price[p.buys].sum())
            used_free = n_buys - p.paid
            ft = int(min(cfg.max_free_transfers, ft - used_free + 1))
        else:
            bank = cfg.budget_tenths - float(price[p.squad].sum())
            ft = 1
        squad = new_squad
        xi = [keys[j] for j in p.xi]
        cap = keys[p.captain]
        mu_now = {k: float(mu[0, keys.index(k)]) for k in squad}
        vice = max((k for k in xi if k != cap), key=lambda k: mu_now[k], default=None)
        pts = d["total_points"].to_dict()
        mins = d["minutes"].to_dict()
        score, subs = autosub_points(squad, xi, cap, vice, pts, mins, pos_of, mu_now)
        hits = HIT * p.paid if t > gws[0] else 0.0
        rows.append({"GW": t, "points": score - hits, "gross": score, "hits": hits, "transfers": n_buys if t > gws[0] else 0,
                     "free_transfers_next": ft, "bank_m": bank / 10, "squad_value_m": float(price[p.squad].sum()) / 10,
                     "autosubs": subs, "captain": d["name"].get(cap, cap) if cap in d.index else cap,
                     "planned_points": p.objective})
        log.info("GW%d %s: %.0f pts (%d transfers, %.0f hit)", t, score_col, score - hits, rows[-1]["transfers"], hits)
    return pd.DataFrame(rows)
