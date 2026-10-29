# AGENTS.md: Technical case study (Senior Quant): full context for coding agents

> **Read this first, agent.** This file is the single source of truth for the repository. Read it fully before you write any code. Facts were verified on **28 Oct 2026**. Where this file and the starter code disagree, this file wins. Do not invent numbers; every figure here has a source in §12.
>
> **Where it lives.** Commit it at the repo root as `AGENTS.md`. The VS Code Copilot agent, the GitHub Copilot coding agent, the Copilot CLI and GitHub.com code review all read `AGENTS.md`. VS Code's *local* code review reads only `.github/copilot-instructions.md`, so copy this file there too if you want that ([GitHub docs](https://docs.github.com/en/copilot/reference/custom-instructions-support)).

---

## STATUS: built and backtested (29 Oct 2026, 06:00 SAST)

This repository implements the spec below. Where the build differs from the spec, **the build wins** and the differences are listed here.

**Results (2025-26, all 38 GWs, walk-forward, captain doubled, GBP 83.5m XI):** two-stage model 58.5 pts/GW; last week 46.6; mean of last 5 46.4; most expensive 33.7; random legal 13.9; single-GBM ablation 53.1. Model vs last week: +11.9 pts/GW (95% CI +5.2 to +18.3), +25.5%. Interval coverage: 80.8% on players who played (target 80%). Robust Γ=4: 10th-percentile week +4.0, mean −0.3. EWMA flags DEF from GW3 (DefCon drift). Full tables: `reports/results.md`.

**Differences from the original spec:**
- Seasons: 2022-23 to 2025-26 (four, not three); a cross-season player key carries form over the summer, so GW1 has history.
- Validation: expanding-window walk-forward, retraining every 4 GWs across 2025-26 (not one fixed training split). CQR scores come from the most recent 8 GWs of out-of-sample predictions, seeded by a 2024-25 warm-up.
- Robust optimisation: solved by the Bertsimas & Sim (2003) decomposition into nominal MILPs (Lee & Kwon 2014 index set, or a theta grid); `solve_team_robust_milp` is kept only to test equivalence. The backtest sweep uses the top 60 per position and a 32-point grid.
- Intervals: a direct quantile GBM at 10/90 plus CQR (not a mixture of conditional quantiles).
- Primary metric view: all 38 GWs; secondary: GW 6-38 (both pre-declared).

**Commands:** `python main.py app` (demo) · `pytest -q` (12 tests) · `python main.py backtest` (~10 min full rebuild) · `python main.py pick --gw 20 --gamma 4`.

**Code map:** `src/fplopt/{config,data,features,model,optimise,baselines,backtest,monitor,reports,app}.py`, `tests/`, `reports/` (generated outputs and figures, committed so the demo runs offline).

---

## 0. TL;DR for the agent

- **Owner:** Nirvana (Nthabiseng) Maoka. Industrial engineer; senior data engineer and data scientist in credit at a bank (feature stores, entity resolution, lineage, IFRS 9).
- **Interview:** **Friday 30 Oct 2026**, 1 hour, same panel. Part A (pitch) is 15 min + 15 min Q&A; Part B (code) is 15 min + 15 min Q&A.
- **Hard deadline:** the GitHub repo is shared with the reviewers **≥ 24 h before the interview slot, i.e. Thursday 29 Oct 2026**. Reviewers read the code before the interview and will ask detailed questions.
- **Part B choice: Problem 2, the Fantasy Sports Team Optimiser**, on **real FPL data**, framed as **constrained portfolio construction**.
- **Build order (do not reorder):** (1) data loader + fair MILP baselines + backtest harness → (2) as-of features + two-stage model + calibrated intervals → (3) optimiser add-ons (robust Γ dial, sensitivity, EWMA monitor) → (4) Streamlit demo, README results table, tests → push.
- **Golden rules:** no leakage (features use only data before the gameweek); never random-split; every baseline obeys the same constraints as the optimiser; never use the `xP` column as a feature; seeds fixed; working beats perfect.

---

## 1. The brief (both parts)

### 1.1 Cover PDF (authoritative)
- Two parts, **equal weight**: Part A Creative, Part B Technical. "1 week" to complete both.
- Assesses: breaking down complex problems, innovative solutions, adaptability and originality. AI tools are allowed, but **the work must be your own and all sources referenced**.
- Part B: submit code via GitHub with access for the specified reviewers; present results plus a code walkthrough (15 min), then Q&A (15 min). Choose ONE of three problems. Synthetic data is a fallback, not mandatory. **Submit the GitHub link or code plus relevant data at least 24 h before the interview.**

### 1.2 Part B README and PROBLEM_OPTIONS (from the zip)
- Expected effort 10–15 h. "We value working solutions over perfect code, clear thinking over complex algorithms, creative problem-solving over templates."
- **Rubric (100):**
  - ML 25: model choice and justification 10, feature engineering 5, validation methodology 5, test-set performance 5.
  - Optimisation 25: formulation 10, algorithm choice and efficiency 5, constraint handling 5, solution quality 5.
  - Engineering 25: code quality 5, **working demo or visualisation 10**, scalability 5, testing and error handling 5.
  - Presentation 25: clear explanation 15, answering questions 10.
- **Submission requirements:** `README.md` (setup + how to run), `requirements.txt`, `main.py` (entry point), `src/`, a 1–2 page write-up (approach, key decisions and trade-offs, **results vs baseline**, known limitations), and a runnable demo on a standard laptop (no GPU, no cloud).
- **Talk structure:** overview 2 min, live demo 5 min, results vs baseline 5 min, insights 3 min.
- **Red flags:** copy-pasted code you can't explain; 90% data wrangling and 10% modelling; "I would have done X but ran out of time". Frame every gap as a deliberate trade-off with a next step.
- **Problem 2 success criteria:** beat baseline strategies (random, top-salary, last-week) by **15%+**; respect all constraints (budget, positions, team limits); provide **confidence intervals**; an **interactive tool** to explore trade-offs.
- **Problem 2 technical requirements:** regression with uncertainty quantification; integer programming or heuristic search; data pipeline, API integration, interactive dashboard. Bonus: multi-week optimisation, opponent strength, transfer strategy, ensembles or gradient boosting.

### 1.3 Contradictions in the provided documents (decided)
| Conflict | Decision |
|---|---|
| PDF says 1 week; README says Fri 9am → Tue 9am, presentation Monday | Follow the PDF and the real interview date (Fri 30 Oct 2026). |
| README says 33% each for ML, optimisation and engineering; rubric says 25/25/25/25 including presentation | Plan to the rubric. |
| Starter README says "don't waste time on extensive unit tests"; rubric gives 5 pts to testing | Write 5–8 focused pytest tests (see §7). |
| Problem 2 promises "3 seasons of CSV data"; `requirements.txt` is referenced | Neither is in the zip. Use the FPL Historical Dataset (§3) and ship our own `requirements.txt`. |

---

## 2. Why Problem 2, and how to stand out

- **Portfolio framing (say it out loud):** budget = capital; position limits = sector limits; max 3 per club = concentration limit; expected points = expected return; prediction interval = risk; −4 points per extra transfer = transaction cost.
- **Problem 1 (traffic) and Problem 3 (warehouse) were rejected.** Both need a simulator built first. Deadlock-free multi-robot planning is research-grade (CBS search trees grow exponentially with conflicts; RHCR exists for lifelong MAPF), and a live-demo bug is very visible.
- **Industrial-engineering edge** (the owner's background): make the optimisation and decision science the star.
  1. **Robust optimisation:** the Bertsimas & Sim (2004) budget of uncertainty Γ as the risk dial. It stays a MILP.
  2. **Sensitivity analysis:** shadow prices from the LP relaxation (`scipy.optimize.linprog(method="highs")` returns `ineqlin.marginals`), plus re-solving with one constraint relaxed.
  3. **Statistical process control:** an EWMA control chart of weekly prediction error by position. It should flag the 2025-26 scoring change (defensive contributions).
  4. **DMAIC as the talk spine:** Define → Measure (fair baselines, harness) → Analyse (model) → Improve (optimiser, robust dial) → Control (tests, EWMA chart).
- **Opening line for both talks:** *"I'm an industrial engineer: I turn decisions into models with explicit constraints, test them against honest baselines, and keep them under statistical control."*
- **Do not use AHP in Part B** (a quant panel will attack subjective weights and rank reversal). Show the Γ trade-off frontier instead. AHP is used in Part A.

---

## 3. Data

### 3.1 Source (MIT-licensed; cite it)
- **FPL Historical Dataset** by Vaastav Anand: `https://github.com/vaastav/Fantasy-Premier-League` (MIT licence). Citation: Anand, Vaastav (2022). *FPL Historical Dataset*.
- Per-season gameweek file: `https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv`.
- Verified present on 28 Oct 2026: every season `2016-17` … `2025-26` (GW 1–38), and `2026-27` up to GW 6. The README says weekly updates stopped after 2024-25, so expect sporadic updates.
- Use **2023-24, 2024-25 and 2025-26**. Commit the three CSVs (about 5 MB each) under `data/raw/` with attribution so the demo runs offline, and provide `scripts/download_data.py` to refresh them.

### 3.2 Profile (from my checks)
| Season | Rows | GWs | Players | Zero-minute rows | Mean pts | Rows with ≥ 10 pts |
|---|---|---|---|---|---|---|
| 2023-24 | 29,725 | 1–38 | 865 | 61.7% | 1.05 | 1.6% |
| 2024-25 | 27,605 | 1–38 | 804 | 58.1% | 1.20 | 1.8% |
| 2025-26 | 29,757 | 1–38 | 841 | 61.4% | 1.16 | 1.7% |
| 2026-27 | 3,883 | 1–6 | 667 | 52.3% | 1.43 | 2.2% |

**Zero minutes dominate**, which is why the model has two stages (§4).

### 3.3 Columns (2025-26)
`name, position, team, xP, assists, bonus, bps, clean_sheets, creativity, element, expected_assists, expected_goal_involvements, expected_goals, expected_goals_conceded, fixture, goals_conceded, goals_scored, ict_index, influence, kickoff_time, minutes, modified, opponent_team, own_goals, penalties_missed, penalties_saved, red_cards, round, saves, selected, starts, team_a_score, team_h_score, threat, total_points, transfers_balance, transfers_in, transfers_out, value, was_home, yellow_cards, clearances_blocks_interceptions, defensive_contribution, recoveries, tackles, GW`

### 3.4 Cleaning rules (MUST)
1. Keep `position ∈ {GK, DEF, MID, FWD}`. Drop `AM` (2024-25 assistant managers).
2. **Double gameweeks:** aggregate to one row per `(element, GW)`: sum counts, points and minutes; take the first `value`, `position` and `team`. A player missing from a GW had a blank.
3. `value` is **price × 10** (e.g. 55 = £5.5m). Budgets are in the same tenths.
4. **Never use `xP` as a feature.** The dataset README warns it can contain post-match information.
5. **Rule changes:** defensive contributions (DefCon) started in 2025-26. In 2026-27, goalkeeper goals are worth 10 pts (was 6) and the bonus point system was rebalanced. Training seasons (2023-24, 2024-25) have no DefCon points, so expect under-prediction of DEF in the 2025-26 test. State this as a known limitation; component-level modelling is the roadmap fix.

### 3.5 Live API (demo only)
`https://fantasy.premierleague.com/api/bootstrap-static/`, `/api/fixtures/`, `/api/element-summary/{id}/`, `/api/event/{gw}/live/`. Optional "next gameweek" view only; cache the responses; never train on live data.

---

## 4. ML specification

### 4.1 Target and decomposition
`E[points] = P(60+ mins)·E[pts | 60+] + P(1–59 mins)·E[pts | 1–59]`. This mirrors credit's **EL = PD × LGD × EAD**.
- **Stage 1:** a multinomial classifier over {0, 1–59, 60+} minutes (or two binary classifiers). Use `HistGradientBoostingClassifier`.
- **Stage 2:** `HistGradientBoostingRegressor` for points given 60+ minutes (and optionally given 1–59). Fit stage 2 only on rows where the player played.
- **Uncertainty:** quantile models (`loss="quantile"`, `quantile=0.1` and `0.9`) on the conditional points, combined by the stage-1 mixture. Calibrate the final 80% interval with **split conformal / CQR** on a calibration block (the last 8 GWs of 2024-25). Report **empirical coverage on the 2025-26 test**; the target is 80%.

### 4.2 Features (as-of only; built with `shift(1)` inside each player, ordered by GW)
- Rolling means over the last 1, 3, 5 and 10 GWs of: `total_points, minutes, starts, expected_goals, expected_assists, bps` (plus `expected_goals_conceded` and `clean_sheets` for GK and DEF).
- `value` (price as the market's prior), position one-hot, `was_home`.
- Opponent strength: the opponent's rolling goals conceded and xGC over its last 5 matches (as-of).
- Availability proxy: share of the last 5 GWs with 60+ minutes, plus days since the last appearance.
- **Leakage test:** for any row at GW t, every feature equals the value recomputed from rows with GW < t.

### 4.3 Validation (no random splits)
- **Train:** 2023-24 + 2024-25 (minus the calibration block). **Calibrate:** the last 8 GWs of 2024-25. **Test:** every GW of 2025-26 (report GW 6–38 so that features have history).
- **Forecast metrics:** RMSE and MAE overall and by position, plus RMSE by outcome band: Zeros (0 mins), Blanks (≤ 2), Tickers (3–4), Haulers (≥ 5), following OpenFPL's bands.
- **Decision metric:** realised XI points per GW from the chosen team vs each baseline, using the **same solver and constraints**. Report the mean, the % uplift and a **paired bootstrap 95% CI** across test GWs.

---

## 5. Optimisation specification

### 5.1 Single-gameweek XI (matches the starter constraints, which equal FPL's formation rules)
Decision variables: `x_i ∈ {0,1}` (in the XI), `c_i ∈ {0,1}` (captain); `μ_i` = predicted points; `p_i` = price in tenths; `B` = budget.

```
max  Σ μ_i x_i + Σ μ_i c_i
s.t. Σ x_i = 11;   Σ p_i x_i ≤ B;   Σ c_i = 1;   c_i ≤ x_i
     1 ≤ #GK ≤ 1;  3 ≤ #DEF ≤ 5;  2 ≤ #MID ≤ 5;  1 ≤ #FWD ≤ 3
     Σ_{i ∈ club j} x_i ≤ 3   for every club j
```
- **Budget for an XI: £83.5m (835 tenths),** the XI share of £100m used by Ramezani & Dinh (2025). Make it configurable.
- **Solver:** `scipy.optimize.milp` (HiGHS) with **sparse** constraint matrices. My reference solve takes 0.16 s per GW for about 600 players.

### 5.2 Robust risk dial (Bertsimas & Sim 2004)
With `d_i ≥ 0` = player i's downside deviation (μ − q10 from the calibrated interval) and integer `Γ ∈ [0, 11]`:
```
max  Σ μ_i x_i − Γ z − Σ q_i      (q_i are the Bertsimas–Sim protection variables)
s.t. z + q_i ≥ d_i x_i,  z ≥ 0,  q_i ≥ 0,  plus all constraints of §5.1
```
Sweep Γ = 0…11 to draw the **expected vs downside frontier** in the app. My check (2025-26, GW 6–38, naive 5-GW-mean predictor, d = SD of the last 5 GWs): Γ = 4 lifted the worst week from 22 to 27 pts, the 10th percentile from 29.2 to 30.0 and the mean from 42.6 to 43.2. That is 33 GWs, so it is suggestive, not proven. Speed: about 2 s per solve with sparse matrices; prefilter to the top 40 per position by μ if you need it faster.

### 5.3 Sensitivity analysis
- **Budget shadow price:** the dual on the budget row of the LP relaxation, i.e. points per +£0.1m (see `budget_shadow_price`).
- **Constraint cost:** re-solve with the club cap relaxed (cap = 11); report the points difference and how many GWs it binds in.
- My check (naive predictor): the budget never bound (+£1m ≈ 0.01 pts), and the club cap bound in 7 of 33 GWs at 0.2 pts per GW. **Expect different numbers with the real model; explain them.**

### 5.4 Baselines (all via the SAME solver, only the objective changes)
1. **Top-price:** maximise Σ value. 2. **Last week:** μ = last GW's points. 3. **Mean of the last 5 GWs.** 4. **Random feasible:** μ ~ U(0,1), averaged over seeds.

Reference results from my check (2025-26, GW 6–38, £83.5m, **no captain**, realised XI points per GW):

| Strategy | Pts/GW | vs top-price | vs last week |
|---|---|---|---|
| Top-price (feasible) | 32.0 | — | −21.5% |
| Last week | 40.8 | +27.4% | — |
| Mean of last 5 | 42.6 | +33.2% | +4.6% |
| Oracle (hindsight) | 138.3 | +332% | +239% |

**Beating top-price by 15% is easy; beating last week by 15% is the real test.** State which baseline your 15% refers to. On the provided *synthetic* data even a perfect-knowledge oracle beats a feasible top-salary XI by only 8.1% (200 seeds), which is one more reason to use real data.

### 5.5 Bonus (roadmap unless time allows)
Multi-week rolling horizon (3 GWs) with a 15-man squad (2 GK, 5 DEF, 5 MID, 3 FWD, £100m), transfer variables, 1 free transfer (bank up to 5), −4 per extra transfer. Chips (two sets in 2026-27). Opponent model.

---

## 6. Monitoring (Control in DMAIC)
EWMA chart of the weekly mean prediction error by position: λ = 0.2, L = 3, limits = target ± L·σ·√(λ/(2−λ)) (NIST e-Handbook). Plot it in the app. Expect a DEF flag in 2025-26 from the DefCon rule change.

---

## 7. Engineering

### 7.1 Layout
```
.
├── AGENTS.md                # this file
├── README.md                # setup, how to run, results table, write-up, limitations, sources
├── requirements.txt
├── main.py                  # CLI: download | train | backtest | app
├── data/raw/                # merged_gw_{season}.csv (committed, MIT, attributed)
├── scripts/download_data.py
├── src/fplopt/
│   ├── data.py              # load_season, cleaning rules §3.4
│   ├── features.py          # as-of features §4.2
│   ├── model.py             # two-stage model + quantiles + conformal §4
│   ├── optimise.py          # solve_xi, robust, sensitivity §5
│   ├── baselines.py         # §5.4
│   ├── backtest.py          # walk the test GWs, metrics, bootstrap
│   ├── monitor.py           # EWMA §6
│   └── app.py               # Streamlit
├── reports/                 # results.csv, figures (generated)
└── tests/                   # pytest §7.3
```

### 7.2 `requirements.txt` (lower bounds; tested with sklearn 1.9.1, scipy 1.18.1, numpy 2.5.3, pandas 3.0.5)
```
numpy>=1.26
pandas>=2.1
scipy>=1.11
scikit-learn>=1.3
streamlit>=1.30
plotly>=5.18
pytest>=7.4
requests>=2.31
```

### 7.3 Tests (MUST pass before pushing)
1. Every team from `solve_xi` has 11 players, cost ≤ budget, valid position counts, ≤ 3 per club, and a captain inside the XI.
2. Every baseline team satisfies (1).
3. **No leakage:** recompute features for a sample of rows from `GW < t` data only; values must match.
4. An infeasible budget (e.g. £30m) raises `ValueError` with a clear message.
5. Loader: one row per `(element, GW)` after double-gameweek aggregation; positions ⊂ {GK, DEF, MID, FWD}.
6. Conformal: on synthetic data with known noise, empirical coverage is within ±5 pp of the target.
7. Robust: Γ = 0 returns the same team as the non-robust solve.

### 7.4 Streamlit app (10 points: the biggest single item)
Controls: test GW, budget, Γ, formation lock, lock and exclude players. Show: the chosen XI on a pitch or table with μ and the 80% interval; the captain; a **"why this player"** panel (μ, interval, points per £m, points lost if you re-solve without him); the Γ frontier; the backtest chart vs baselines with the CI; the EWMA chart.

### 7.5 Conventions
Python ≥ 3.10; type hints on public functions; pure functions in `src/`; logging, not prints; `numpy.random.default_rng(seed)` with seeds fixed; no notebooks in `src/`; cache processed parquet under `data/processed/` (git-ignored).

### 7.6 Definition of done (Thursday deadline)
- [ ] `python main.py backtest` regenerates `reports/results.csv` and the README results table.
- [ ] `streamlit run src/fplopt/app.py` works offline.
- [ ] All tests pass; README has setup, results vs baselines (with CI), limitations, sources and an AI-use acknowledgement.
- [ ] Repo shared with the reviewers ≥ 24 h before the slot.

---

## 8. Reference implementation (tested 28 Oct 2026)

Copy, adapt and understand it. The owner must be able to explain every line.

```python
"""Reference implementations for AGENTS.md (tested on FPL 2025-26, 28 Oct 2026)."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds, linprog

POSITIONS = ["GK", "DEF", "MID", "FWD"]
XI_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
RAW_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"

def load_season(path_or_season: str) -> pd.DataFrame:
    """One row per (player, gameweek). Double gameweeks are summed; AM rows and blanks dropped."""
    src = RAW_URL.format(season=path_or_season) if path_or_season[:2] == "20" and len(path_or_season) == 7 else path_or_season
    df = pd.read_csv(src, low_memory=False)
    df = df[df["position"].isin(POSITIONS)].copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    num = ["total_points", "minutes", "starts", "goals_scored", "assists", "clean_sheets", "bps",
           "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded"]
    num = [c for c in num if c in df.columns]
    agg = {c: "sum" for c in num}
    agg.update({"value": "first", "position": "first", "team": "first", "name": "first",
                "kickoff_time": "min", "was_home": "first", "opponent_team": "first"})
    g = df.groupby(["element", "GW"], as_index=False).agg(agg)
    return g.sort_values(["element", "GW"]).reset_index(drop=True)

def add_asof_features(g: pd.DataFrame, windows=(1, 3, 5, 10)) -> pd.DataFrame:
    """Rolling means that use ONLY gameweeks before the row's gameweek (shift(1) inside each player)."""
    g = g.sort_values(["element", "GW"]).copy()
    base = [c for c in ["total_points", "minutes", "starts", "expected_goals", "expected_assists", "bps"] if c in g]
    grp = g.groupby("element", group_keys=False)
    for c in base:
        prev = grp[c].shift(1)
        for w in windows:
            g[f"{c}_r{w}"] = prev.groupby(g["element"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
    g["played60"] = (g["minutes"] >= 60).astype(int)          # stage-1 target
    return g

def solve_xi(t: pd.DataFrame, mu: np.ndarray, budget_tenths: int = 835, club_cap: int = 3,
             captain: bool = True, d: np.ndarray | None = None, gamma: float = 0.0):
    """Best XI (+captain) under the starter constraints. Optional Bertsimas-Sim robustness with
    downside deviations d >= 0 and budget of uncertainty gamma. Sparse constraints keep it fast."""
    n = len(t); mu = np.asarray(mu, float)
    robust = gamma > 0 and d is not None
    nx = n; nc = n if captain else 0; nz = 1 if robust else 0; npv = n if robust else 0
    N = nx + nc + nz + npv
    c = np.zeros(N); c[:nx] = -mu
    if captain: c[nx:nx + nc] = -mu
    if robust: c[nx + nc] = gamma; c[nx + nc + 1:] = 1.0
    rows, lo, hi = [], [], []
    def add(row_idx, vals, l, h):
        rows.append((row_idx, vals)); lo.append(l); hi.append(h)
    idx = np.arange(n)
    add(idx, np.ones(n), 11, 11)
    add(idx, t["value"].to_numpy(float), 0, budget_tenths)
    pos = t["position"].to_numpy()
    for p, (a, b) in XI_LIMITS.items():
        m = np.flatnonzero(pos == p); add(m, np.ones(len(m)), a, b)
    for _, m in pd.Series(idx).groupby(t["team"].to_numpy()):
        add(m.to_numpy(), np.ones(len(m)), 0, club_cap)
    data, ri, ci = [], [], []
    for r, (cols, vals) in enumerate(rows):
        ri += [r] * len(cols); ci += list(cols); data += list(vals)
    k = len(rows)
    if captain:
        ri += [k] * n; ci += list(nx + idx); data += [1.0] * n; lo.append(1); hi.append(1); k += 1   # one captain
        for i in range(n):                                                                          # c_i <= x_i
            ri += [k, k]; ci += [nx + i, i]; data += [1.0, -1.0]; lo.append(-np.inf); hi.append(0); k += 1
    if robust:
        z = nx + nc
        for i in range(n):                                                                          # z + p_i - d_i x_i >= 0
            ri += [k, k, k]; ci += [z, z + 1 + i, i]; data += [1.0, 1.0, -float(d[i])]; lo.append(0); hi.append(np.inf); k += 1
    A = sparse.csr_matrix((data, (ri, ci)), shape=(k, N))
    integ = np.r_[np.ones(nx + nc), np.zeros(nz + npv)]
    ub = np.r_[np.ones(nx + nc), np.full(nz + npv, np.inf)]
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(0, ub))
    if res.status != 0:
        raise ValueError(f"No feasible XI: {res.message}")
    x = res.x[:nx] > 0.5
    cap = int(np.argmax(res.x[nx:nx + nc])) if captain else None
    return np.flatnonzero(x), cap

def budget_shadow_price(t, mu, budget_tenths=835):
    """LP relaxation of the XI problem (no captain): dual on the budget row = points per 0.1m."""
    n = len(t); A_ub, b_ub = [], []
    A_ub.append(t["value"].to_numpy(float)); b_ub.append(budget_tenths)
    pos = t["position"].to_numpy()
    for p, (a, b) in XI_LIMITS.items():
        m = (pos == p).astype(float); A_ub.append(m); b_ub.append(b); A_ub.append(-m); b_ub.append(-a)
    for team in t["team"].unique():
        A_ub.append((t["team"].to_numpy() == team).astype(float)); b_ub.append(3)
    r = linprog(-np.asarray(mu, float), A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.ones((1, n)), b_eq=[11],
                bounds=(0, 1), method="highs")
    return -r.ineqlin.marginals[0]          # >= 0: extra predicted points per +0.1m of budget

def paired_bootstrap(a, b, n_boot=5000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float); rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n_boot, len(d))).mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def ewma_chart(errors, lam=0.2, L=3.0, target=0.0, sigma=None):
    """EWMA of weekly mean error; limits target +/- L*sigma*sqrt(lam/(2-lam)) (NIST handbook, steady state)."""
    e = np.asarray(errors, float); sigma = np.std(e, ddof=1) if sigma is None else sigma
    z = np.empty_like(e); prev = target
    for i, v in enumerate(e):
        prev = lam * v + (1 - lam) * prev; z[i] = prev
    half = L * sigma * np.sqrt(lam / (2 - lam))
    return z, target - half, target + half, (z < target - half) | (z > target + half)

def split_conformal(y_cal, lo_cal, hi_cal, alpha=0.2):
    """Conformalised quantile regression (CQR) correction: widen [lo, hi] by q to reach 1-alpha coverage."""
    s = np.maximum(np.asarray(lo_cal) - y_cal, np.asarray(y_cal) - hi_cal)
    n = len(s); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(s)[min(k, n) - 1]
```

Smoke-test output on 2025-26 data: 29,338 rows after aggregation, zero-minute share 0.613; an XI at GW 20 had 11 players, cost 72.4 of 83.5, MID 5 / DEF 3 / FWD 2 / GK 1, max 3 per club, captain in the XI, solved in 0.16 s; the robust Γ = 4 solve took 2.1 s; an infeasible £30m budget raised `ValueError`.

---

## 9. Starter code bugs (mention in the README; it shows rigour)

| File | Bug | Evidence |
|---|---|---|
| `problem2_fantasy/fantasy_optimizer.py` `BaselineStrategy` | Baselines ignore the constraints | Top-salary XI costs 130 vs a budget of 100; last-week XI picks 4 GKs; 69 of 2,000 random teams feasible |
| same, `load_historical_data` | Points are Poisson(5), independent of everything | corr(salary, pts) = −0.015 |
| same, `TeamOptimizer` / `explain_selection` | Random pick; reports week-1 points, not predictions | 11 of 200 placeholder teams feasible; the demo run picked 4 FWD |
| `synthetic_data.generate_player_stats` | "Form" is i.i.d. noise; salary tracks hidden quality | Lag-1 autocorrelation −0.12; corr(salary, quality) 0.82 |
| `synthetic_data.generate_multi_seasonal` | Comment says the peak is at 17:00, but it peaks 12 h after run start (`datetime.now()`) | A run started in the 10:00 hour peaked at 22:00 |
| repo | `requirements.txt` referenced but missing | — |

---

## 10. Part A (creative pitch): context, in case an agent edits the deck or notes

**Concept: Mycelium, credit that grows through stokvel networks** (biomimicry plus Blue Ocean plus Lean). The deck has 5 slides + 4 appendix slides with speaker notes (`Mycelium_PartA_pitch_v2.pptx`).

- **Inspiration:** mycorrhizal fungi trade phosphorus for plant carbon in a "biological market"; both sides reward fair partners and sanction hoarders (**Kiers et al. 2011, *Science* 333:880**). Do NOT use the "mother tree" claim, which is disputed (**Karst, Jones & Hoeksema 2023**; Simard rebutted it in 2025).
- **Problem:** ~11m stokvel members in ~800,000 groups save >R50bn a year (NASASA estimates, unaudited). Stokvel loans cost **25–35% a month** (Ngcobo & Chisasa 2018, survey of 386 members). The NCA caps short-term credit at **5% a month for a first loan and 3% after**.
- **Red ocean:** Absa, Capitec (launched 10 Sept 2026), FNB (~120,000 clubs, fully digital from Feb 2026), Nedbank and Standard Bank all compete on deposit features. **None offers member credit; neither does the Stokfella app.**
- **Mechanism:** ledger → member limit sized by group reliability → draw and repay from contributions → graduate to individual credit. **Design rules:** individual liability (Giné & Karlan: no more default, more clients); group data only raises a score (Óskarsdóttir et al.); 12 months of history plus KYC for every member.
- **ERRC:** Eliminate rate-chasing and in-branch onboarding; Reduce extras and bureau reliance; Raise member view and speed to cash; Create a credit file from contributions and reliability-based pricing.
- **Why now:** deposits are commoditising (Capitec, FNB); PayShap pays out in seconds across 12 banks (>R100bn); WEF/CARE (30 Jun 2026) say exclusion is "a problem of legibility, not behaviour".
- **Precedents:** Mission Asset Fund (US) lending circles: average credit-score change +168 vs +41 for controls, including people who started with no score. Equity Bank chama loans (Kenya), with groups registered ≥ 1 year. CARE Uganda pilot: NPL < 2%; savings groups repay > 95%.
- **Unit economics (illustrative):** R2,000 over 3 months earns R180 (3%/month) to R300 (5%/month) of interest, less R35 funding (~7% a year) and R100 expected loss (5%), leaving **R45–R165 before servicing**. The prize is ~34 customers per group, retained deposits and data.
- **Lean:** a value stream (today: wait, 25–35% a month, record wasted; Mycelium: pull a limit, repay from contributions, build a credit file) and a 90-day build–measure–learn MVP. Guardrails: cap per group, entity-resolution checks for fake stokvels, NCA Reg 23A affordability, POPIA consent, IFRS 9 staging per member.
- **AHP (draft judgements):** Mycelium 0.40 > early-warning score 0.25 > scam defence 0.18 > behaviour pricing 0.17; every CR ≤ 0.013; Mycelium is still first with equal weights.
- **Group-limit LP:** max Σ m(g)L(g) s.t. Σ L(g) ≤ B; Σ PD(g)·LGD·L(g) ≤ E; 0 ≤ L(g) ≤ n(g)·ℓ·r(g).

---

## 11. Panel Q&A prep

| Q | Answer in one line |
|---|---|
| Why two stages? | 58–62% of rows are zero minutes; it's a mixture, just like PD × LGD. |
| Leakage? | As-of features, a test enforces them, `xP` excluded, walk-forward splits. |
| Is the uplift real? | Paired bootstrap CI across test GWs, against a named baseline under the same constraints. |
| Why MILP, not greedy? | Provably optimal and fast; show the points greedy loses. |
| Are the intervals calibrated? | Show empirical coverage vs the 80% target. |
| What does Γ mean? | How many players may hit their downside at once; the frontier shows its price. |
| Which constraint drives the team? | Shadow prices and re-solves (§5.3). |
| How would it scale? | Linear in players for one week; multi-week rolls the horizon; sparse matrices. |
| Bank analogue? | Expected return, sector and single-name limits, transaction costs, CVaR. |
| What's next? | Multi-week transfers, component scoring for rule changes, opponent model, chips. |

---

## 12. Sources (all opened 28 Oct 2026)
- FPL Historical Dataset (MIT): https://github.com/vaastav/Fantasy-Premier-League
- FPL 2026/27 rules: https://www.beatfpl.com/guides/fpl-26-27-rules-changes · https://www.chaseyoursport.com/fantasy-football/complete-beginners-guide-fantasy-premier-league-2026-27/12878
- FPL API endpoints: https://glama.ai/mcp/servers/@owen-lacey/fpl-mcp/blob/e9171d6bb4bc00c522d02752962554112a340d30/docs/fpl-api-cheatsheet.md
- Groos (2025), OpenFPL, arXiv:2508.09992: https://arxiv.org/html/2508.09992v1
- Ramezani & Dinh, arXiv:2505.02170: https://arxiv.org/html/2505.02170v3
- Hunter, Vielma & Zaman, arXiv:1604.01455: https://arxiv.org/abs/1604.01455v3
- Bertsimas & Sim (2004), *Operations Research* 52(1):35–53: https://ideas.repec.org/a/inm/oropre/v52y2004i1p35-53.html
- SciPy `linprog` (HiGHS) marginals: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html
- NIST EWMA chart: https://itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm
- MAPIE (optional): https://pypi.org/project/MAPIE
- Sharon et al. (2015), CBS: https://people.engr.tamu.edu/guni/pistar/Papers/CBS-AIJ15.pdf · Li et al., RHCR: https://arxiv.org/abs/2005.07371v2 · Varaiya, max-pressure: https://people.eecs.berkeley.edu/~varaiya/publications.html
- GitHub Copilot instruction-file support: https://docs.github.com/en/copilot/reference/custom-instructions-support
- Part A: Kiers et al. 2011 (ScienceDaily): https://sciencedaily.com/releases/2011/08/110821210224.htm · Karst et al. 2023 coverage: https://scienceblog.com/t-suzanne-simards-1997-experiment-really-did-find-carbon-moving-between-tree-species-through-shared-fungal-threads-but-the-much-more-famous-claim-that-grew-from-it-that-mature-trees-deliberately-send/ · Ngcobo & Chisasa 2018: https://ojs.amhinternational.com/index.php/jebs/article/view/2614 · UNSGSA (NASASA, Ipsos): https://www.unsgsa.org/stories/embedding-trust-stokvels-and-nedbank-build-culture-financial-health · Capitec: https://www.africaninsider.com/business/billionaire-michiel-le-rouxs-capitec-moves-into-south-africas-lucrative-r50bn-stokvel-market/ · FNB: https://www.timeslive.co.za/news/business/2026-02-12-digital-stokvels-boost-safety-accessibility/ · Absa: https://absa.co.za/personal/save-invest/products/club-account · Standard Bank: https://www.standardbank.co.za/southafrica/personal/products-and-services/grow-your-money/savings-and-investment/our-accounts/society-scheme-savings-account · NCA caps: https://mjkinc.co.za/finance-credit-law/interest-and-the-in-duplum-rule · NCA Reg 23A: https://www.polity.org.za/article/national-credit-act-new-affordability-assessment-regulations-2016-02-15 · PayShap: https://stitch.money/blog/real-time-payments-in-south-africa-the-state-of-payshap-in-2026 · WEF/CARE: https://www.weforum.org/stories/financial-and-monetary-systems/informal-economies-solutions-recognition/ · Mission Asset Fund: https://www.bostonfed.org/publications/communities-and-banking/2016/winter/social-lending-improving-credit-for-the-underbanked.aspx · Giné & Karlan: https://www.povertyactionlab.org/evaluation/group-versus-individual-liability-philippines · Óskarsdóttir et al.: https://arxiv.org/abs/2002.09931v1 · Equity Bank: https://hapakenya.com/2025/07/09/how-chamas-smes-can-access-loans-from-equity-bank-through-pamoja-banking · Stokfella: https://www.cbinsights.com/company/stokfella · Blue Ocean ERRC: https://www.blueoceanstrategy.com/?p=266408 · Lean principles: https://www.lean.org/lexicon-terms/lean-thinking-and-practice/ · Lean Startup (Ries, LSE slides): https://www.lse.ac.uk/assets/richmedia/channels/publicLecturesAndEvents/slides/20120112_1830_theLeanStartup_sl.pdf · Biomimicry design process: https://toolbox.biomimicry.org/methods/process/

## 13. AI-use acknowledgement (paste into the README)
> Research, scoping and reference snippets were assisted by Claude (Anthropic) and GitHub Copilot. All numbers were checked against the cited sources and re-run locally. Design decisions, final code and conclusions are my own.
