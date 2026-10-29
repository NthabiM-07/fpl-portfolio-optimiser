# FPL team selection as constrained portfolio optimisation

**Problem 2 (Fantasy Sports Team Optimiser).** A two-stage machine-learning model predicts each player's points with a calibrated interval; an exact mixed-integer programme picks the best legal XI and captain; a walk-forward backtest over every gameweek of the **2025-26 Premier League season** scores the picks against baselines that obey the same rules.

> **Headline (2025-26, all 38 gameweeks, out of sample):** the model's XI scored **58.5 points per gameweek**. That is **+25.5% vs last week's points** (paired bootstrap 95% CI +5.2 to +18.3 pts/GW), **+25.9% vs the 5-week average** (CI +6.2 to +17.6), **+73.5% vs the most expensive XI** and **+319% vs a random legal XI**. The 15% target is cleared against every baseline, with confidence intervals that exclude zero.

![Mean points per gameweek](reports/figures/mean_points.png)

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py app            # interactive demo, runs offline from reports/ (about 10 seconds)
pytest -q                     # 12 tests
python main.py backtest       # optional: rebuild everything from raw data (about 10 minutes on a laptop)
python main.py pick --gw 20 --gamma 4     # print a robust XI for one gameweek
```

Python 3.10+; no GPU, no internet needed (the four season files are committed under `data/raw/`).

---

## 1. Framing: team selection is portfolio construction

| Fantasy football | Portfolio |
|---|---|
| Expected points | Expected return |
| 80% prediction interval | Risk |
| GBP 83.5m budget for the XI | Capital |
| 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD | Sector limits |
| At most 3 players per club | Concentration limit |
| Captain scores double | One leveraged position |

So the work splits into a **forecasting** problem (expected return and risk per player) and an **allocation** problem (choose the portfolio under constraints). Both are evaluated on what actually happened.

## 2. Data

- **FPL Historical Dataset** (Vaastav Anand, MIT licence): gameweek-level stats for every player, seasons **2022-23 to 2025-26** (109,956 player-gameweeks after cleaning). Columns include minutes, starts, goals, assists, xG, xA, xGC, BPS, price, ownership, transfers, fixture and opponent.
- Cleaning (`src/fplopt/data.py`): drop the 2024-25 assistant-manager rows; sum double gameweeks into one row per player per gameweek; derive each fixture's opponent; build a cross-season player key so a player's form carries over the summer.
- **Never used as a feature:** the dataset's `xP` column, which can contain post-match information.

## 3. Features: strictly as-of

Every feature for gameweek *t* uses only information available before the *t* deadline (`src/fplopt/features.py`):

- Player form: rolling means over the previous 1, 3, 5 and 10 gameweeks of points, minutes, starts, xG, xA, xGC, BPS, clean sheets and saves; share of recent games with 60+ minutes; points per 90; gameweeks since the last appearance; season-to-date and previous-season averages.
- Market signals: current price (set before the deadline), price change, and **lagged** ownership and net transfers.
- Fixture context: number of fixtures (double gameweeks), home or away, own team's and opponent's rolling goals and xG for and against (aggregated per gameweek, so a double gameweek's second match cannot leak into the first).
- A **scoring-regime flag** for 2025-26, when defensive-contribution points were introduced.

A test overwrites every outcome from gameweek 20 onward and asserts that no feature up to gameweek 20 changes (`tests/test_data_features.py`).

## 4. Model: will he play, and how much will he score if he does?

61% of player-gameweeks are zero-minute rows. A single regressor has to average two very different populations, so the model is split, mirroring credit's **EL = PD x LGD x EAD**:

```
E[points] = P(1-59 min) x E[pts | 1-59 min]  +  P(60+ min) x E[pts | 60+ min]
```

- Stage 1: a gradient-boosted classifier over minutes buckets {0, 1-59, 60+}.
- Stage 2: gradient-boosted regressors for points in each playing bucket.
- **Intervals:** quantile gradient boosting at 10% and 90%, then a conformalised quantile regression (CQR) correction computed from the most recent 8 gameweeks of *out-of-sample* predictions (seeded by a 2024-25 warm-up).
- **Why gradient boosting (scikit-learn `HistGradientBoosting`):** it handles missing values natively (new players, promoted teams), captures non-linear interactions (price x minutes x fixture), trains in seconds on a laptop, and needs no scaling. The ablation below justifies the two-stage design over a single model.

## 5. Optimisation

**Nominal MILP** (`src/fplopt/optimise.py`), solved exactly with SciPy's HiGHS in about 0.1 s per gameweek for about 750 players:

```
max   sum_i mu_i x_i + sum_i mu_i c_i
s.t.  sum_i x_i = 11;  sum_i price_i x_i <= B;  sum_i c_i = 1;  c_i <= x_i
      1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD;  sum_{i in club j} x_i <= 3 for every club j
```

**Robust dial (Bertsimas & Sim).** Each player can fall short of his expected points by d_i (the distance to the bottom of his 80% interval). The robust team maximises points after the worst **Gamma** shortfalls:

```
max_x  sum_i mu_i x_i (+ captain)  -  max_{|S| <= Gamma} sum_{i in S} d_i x_i
```

The direct robust MILP is slow (more than 7 s per solve at Gamma = 4). Bertsimas & Sim (2003) prove it equals a set of ordinary MILPs with adjusted scores, `max_theta [ -Gamma*theta + nominal(mu_i - max(d_i - theta, 0)) ]`, and Lee & Kwon (2014) show about half of the theta values suffice. The code solves that sequence (exactly, or on a theta grid for speed), and a test confirms it matches the direct MILP.

**Sensitivity:** the budget's shadow price comes from the LP relaxation's dual values; the cost of the club cap comes from re-solving with the cap removed.

## 6. Validation

- **Walk-forward, expanding window:** retrain every 4 gameweeks on all data before the block; predict the block; never random splits.
- **Fair baselines:** every strategy goes through the same MILP with the same budget, formation, club cap and captain rule; only the score changes. (The starter code's baselines broke the budget and position rules, so comparing against them would have been meaningless.)
- **Decision metric:** realised XI points (captain doubled) per gameweek. **Significance:** paired bootstrap over gameweeks.
- **Pre-declared views:** the primary view is all 38 gameweeks; the secondary view is gameweeks 6-38 (both reported below, whatever they showed).

## 7. Results (2025-26, out of sample)

### Team selection

| Strategy (same MILP, same rules) | Points per GW | vs last week | vs mean of last 5 | vs most expensive |
|---|---|---|---|---|
| **Two-stage model** | **58.5** | **+25.5%** | **+25.9%** | **+73.5%** |
| Single-GBM model (ablation) | 53.1 | +14.1% | +14.4% | +57.6% |
| Mean of last 5 gameweeks | 46.4 | -0.3% | 0.0% | +37.8% |
| Last week's points | 46.6 | 0.0% | +0.3% | +38.2% |
| Most expensive XI | 33.7 | -27.6% | -27.4% | 0.0% |
| Random legal XI (10 seeds) | 13.9 | -70.1% | -70.0% | -58.6% |
| Hindsight oracle (upper bound) | 155.0 | | | |

Gameweeks 6-38 only: model 58.3 per GW, +25.7% vs last week, +24.5% vs mean of last 5.

| Model minus baseline | Mean pts/GW | 95% CI (paired bootstrap) | GWs model ahead |
|---|---|---|---|
| Last week | +11.9 | +5.2 to +18.3 | 25 / 38 |
| Mean of last 5 | +12.0 | +6.2 to +17.6 | 27 / 38 |
| Most expensive XI | +24.8 | +18.5 to +31.0 | 33 / 38 |
| Single GBM (ablation) | +5.3 | +0.8 to +10.0 | 23 / 38 |

![Cumulative points](reports/figures/cumulative_points.png)

### Forecast accuracy

| Forecast | RMSE | MAE | RMSE when played | RMSE on 5+ pt hauls |
|---|---|---|---|---|
| Two-stage model | **1.938** | **0.978** | **2.970** | **5.592** |
| Single GBM | 1.942 | 0.990 | 2.974 | 5.605 |
| Mean of last 5 | 2.144 | 1.066 | 3.244 | 5.754 |
| Last week | 2.613 | 1.163 | 4.034 | 6.187 |

The two models are close on RMSE, but the two-stage model picks better teams (+5.3 pts/GW, CI excludes zero). Its errors are smaller where selection happens, on likely starters and on hauls.

### Prediction intervals (target 80%)

| Group | Coverage | Mean width (pts) |
|---|---|---|
| Players who played | **80.8%** | 5.5 |
| All rows (incl. 0 minutes) | 91.8% | 2.6 |

For players who play, the interval hits its 80% target. Over all rows it over-covers, because a zero-minute player scores exactly 0, which every interval with a lower bound of 0 contains.

### Robust dial (Gamma)

![Robust dial](reports/figures/robust_frontier.png)

| Gamma | Mean pts/GW | 10th-percentile week |
|---|---|---|
| 0 | 58.5 | 33.7 |
| 3 | 57.7 | 37.1 |
| **4** | **58.2** | **37.7** |
| 8 | 56.8 | 35.5 |

At Gamma = 4, the weakest weeks improve by about 4 points while the mean falls by about 0.3. That is cheap insurance, though 38 weeks is too few to call it proven.

### Sensitivity

- The budget binds in 21 of 38 gameweeks, but its median shadow price is small: 0.04 expected points per extra GBP 1m.
- The 3-per-club rule binds in 11 gameweeks, costing 0.85 expected points on average and about 15 in the worst gameweek, when one club had four strong picks.
- The model's XI cost GBP 80.5m on average against the GBP 83.5m budget.

### Monitoring (statistical process control)

![EWMA](reports/figures/ewma_monitor.png)

An EWMA control chart (lambda = 0.2, 3-sigma limits) of weekly prediction error **flags defenders from gameweek 3**: the model under-predicts them because defensive-contribution points started in 2025-26 and the training seasons don't have them. The flag clears as walk-forward retraining absorbs 2025-26 data and the regime flag. The other positions stay in control.

## 8. Design decisions and trade-offs

- **Real data over synthetic.** On the provided synthetic data, even a perfect-knowledge oracle beats a legal top-salary XI by only about 8%, so the 15% target was unreachable there by construction.
- **XI + captain, single gameweek.** This matches the starter constraints and isolates forecast quality. The 15-man squad with transfers (a multi-period rebalancing problem with -4 per extra transfer) is the next step.
- **Gradient boosting over deep learning:** tabular data, missing values, laptop-scale training, explainable feature effects.
- **Robust via decomposition, not a bigger solver:** exact theory, fast, and testable against the direct formulation.
- **Robust sweep on a prefiltered pool** (top 60 per position by expected points) with a 32-point theta grid, for speed; the app uses the full pool.

## 9. Known limitations

- One test season (38 decisions); uplifts carry the wide CIs shown.
- Candidates are players with a fixture in that gameweek, and prices are taken at that gameweek; both are known before the deadline. No bench, vice-captain or auto-substitutions.
- Ownership and transfer signals are lagged one gameweek to stay safe, which likely leaves some injury information unused.
- Points are predicted in total, not by component (goals, clean sheets, defensive contributions), so a rule change is learned only after it appears in the data. Component modelling is the roadmap fix the EWMA chart points to.

## 10. Engineering

- `src/fplopt/`: `data`, `features`, `model`, `optimise`, `baselines`, `backtest`, `monitor`, `reports`, `app` (Streamlit).
- `tests/` (pytest, 12 tests): every team obeys every rule; every baseline obeys every rule; an infeasible budget raises a clear error; Gamma = 0 equals the risk-neutral team; the robust team cuts the worst-case exposure; locks and exclusions hold; the decomposition matches the direct robust MILP; CQR reaches its target coverage; the EWMA flags a persistent shift; one row per player-gameweek; no feature sees the future; rolling means use earlier rows only.
- **Scalability:** sparse constraint matrices; a single-gameweek solve grows linearly with players (about 0.1 s for 750); walk-forward training is about 15 s per block; the full rebuild takes about 10 minutes on two CPU cores. A multi-week horizon would roll, keeping each MILP small.

## 11. Starter-code issues found (and avoided)

- The baselines ignore the constraints (the top-salary XI costs 130 against a budget of 100; the last-week XI picks 4 goalkeepers; only 69 of 2,000 random teams were feasible).
- `load_historical_data` returns Poisson(5) points unrelated to anything.
- In `generate_player_stats`, form is independent noise, so recent form carries no signal.
- `generate_multi_seasonal` peaks 12 hours after the run starts, not at 17:00.
- `requirements.txt` is referenced but missing.

## 12. Roadmap

Multi-week squad and transfer planning (rolling horizon); component-level scoring to absorb rule changes; an opponent-strength model; chips; same-gameweek transfer and ownership signals once their timing is confirmed.

## Sources

- Anand, V. *FPL Historical Dataset* (MIT): https://github.com/vaastav/Fantasy-Premier-League
- Bertsimas, D. & Sim, M. (2004). The price of robustness. *Operations Research* 52(1):35-53.
- Bertsimas, D. & Sim, M. (2003). Robust discrete optimization and network flows. *Mathematical Programming* 98:49-71.
- Lee, T. & Kwon, C. (2014). A short note on the robust combinatorial optimization problems with cardinality constrained uncertainty. *4OR* 12(4):373-378.
- Romano, Y., Patterson, E. & Candès, E. (2019). Conformalized quantile regression. NeurIPS 2019, arXiv:1905.03222.
- NIST/SEMATECH e-Handbook, EWMA control charts: https://itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm
- Groos (2025), OpenFPL: https://arxiv.org/html/2508.09992v1 · Ramezani & Dinh, team selection in FPL: https://arxiv.org/html/2505.02170v3 · Hunter, Vielma & Zaman, integer programming for fantasy sports: https://arxiv.org/abs/1604.01455v3
- FPL 2026/27 rules: https://www.beatfpl.com/guides/fpl-26-27-rules-changes

**AI use.** Research, scoping and code drafting were assisted by Claude (Anthropic). All results were re-run locally from the committed data; design decisions, final code and conclusions are my own.
