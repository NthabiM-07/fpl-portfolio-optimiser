# FPL team selection as constrained portfolio optimisation

**Nthabiseng Maoka** · [![tests](https://github.com/NthabiM-07/fpl-portfolio-optimiser/actions/workflows/tests.yml/badge.svg)](https://github.com/NthabiM-07/fpl-portfolio-optimiser/actions/workflows/tests.yml) · Python 3.11–3.13 · macOS, Windows, Linux · MIT licence

A forecasting and optimisation system for Fantasy Premier League (FPL). A two-stage machine-learning model predicts each player's points with a calibrated interval; an exact mixed-integer programme (MILP) picks the best legal team and captain; a walk-forward backtest over every gameweek of the **2025-26 Premier League season** scores the picks against baselines that obey the same rules.

> **Result (2025-26, all 38 gameweeks, out of sample):** the model's team scored **58.5 points per gameweek**, **+25.5% vs picking on last week's points** (paired bootstrap 95% CI +5.2 to +18.3 points a week) and **+25.9% vs the 5-week average** (CI +6.2 to +17.6). Its forecast error was lower than both naive rules in **all 38 gameweeks** (Diebold-Mariano, p < 0.0001).

![Mean points per gameweek](reports/figures/mean_points.png)

**Highlights**
- **Decision-focused evaluation.** Market and fixture signals barely move RMSE, yet scrambling them costs the picked team 7–8 points a week: forecasts are judged by the decisions they drive.
- **Calibrated robust optimisation.** Conformal intervals feed a Bertsimas-Sim robustness budget whose size is set from past data by a rule fixed in advance.
- **Leakage forensics.** FPL's own `xP` column was tested rather than trusted; it carries post-match information and is excluded.
- **The full game.** A rolling-horizon MILP manages a 15-man squad with transfers, hits and automatic substitutions over the season.
- **Engineering.** 37 unit and property-based tests, CI on three operating systems, pinned versions that reproduce every committed result.

## Quick start

Python 3.11–3.13; no GPU, and no internet after installing (four seasons of data are committed under `data/raw/`).

**macOS or Linux**

```bash
git clone https://github.com/NthabiM-07/fpl-portfolio-optimiser.git && cd fpl-portfolio-optimiser
make setup     # creates .venv with the exact tested versions (requirements-lock.txt)
make test      # 37 tests, about 1.5 minutes
make app       # interactive demo in the browser
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/NthabiM-07/fpl-portfolio-optimiser.git; cd fpl-portfolio-optimiser
py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
python -m pytest -q
python main.py app
```

| Command | What it does |
|---|---|
| `python main.py app` | Interactive demo (Streamlit), offline |
| `python main.py pick --gw 20 --gamma 4` | Print a robust team for one gameweek |
| `python main.py backtest` | Rebuild every result from the raw data (~10 min) |
| `python main.py explain` | Decision-focused signal importance (~3 min) |
| `python main.py season` | Full-rules season with transfers (~3 min) |
| `python main.py mcp` | Serve the optimiser to an AI assistant over MCP |

**Reproducibility.** Random steps are seeded and every split is by time. On a fresh clone with `requirements-lock.txt`, `backtest` reproduces every committed results table exactly; other library versions can differ in the last decimals. No local setup: *Code → Codespaces* opens the demo in the browser.

## Approach

**Framing.** Team selection is portfolio construction: forecast return and risk for each player, then allocate under constraints.

| Fantasy football | Portfolio |
|---|---|
| Expected points | Expected return |
| 80% prediction interval | Risk |
| GBP 83.5m budget for the XI | Capital |
| 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD | Sector limits |
| At most 3 players per club | Concentration limit |
| Captain scores double | One leveraged position |

**Data.** The FPL Historical Dataset (Vaastav Anand, MIT licence), seasons 2022-23 to 2025-26: 109,956 player-gameweeks after summing double gameweeks. Features use only information available before each deadline: rolling form over 1–10 gameweeks (points, minutes, xG, xA, BPS and more), price and lagged ownership, and fixture strength. A test overwrites every outcome from gameweek 20 onward and checks that no earlier feature changes.

**Model.** About 60% of player-gameweeks are zero minutes, so the model has two stages, like PD × LGD in credit:

```
E[points] = P(1-59 min) · E[pts | 1-59 min]  +  P(60+ min) · E[pts | 60+ min]
```

Gradient boosting (scikit-learn `HistGradientBoosting`) fits the minutes classifier and the conditional points regressors. Quantile models give the 10% and 90% bounds, checked by Mondrian conformal calibration (CQR) on recent out-of-sample weeks, separately for likely starters and everyone else.

**Optimisation.** An exact MILP (SciPy/HiGHS, about 0.1 s for ~770 candidates) picks 11 players and a captain:

```
max   Σ μ_i x_i + Σ μ_i c_i
s.t.  Σ x_i = 11;  Σ price_i x_i ≤ B;  Σ c_i = 1;  c_i ≤ x_i
      1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD;  at most 3 per club
```

The robust version maximises points after the worst Γ shortfalls, where each player's shortfall is the distance to the bottom of his interval. It is solved by the Bertsimas & Sim decomposition into ordinary MILPs on a θ grid (about 1.5 s), with each candidate team scored by its true worst case; property-based tests check it against the direct robust MILP.

**Validation.** Expanding-window walk-forward: retrain every 4 gameweeks and predict the next block, never random splits. Every baseline goes through the same MILP and rules; only the scores change. Significance is a paired bootstrap over gameweeks. Gameweeks 6–38 are a pre-declared secondary view.

## Results (2025-26, out of sample)

### Team selection

| Strategy (same MILP, same rules) | Points per GW | vs last week | Model minus this, 95% CI |
|---|---|---|---|
| **Two-stage model** | **58.5** | **+25.5%** | |
| Single gradient-boosting model (ablation) | 53.1 | +14.1% | +5.3 (+0.8 to +10.0) |
| Last week's points | 46.6 | 0.0% | +11.9 (+5.2 to +18.3) |
| Mean of last 5 gameweeks | 46.4 | −0.3% | +12.0 (+6.2 to +17.6) |
| Most expensive XI | 33.7 | −27.6% | +24.8 (+18.5 to +31.0) |
| Random legal XI | 13.9 | −70.1% | |
| Hindsight oracle (upper bound) | 155.0 | | |

Captain doubled, GBP 83.5m XI. On gameweeks 6–38 the gains over both naive rules stay significant (+4.2 to +19.4 vs last week); the gain over the ablation does not (−1.5 to +8.5).

![Cumulative points](reports/figures/cumulative_points.png)

### Forecast accuracy

| Forecast | RMSE | MAE | Model has lower error in |
|---|---|---|---|
| Two-stage model | **1.938** | **0.978** | |
| Single gradient-boosting model | 1.942 | 0.990 | 22 of 38 GWs (p = 0.37) |
| Mean of last 5 | 2.144 | 1.066 | 38 of 38 (p < 0.0001) |
| Last week | 2.613 | 1.163 | 38 of 38 (p < 0.0001) |

Diebold-Mariano tests on gameweek-averaged squared error with a Newey-West variance. The ablation is the interesting case: squared error cannot separate it from the model, yet its teams score 5.3 points a week less. 72% of squared error comes from the 8.6% of rows that are large hauls nobody can foresee, while the optimiser only uses the ranking at the top.

### Which signals drive the decisions

![Signal value](reports/figures/signal_value.png)

Each feature family is shuffled within gameweeks (10 times, frozen model trained before gameweek 20) and scored on gameweeks 20–38:

| Signal family | RMSE added | Team points lost per GW |
|---|---|---|
| Playing time (minutes, starts, availability) | 0.449 | 12.3 |
| Market (price, ownership, transfers) | 0.046 | 8.4 |
| Fixture (opponent strength, home, double gameweeks) | 0.043 | 7.0 |
| Underlying stats (xG, xA, clean sheets, saves) | 0.002 | 2.0 |
| Points form | 0.004 | −0.6 |

Market and fixture signals matter far more to the decision than RMSE suggests, because they separate the top-end players the optimiser chooses between (Elmachtoub & Grigas 2022). This measures how much the model relies on a family, not its causal value.

### Leakage forensics: FPL's `xP` column

FPL's published expected points beat every pre-match forecast on the same week. Two checks on the 7,452 player-gameweeks where `xP` is filled in (`src/fplopt/audit.py`):

| Forecast for week t | Corr. with week t points | Corr. with week t+1 points | Edge kept |
|---|---|---|---|
| Two-stage model | 0.557 | **0.527** | 95% |
| FPL `xP` | **0.734** | 0.478 | 65% |

After removing everything the pre-match model explains, `xP`'s remainder correlates 0.43 with this week's bonus points and 0.04 with next week's. A pre-match forecast cannot know who gets bonus, so `xP` is used neither as a feature nor as a benchmark.

### Risk: intervals, robustness and constraints

- **Interval coverage** (target 80%): 84.6% for likely starters, 80.8% for players who played (`reports/interval_coverage.csv`).
- **Robustness budget Γ from data.** Each week Γ is set to the 80th percentile of the downside budget the team actually used in earlier weeks (it settled on 5–6). It costs 0.9 points a week on average (CI −2.9 to +1.1, not significant), raises the 10th-percentile week from 33.7 to 36.4, and the robust floor held in 31 of 34 weeks. No setting protects the worst week. The full Γ sweep is in `reports/robust_frontier.csv`.
- **Constraints.** The budget binds in 21 of 38 gameweeks but is worth only ~0.09 points per extra GBP 1m. The club cap binds in 11 gameweeks, costing 2.9 points when it does; in double gameweek 26 the uncapped team would have picked 9 Arsenal players.

### Monitoring

![EWMA](reports/figures/ewma_monitor.png)

An EWMA control chart of weekly prediction error flags defenders from gameweek 3: the 2025-26 defensive-contribution points did not exist in the training seasons. The flag clears as retraining absorbs the new season.

### The full game: a 15-man squad through the season

A rolling-horizon MILP (`src/fplopt/planner.py`) plans the next H weeks for a 15-man squad within GBP 100m, with one free transfer a week (bankable to 5), −4 points per extra transfer and automatic substitutions, and makes only this week's moves. Horizon and transfer penalty were tuned on gameweeks 1–19 and judged on 20–38.

![Season](reports/figures/season_cumulative.png)

- **The forecast matters.** With the planner's settings held fixed, the model's forecasts beat the 5-week average at 8 of 8 settings over the season (+172 points on average) and pay fewer or equal hit points (140 vs 228 at the real −4 cost, three-week horizon).
- **The settings don't, detectably.** The ranking of settings on the first half did not carry over to the second (rank correlation −0.81), and every 19-week bootstrap interval includes zero, so no best horizon or penalty is claimed. One season is a single path; comparing settings over many restarted paths is the next step.

### Foundation-model challenger (optional)

`scripts/tabicl_challenger.py` runs TabICL v2, an open tabular foundation model, on the same walk-forward blocks (gameweeks 20–38). On a CPU it reads only the 4,000 most recent rows, so a gradient-boosting model on the same rows is the fair control.

| Forecast | Training rows | RMSE | Team points per GW |
|---|---|---|---|
| Two-stage model | ~80,000 | 1.86 | 64.4 |
| TabICL v2 | 4,000 | 1.89 | 50.9 |
| Gradient boosting, same rows | 4,000 | 1.99 | 47.5 |

TabICL is data-efficient on RMSE but its teams trail the two-stage model by 13.5 points a week (CI +3.5 to +22.9): structure and history matter more than model class here.

## Design decisions and limitations

- **Real data.** On the provided synthetic data even a perfect-foresight team beats the most expensive legal XI by only ~8%, so a 15% improvement target could not be met honestly.
- **Gradient boosting over deep learning:** tabular data, native missing values, seconds to train on a laptop.
- **One test season** (38 decisions), so intervals are wide; both are reported.
- The weekly backtest picks an XI and captain; the bench, vice-captain and substitutions appear only in the season planner.
- Ownership and transfer signals are lagged a week, because their timing is not confirmed.
- Points are predicted in total, not by component, so a scoring-rule change is learned only after it shows up in the data.
- **Next:** decision-focused training (SPO+ loss), component-level scoring, chips and sell-price rules in the season planner.

## Repository

```
main.py                 command-line entry point
src/fplopt/             data, features, model, optimise, baselines, backtest, monitor, reports,
                        explain, audit, planner, season, app (Streamlit), assistant + mcp_server (MCP tools)
tests/                  37 tests: rules, leakage, calibration, planner, Hypothesis property tests
reports/                committed results, forecasts and figures (the demo runs from these)
data/raw/               FPL Historical Dataset, 2022-23 to 2025-26 (MIT)
scripts/                data download, TabICL challenger, Hugging Face Space packaging
```

**AI-assistant interface.** `python main.py mcp` exposes `pick_team`, `explain_player`, `compare_risk` and `backtest_summary` over the Model Context Protocol (registered in `.vscode/mcp.json`). The language model only chooses the tool; every number comes from the saved forecasts and the MILP, so it cannot invent a points figure or an illegal team.

**Built for** a technical case study (fantasy sports team optimiser), extended afterwards: see [CHANGELOG.md](CHANGELOG.md). The provided starter code's baselines broke the budget and position rules, and its synthetic points were unrelated to the player data, so neither was used.

## References

- Anand, V. *FPL Historical Dataset* (MIT). https://github.com/vaastav/Fantasy-Premier-League
- Bertsimas, D. & Sim, M. (2003). Robust discrete optimization and network flows. *Mathematical Programming* 98:49–71; (2004) The price of robustness. *Operations Research* 52(1):35–53.
- Lee, T. & Kwon, C. (2014). Robust combinatorial optimization with cardinality constrained uncertainty. *4OR* 12(4):373–378.
- Romano, Y., Patterson, E. & Candès, E. (2019). Conformalized quantile regression. NeurIPS.
- Johnstone, C. & Cox, B. (2021). Conformal uncertainty sets for robust optimization. *PMLR* 152:72–90.
- Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. *JBES* 13(3):253–263; Harvey, Leybourne & Newbold (1997), *IJF* 13(2):281–291.
- Elmachtoub, A. & Grigas, P. (2022). Smart "Predict, then Optimize". *Management Science* 68(1):9–26.
- Smith, J. & Winkler, R. (2006). The optimizer's curse. *Management Science* 52(3):311–322.
- Groos, J. (2025). OpenFPL. arXiv:2508.09992. Ramezani & Dinh (2025). arXiv:2505.02170.
- Zhou & Zhu (2025). Calibrating robustness levels in predict-then-optimise. arXiv:2510.07750.
- Qu, J. et al. (2025). TabICL. ICML; TabICLv2 (2026), arXiv:2602.11139.
- NIST/SEMATECH e-Handbook: EWMA control charts.

AI use: Claude (Anthropic) assisted with research and code drafting; every result was re-run from the committed data, and the design decisions and conclusions are my own.
