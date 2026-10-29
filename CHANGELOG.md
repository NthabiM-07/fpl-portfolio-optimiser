# Changelog

## 1.1.0 (29 October 2026, after submission)

- **Full-rules season planner** (`src/fplopt/planner.py`, `src/fplopt/season.py`, `python main.py season`): a rolling-horizon MILP manages a 15-man squad through 2025-26 with free transfers (banked up to 5), -4 hits only once free transfers are used, and automatic substitutions. Settings are chosen on gameweeks 1-19 and judged on 20-38; the README reports that no setting is detectably better in one season and that the forecast is what matters (8 of 8 settings beat the 5-week mean).
- **TabICL v2 challenger** (`scripts/tabicl_challenger.py`, optional): an open tabular foundation model compared on the same walk-forward blocks, with a same-rows boosting control.
- **Repository tooling:** CI on macOS, Windows and Linux (Python 3.11-3.13) with linting; `requirements-lock.txt` with the exact tested versions; `pyproject.toml`; `Makefile`; MIT licence; consistent line endings.
- **Fixes:** UTF-8 console output for player names on any OS; the TabICL script rebuilds features on a fresh clone.

## 1.0.0 (29 October 2026, submitted)

Tag [`submission-2026-10-29`](https://github.com/NthabiM-07/fpl-portfolio-optimiser/tree/submission-2026-10-29): two-stage model with Mondrian CQR intervals, exact MILP selection with a calibrated Bertsimas-Sim robustness budget, walk-forward backtest with paired bootstrap and Diebold-Mariano tests, decision-focused signal value, xP leakage forensics, Streamlit demo, MCP tools and 31 tests.
