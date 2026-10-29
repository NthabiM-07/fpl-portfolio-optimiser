# Changelog

## 1.1.0 (29 October 2026)

- **Season planner** (`python main.py season`): a rolling-horizon MILP manages a 15-man squad through 2025-26 with banked free transfers, −4 hits and automatic substitutions; settings tuned on gameweeks 1–19 and judged on 20–38.
- **TabICL v2 challenger** (`scripts/tabicl_challenger.py`, optional): a tabular foundation model on the same walk-forward blocks, with a same-rows gradient-boosting control.
- **Tooling:** CI on macOS, Windows and Linux (Python 3.11–3.13) with linting; pinned versions in `requirements-lock.txt`; `pyproject.toml`; `Makefile`; MIT licence.
- **Fixes:** UTF-8-safe console output on any OS; the TabICL script rebuilds features on a fresh clone.

## 1.0.0 (29 October 2026)

Commit [`38409b3`](https://github.com/NthabiM-07/fpl-portfolio-optimiser/tree/38409b36e85980f08b8962556acd87bc5fbdbaa1), the code as submitted for the case study: two-stage model with Mondrian conformal intervals, exact MILP selection with a calibrated robustness budget, walk-forward backtest with paired bootstrap and Diebold-Mariano tests, decision-focused signal importance, xP leakage forensics, Streamlit demo, MCP tools and 31 tests.
