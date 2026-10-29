---
title: FPL Portfolio Optimiser
emoji: ⚽
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
short_description: Two-stage forecasts + exact MILP team selection, backtested
---

# FPL team selection as constrained portfolio optimisation

Interactive demo of a walk-forward backtest on every 2025-26 Premier League gameweek: a two-stage
gradient-boosting model (will he play? how much if he does?) with calibrated intervals feeds an
exact integer programme that picks the best legal XI and captain. Every baseline goes through the
same optimiser and rules.

The app runs from precomputed results; the full code, tests and write-up are in the GitHub
repository. Data: FPL Historical Dataset by Vaastav Anand (MIT licence).
