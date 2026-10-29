# Convenience targets for macOS and Linux (Windows: run the commands in README "Quick start").
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup test lint app pick season backtest explain clean

setup:            ## create .venv and install the exact tested versions
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements-lock.txt

test:             ## unit + property-based tests (~1.5 min)
	$(BIN)/python -m pytest -q

lint:
	$(BIN)/python -m pip install -q ruff && $(BIN)/ruff check .

app:              ## interactive demo, offline from reports/
	$(BIN)/python main.py app

pick:             ## print the GW 20 XI with a robustness budget of 4
	$(BIN)/python main.py pick --gw 20 --gamma 4

season:           ## full-rules season backtest (~3 min)
	$(BIN)/python main.py season

backtest:         ## rebuild every result from the raw data (~10 min)
	$(BIN)/python main.py backtest

explain:          ## decision-focused signal importance (~3 min)
	$(BIN)/python main.py explain

clean:
	rm -rf data/processed .pytest_cache .hypothesis .ruff_cache dist
