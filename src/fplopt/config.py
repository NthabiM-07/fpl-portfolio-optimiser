"""Project-wide settings. Change values here, not inside modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
TEST_SEASON = "2025-26"
# Defensive-contribution points started in 2025-26: a regime flag lets the model adapt as that data arrives.
SCORING_ERA = {"2022-23": 0, "2023-24": 0, "2024-25": 0, "2025-26": 1}

POSITIONS = ["GK", "DEF", "MID", "FWD"]
XI_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}  # FPL starting-XI formation rules


@dataclass(frozen=True)
class Settings:
    budget_tenths: int = 835          # GBP 83.5m: the XI share of the GBP 100m squad budget (Ramezani & Dinh)
    club_cap: int = 3                 # max players from one club
    captain: bool = True              # captain scores double
    retrain_every: int = 4            # walk-forward block length in gameweeks
    calib_gws: int = 8                # rolling conformal window (gameweeks of out-of-sample scores)
    interval_alpha: float = 0.2       # 80% prediction interval
    gammas: tuple = field(default=(0, 1, 2, 3, 4, 6, 8))  # robust budgets of uncertainty to sweep
    gamma_quantile: float = 0.8       # calibrated Gamma: the 80th percentile of the uncertainty budget used in past weeks
                                      # (same 80% level as the prediction intervals, declared before looking at results)
    gamma_min_history: int = 4        # gameweeks of history before the calibrated Gamma switches on (0 before)
    random_baseline_seeds: int = 10
    seed: int = 42


SETTINGS = Settings()
