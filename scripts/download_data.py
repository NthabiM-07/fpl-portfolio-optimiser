"""Refresh data/raw from the FPL Historical Dataset (MIT licence, Vaastav Anand)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fplopt.data import download  # noqa: E402

if __name__ == "__main__":
    download()
