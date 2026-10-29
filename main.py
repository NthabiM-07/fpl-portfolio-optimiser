"""Entry point.

    python main.py backtest            # features -> walk-forward model -> MILP picks -> reports/ (~10 min)
    python main.py pick --gw 20        # best XI for a 2025-26 gameweek from the saved predictions
    python main.py app                 # interactive demo (Streamlit)
    python main.py explain             # which signal families earn points (permutation importance, ~3 min)
    python main.py season              # full FPL rules: 15-man squad, transfers and hits over the season (~3 min)
    python main.py mcp                 # serve the optimiser to an AI assistant over MCP (stdio)
    python main.py download            # refresh data/raw from the FPL Historical Dataset
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from fplopt.config import PROCESSED_DIR, REPORTS_DIR, SETTINGS  # noqa: E402

log = logging.getLogger("fplopt")


def cmd_backtest(_: argparse.Namespace) -> None:
    from fplopt.backtest import decide, walk_forward_predict
    from fplopt.data import build_tables
    from fplopt.features import build_features
    from fplopt.reports import write_all

    t0 = time.time()
    players, teams = build_tables()
    F = build_features(players, teams)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    F.to_pickle(PROCESSED_DIR / "features.pkl")
    log.info("features %s in %.0fs", F.shape, time.time() - t0)
    pred = walk_forward_predict(F, SETTINGS)
    log.info("predictions %s in %.0fs", pred.shape, time.time() - t0)
    gw, rob = decide(pred, SETTINGS)
    tabs = write_all(pred, gw, rob)
    log.info("done in %.0fs. Headline:\n%s", time.time() - t0,
             tabs["strategies_all_gws"][["strategy", "mean_pts_per_gw", "uplift_vs_last_week_%"]].round(1).to_string(index=False))


def cmd_pick(a: argparse.Namespace) -> None:
    from fplopt.optimise import realised_points, solve_team

    pred = pd.read_csv(REPORTS_DIR / "predictions_2025-26.csv.gz")
    c = pred[pred["GW"] == a.gw].reset_index(drop=True)
    if c.empty:
        raise SystemExit(f"no predictions for GW {a.gw}")
    d = (c["mu"] - c["lo"]).clip(lower=0).to_numpy()
    team = solve_team(c, c["mu"].to_numpy(), int(round(a.budget * 10)), SETTINGS.club_cap, True, d=d, gamma=a.gamma)
    xi = c.iloc[team.idx].assign(captain=lambda t: t.index == team.captain)
    xi = xi.assign(_o=xi["position"].map({"GK": 0, "DEF": 1, "MID": 2, "FWD": 3})).sort_values(["_o", "mu"], ascending=[True, False])
    cols = ["name", "position", "team", "value", "mu", "lo", "hi", "p_play60", "total_points", "captain"]
    print(xi[cols].round(2).to_string(index=False))
    expected = c["mu"].to_numpy()[team.idx].sum() + c["mu"].to_numpy()[team.captain]
    worst = f" | worst case with {a.gamma:g} shortfalls {team.objective:.1f}" if a.gamma > 0 else ""
    print(f"\nexpected {expected:.1f}{worst} | realised {realised_points(c, team):.0f} | cost GBP {xi['value'].sum() / 10:.1f}m")


def cmd_app(_: argparse.Namespace) -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "src" / "fplopt" / "app.py")], check=True)


def cmd_explain(_: argparse.Namespace) -> None:
    from fplopt.explain import signal_value
    from fplopt.reports import write_signal_value

    path = PROCESSED_DIR / "features.pkl"
    if path.exists():
        F = pd.read_pickle(path)
    else:
        from fplopt.data import build_tables
        from fplopt.features import build_features
        F = build_features(*build_tables())
    sv = signal_value(F, SETTINGS)
    write_signal_value(sv)
    print(sv.round(3).to_string(index=False))


def cmd_season(_: argparse.Namespace) -> None:
    from fplopt import season

    pred = pd.read_csv(REPORTS_DIR / "predictions_2025-26.csv.gz")
    tabs = season.run(pred)
    season.write(tabs)
    print(season.headline(tabs))


def cmd_mcp(_: argparse.Namespace) -> None:
    logging.getLogger().handlers.clear()            # stdout is the MCP channel: keep it clean
    from fplopt.mcp_server import main as serve
    serve()


def cmd_download(_: argparse.Namespace) -> None:
    from fplopt.data import download
    download()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # player names such as Ødegaard print on any console (e.g. Windows cp1252)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="FPL team selection as constrained portfolio optimisation")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backtest").set_defaults(fn=cmd_backtest)
    pk = sub.add_parser("pick")
    pk.add_argument("--gw", type=int, required=True)
    pk.add_argument("--budget", type=float, default=SETTINGS.budget_tenths / 10)
    pk.add_argument("--gamma", type=float, default=0)
    pk.set_defaults(fn=cmd_pick)
    sub.add_parser("app").set_defaults(fn=cmd_app)
    sub.add_parser("explain").set_defaults(fn=cmd_explain)
    sub.add_parser("season").set_defaults(fn=cmd_season)
    sub.add_parser("mcp").set_defaults(fn=cmd_mcp)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
