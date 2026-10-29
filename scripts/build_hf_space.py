"""Assemble a Hugging Face Docker Space for the demo in dist/hf_space/ (nothing is uploaded).

    python scripts/build_hf_space.py
    # then, after creating an empty Docker Space on huggingface.co (New Space -> Docker -> Blank):
    hf upload <user>/<space-name> dist/hf_space . --repo-type space

Only the app, its modules and the precomputed reports are copied, so the Space builds quickly.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "hf_space"
APP_MODULES = ["__init__.py", "app.py", "baselines.py", "config.py", "monitor.py", "optimise.py"]


def build(out: Path = OUT) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "src" / "fplopt").mkdir(parents=True)
    for f in ["Dockerfile", "README.md", "requirements.txt"]:
        shutil.copy(ROOT / "deploy" / "hf_space" / f, out / f)
    for m in APP_MODULES:
        src = ROOT / "src" / "fplopt" / m
        if src.exists():
            shutil.copy(src, out / "src" / "fplopt" / m)
    shutil.copytree(ROOT / "reports", out / "reports", ignore=shutil.ignore_patterns("*.md", "summary.json"))
    return out


if __name__ == "__main__":
    out = build()
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"built {out} ({size:.1f} MB). Upload with:\n  hf upload <user>/<space-name> {out.relative_to(ROOT)} . --repo-type space")
