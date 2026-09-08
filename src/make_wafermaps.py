"""
Synthesise WM-811K-compatible wafer maps so the provided `kit/generate_data.py`
can run before/without the real LSWMD.pkl download.

Output is a pandas DataFrame pickle with the exact columns and dtypes that
`kit/generate_data.py::load_wm811k` expects:
    waferMap    : 2D np.ndarray, 0 = no die, 1 = pass, 2 = fail
    failureType : np.ndarray shaped (1, 1) holding the pattern-class string
    lotName     : str

Drop the real LSWMD.pkl at data/LSWMD.pkl and nothing downstream changes.

Usage:
    python -m src.make_wafermaps --out data/LSWMD.pkl --n 400
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# WM-811K's eight labelled pattern classes, plus "none" for clean wafers.
PATTERN_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch",
]

# Realistic WM-811K grid shapes (rows, cols). The real set is very heterogeneous;
# these span the common range so downstream code meets varied geometry.
GRID_SHAPES = [(26, 26), (25, 27), (28, 29), (30, 30), (32, 33), (34, 34), (36, 38)]


def _die_mask(rows: int, cols: int) -> np.ndarray:
    """Circular wafer footprint: True where a die physically exists."""
    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    yy, xx = np.mgrid[0:rows, 0:cols]
    r = np.sqrt(((yy - cy) / (rows / 2.0)) ** 2 + ((xx - cx) / (cols / 2.0)) ** 2)
    return r <= 1.0


def _radius(rows: int, cols: int) -> np.ndarray:
    """Normalised radial distance in [0, ~1] from the wafer centre."""
    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    yy, xx = np.mgrid[0:rows, 0:cols]
    return np.sqrt(((yy - cy) / (rows / 2.0)) ** 2 + ((xx - cx) / (cols / 2.0)) ** 2)


def _fail_probability(pattern: str, rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Per-die failure probability field for a given defect signature."""
    r = _radius(rows, cols)
    yy, xx = np.mgrid[0:rows, 0:cols]
    base = 0.01  # sprinkle of random defects on every wafer

    if pattern == "Center":
        p = 0.85 * np.exp(-(r ** 2) / (2 * 0.22 ** 2))
    elif pattern == "Donut":
        mid = rng.uniform(0.45, 0.62)
        p = 0.85 * np.exp(-((r - mid) ** 2) / (2 * 0.10 ** 2))
    elif pattern == "Edge-Ring":
        p = 0.85 * np.exp(-((r - 0.93) ** 2) / (2 * 0.07 ** 2))
    elif pattern == "Edge-Loc":
        theta0 = rng.uniform(0, 2 * np.pi)
        ang = np.arctan2(yy - (rows - 1) / 2.0, xx - (cols - 1) / 2.0)
        d_ang = np.abs(np.angle(np.exp(1j * (ang - theta0))))
        p = 0.85 * np.exp(-((r - 0.9) ** 2) / (2 * 0.10 ** 2)) * np.exp(-(d_ang ** 2) / (2 * 0.45 ** 2))
    elif pattern == "Loc":
        cy0 = rng.uniform(0.25, 0.75) * rows
        cx0 = rng.uniform(0.25, 0.75) * cols
        sigma = rng.uniform(0.08, 0.16) * max(rows, cols)
        d2 = (yy - cy0) ** 2 + (xx - cx0) ** 2
        p = 0.85 * np.exp(-d2 / (2 * sigma ** 2))
    elif pattern == "Near-full":
        p = np.full((rows, cols), 0.80)
    elif pattern == "Random":
        p = np.full((rows, cols), rng.uniform(0.10, 0.20))
    elif pattern == "Scratch":
        # A thin, slightly curved line across the wafer -- mechanical handling damage.
        theta = rng.uniform(0, np.pi)
        cy0, cx0 = rows / 2.0, cols / 2.0
        curve = rng.uniform(-0.004, 0.004)
        yn, xn = yy - cy0, xx - cx0
        perp = np.abs(np.sin(theta) * xn - np.cos(theta) * yn + curve * (xn ** 2))
        width = rng.uniform(0.7, 1.5)
        p = 0.85 * np.exp(-(perp ** 2) / (2 * width ** 2))
    else:  # "none"
        p = np.zeros((rows, cols))

    return np.clip(p + base, 0.0, 0.95)


def make_wafer(pattern: str, rng: np.random.Generator) -> np.ndarray:
    """Build one wafer map: 0 = no die, 1 = pass, 2 = fail."""
    rows, cols = GRID_SHAPES[rng.integers(len(GRID_SHAPES))]
    mask = _die_mask(rows, cols)
    p = _fail_probability(pattern, rows, cols, rng)

    wmap = np.zeros((rows, cols), dtype=np.int64)
    wmap[mask] = 1
    fail = (rng.random((rows, cols)) < p) & mask
    wmap[fail] = 2
    return wmap


def build(n_wafers: int, seed: int = 42) -> pd.DataFrame:
    """Generate an LSWMD-shaped DataFrame of `n_wafers` wafers."""
    rng = np.random.default_rng(seed)
    # ~55% patterned / 45% clean keeps the overall fail rate near the real set's.
    n_pattern = int(n_wafers * 0.55)

    records = []
    for i in range(n_wafers):
        ftype = PATTERN_CLASSES[i % len(PATTERN_CLASSES)] if i < n_pattern else "none"
        records.append({
            "waferMap": make_wafer(ftype, rng),
            "failureType": np.array([[ftype]], dtype=object),
            "lotName": f"lot{i // 25:04d}",
            "dieSize": float(np.count_nonzero(make_wafer("none", rng))),
            "trianTestLabel": np.array([["Training"]], dtype=object),
        })

    df = pd.DataFrame(records)
    # Match load_wm811k's shuffled ordering assumptions.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesise a WM-811K-compatible wafer map pickle")
    ap.add_argument("--out", default="data/LSWMD.pkl")
    ap.add_argument("--n", type=int, default=400, help="number of wafers to synthesise")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite existing {out} (is this the real LSWMD.pkl?)")
        print("Delete it first if you really want to regenerate the synthetic stand-in.")
        return

    df = build(args.n, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)

    n_pat = sum(1 for f in df["failureType"] if f[0][0] != "none")
    total_die = sum(int((w > 0).sum()) for w in df["waferMap"])
    total_fail = sum(int((w == 2).sum()) for w in df["waferMap"])
    print(f"Wrote {out}  ({len(df)} wafers, {n_pat} patterned, {len(df) - n_pat} none)")
    print(f"  dies={total_die:,}  pre-test fails={total_fail:,} ({100 * total_fail / total_die:.2f}%)")


if __name__ == "__main__":
    main()
