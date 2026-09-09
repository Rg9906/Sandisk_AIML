"""
Spatial feature factory -- the learned replacement for GDBN.

All wafer-grid operations live here and are the canonical implementations; `recon.py`,
`baselines.py` and `figures.py` import from this module rather than redefining them.

Every operation is a vectorised `scipy.ndimage` call over a whole wafer grid. There is
no per-die Python loop anywhere in this file (CLAUDE.md rule 5) -- we loop over wafers
(a few hundred) and never over dies (a few hundred thousand).

Feature versions:
    v1  (wave 1) -- deliberately trivial: 3x3 bad-neighbour ratio, radial distance,
                    wafer yield. Nothing else.
    v2  (wave 2) -- the real factory: multi-scale ratios, directional asymmetry, run
                    lengths, polar geometry, lot/wafer yields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, uniform_filter

# The kit generates labels with a 5x5 neighbourhood window (config: neighborhood_window).
# We match that scale exactly and bracket it with neighbours either side.
SCALES = (3, 5, 7, 9)


# --------------------------------------------------------------------------- grid ops
def wafer_grid(w: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebuild one wafer's map from its die rows.

    Returns (grid, rows, cols) where grid is 0 = no die, 1 = pass, 2 = pre-test fail,
    and rows/cols are the die coordinates in the same order as `w`.
    """
    r = w["die_row"].to_numpy()
    c = w["die_col"].to_numpy()
    grid = np.zeros((int(r.max()) + 1, int(c.max()) + 1), dtype=np.int8)
    grid[r, c] = 1
    old = w["old_label"].to_numpy() == 1
    grid[r[old], c[old]] = 2
    return grid, r, c


def bad_neighbour_ratio(grid: np.ndarray, size: int) -> np.ndarray:
    """
    Fraction of *valid* neighbours in a size x size window that are pre-test failures.

    Deliberately identical to the kit's `compute_neighborhood_fail_density`
    (fail_sum / valid_sum via uniform_filter), so our strongest spatial feature is
    measured on exactly the scale the process operates on.
    """
    fail = (grid == 2).astype(np.float32)
    valid = (grid > 0).astype(np.float32)
    fs = uniform_filter(fail, size=size, mode="constant", cval=0.0)
    vs = uniform_filter(valid, size=size, mode="constant", cval=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vs > 0, fs / vs, 0.0)


def radial_map(grid: np.ndarray) -> np.ndarray:
    """Normalised distance from grid centre -- matches the kit's `compute_radial_map`."""
    rows, cols = grid.shape
    cy, cx = rows / 2.0, cols / 2.0
    yy, xx = np.mgrid[0:rows, 0:cols]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return (r / r.max() if r.max() > 0 else r).astype(np.float32)


def polar_angle_map(grid: np.ndarray) -> np.ndarray:
    """Angle from wafer centre in [-pi, pi] -- separates directional from radial faults."""
    rows, cols = grid.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    return np.arctan2(yy - rows / 2.0, xx - cols / 2.0).astype(np.float32)


def directional_counts(grid: np.ndarray, reach: int = 3) -> dict[str, np.ndarray]:
    """
    Pre-test failures within `reach` dies in each cardinal direction.

    A scratch is directionally lopsided; an edge band is not. This is what separates
    two signatures that demand opposite engineering interventions.
    """
    fail = (grid == 2).astype(np.float32)
    out = {}
    for name, axis, sign in (("n", 0, -1), ("s", 0, 1), ("w", 1, -1), ("e", 1, 1)):
        acc = np.zeros_like(fail)
        for k in range(1, reach + 1):
            acc += np.roll(fail, sign * k, axis=axis)
        out[name] = acc
    return out


def longest_bad_run(grid: np.ndarray, reach: int = 5) -> np.ndarray:
    """
    Longest run of consecutive pre-test failures NEAR each die, over 4 orientations.

    Vectorised via cumulative-reset along each axis; a scratch scores high, scattered
    random defects do not.

    The run length is computed on failing cells, then spread with a maximum filter so a
    *passing* die inherits the longest streak in its neighbourhood. Without that spread
    the feature is identically zero on every eligible die -- which is the only population
    we ever score.
    """
    fail = (grid == 2)
    best = np.zeros(grid.shape, dtype=np.float32)

    def run_along(arr: np.ndarray) -> np.ndarray:
        # Forward run length ending at each cell.
        out = np.zeros(arr.shape, dtype=np.float32)
        acc = np.zeros(arr.shape[0], dtype=np.float32)
        for j in range(arr.shape[1]):
            acc = np.where(arr[:, j], acc + 1, 0.0)
            out[:, j] = acc
        return out

    for a in (fail, fail.T):
        f = run_along(a)
        b = run_along(a[:, ::-1])[:, ::-1]
        total = f + b - np.where(a, 1.0, 0.0)
        best = np.maximum(best, total if a is fail else total.T)

    # Both diagonals, via shear.
    for k in (1, -1):
        rows, cols = fail.shape
        sheared = np.zeros((rows, cols + rows), dtype=bool)
        for i in range(rows):
            off = i if k == 1 else rows - 1 - i
            sheared[i, off:off + cols] = fail[i]
        f = run_along(sheared)
        b = run_along(sheared[:, ::-1])[:, ::-1]
        tot = f + b - np.where(sheared, 1.0, 0.0)
        back = np.zeros(fail.shape, dtype=np.float32)
        for i in range(rows):
            off = i if k == 1 else rows - 1 - i
            back[i] = tot[i, off:off + cols]
        best = np.maximum(best, back)

    # Spread to neighbours so passing dies see the streak beside them.
    return maximum_filter(best, size=reach, mode="constant", cval=0.0)


# ------------------------------------------------------------------- feature assembly
def spatial_features(dies: pd.DataFrame, version: str = "v1") -> pd.DataFrame:
    """
    Build spatial features for every die, wafer by wafer.

    IMPORTANT: the grid is built from ALL dies including `old_label == 1`. Pre-test
    failures are INPUT here -- they are precisely the neighbourhood signal. They are
    excluded later, at train/eval time, never at feature time.
    """
    assert version in ("v1", "v2"), f"unknown spatial feature version {version!r}"
    frames = []

    for wid, w in dies.groupby("wafer_id", sort=False, observed=True):
        grid, r, c = wafer_grid(w)
        n_valid = int((grid > 0).sum())
        n_bad = int((grid == 2).sum())
        rad = radial_map(grid)

        f = {"_row_id": w.index.to_numpy()}

        if version == "v1":
            # Wave 1: exactly three features. Resist making this good.
            f["sp_bnr_3"] = bad_neighbour_ratio(grid, 3)[r, c]
            f["sp_radial"] = rad[r, c]
            f["sp_wafer_yield"] = np.full(len(w), 1.0 - n_bad / max(n_valid, 1), np.float32)
        else:
            for s in SCALES:
                f[f"sp_bnr_{s}"] = bad_neighbour_ratio(grid, s)[r, c]
            # Ratio of ratios: a tight cluster vs a broad band have different profiles.
            f["sp_bnr_ratio_3_9"] = f["sp_bnr_3"] / (f["sp_bnr_9"] + 1e-6)

            f["sp_radial"] = rad[r, c]
            f["sp_radial_sq"] = f["sp_radial"] ** 2
            ang = polar_angle_map(grid)[r, c]
            f["sp_angle_sin"] = np.sin(ang)
            f["sp_angle_cos"] = np.cos(ang)
            f["sp_edge_flag"] = (f["sp_radial"] > 0.85).astype(np.float32)

            d = directional_counts(grid)
            for k, v in d.items():
                f[f"sp_dir_{k}"] = v[r, c]
            ns = f["sp_dir_n"] - f["sp_dir_s"]
            ew = f["sp_dir_e"] - f["sp_dir_w"]
            # Asymmetry magnitude: high for scratches/edge-locs, ~0 for rings.
            f["sp_dir_asym"] = np.sqrt(ns ** 2 + ew ** 2)

            f["sp_longest_run"] = longest_bad_run(grid)[r, c]
            f["sp_wafer_yield"] = np.full(len(w), 1.0 - n_bad / max(n_valid, 1), np.float32)
            f["sp_wafer_n_dies"] = np.full(len(w), n_valid, np.float32)

            # The kit's own generative form: p_new ~ r * (1 + 3*d5 + 1.5*radial).
            # Handing the model the interaction directly beats making it rediscover it.
            f["sp_hazard"] = (1.0 + 3.0 * f["sp_bnr_5"] + 1.5 * f["sp_radial"])

            # Vectorized pattern signature classification (Idea 1)
            bnr = f["sp_bnr_5"]
            rad_val = f["sp_radial"]
            run_val = f["sp_longest_run"]
            asym_val = f["sp_dir_asym"]
            yw_val = f["sp_wafer_yield"]

            s_clean = bnr < 0.05
            s_nearfull = (~s_clean) & (yw_val < 0.45)
            s_scratch = (~s_clean) & (~s_nearfull) & (run_val >= 6) & (asym_val > 1.0)
            s_edgeloc = (~s_clean) & (~s_nearfull) & (~s_scratch) & (rad_val > 0.80) & (asym_val > 0.8)
            s_edgering = (~s_clean) & (~s_nearfull) & (~s_scratch) & (rad_val > 0.80) & (asym_val <= 0.8)
            s_center = (~s_clean) & (~s_nearfull) & (~s_scratch) & (~s_edgeloc) & (~s_edgering) & (rad_val < 0.35)
            s_donut = ((~s_clean) & (~s_nearfull) & (~s_scratch) & (~s_edgeloc) & (~s_edgering)
                       & (~s_center) & (rad_val >= 0.40) & (rad_val <= 0.75) & (bnr > 0.15))
            s_loc = ((~s_clean) & (~s_nearfull) & (~s_scratch) & (~s_edgeloc) & (~s_edgering)
                     & (~s_center) & (~s_donut) & (bnr > 0.15))
            s_random = ((~s_clean) & (~s_nearfull) & (~s_scratch) & (~s_edgeloc) & (~s_edgering)
                        & (~s_center) & (~s_donut) & (~s_loc))

            f["sp_sig_clean"] = s_clean.astype(np.float32)
            f["sp_sig_nearfull"] = s_nearfull.astype(np.float32)
            f["sp_sig_scratch"] = s_scratch.astype(np.float32)
            f["sp_sig_edgeloc"] = s_edgeloc.astype(np.float32)
            f["sp_sig_edgering"] = s_edgering.astype(np.float32)
            f["sp_sig_center"] = s_center.astype(np.float32)
            f["sp_sig_donut"] = s_donut.astype(np.float32)
            f["sp_sig_loc"] = s_loc.astype(np.float32)
            f["sp_sig_random"] = s_random.astype(np.float32)

        frames.append(pd.DataFrame(f))

    out = pd.concat(frames, ignore_index=True).sort_values("_row_id")
    out = out.drop(columns=["_row_id"]).reset_index(drop=True)
    return out.astype(np.float32)


def feature_names(version: str = "v1") -> list[str]:
    """Column names this version produces, without building anything."""
    if version == "v1":
        return ["sp_bnr_3", "sp_radial", "sp_wafer_yield"]
    names = [f"sp_bnr_{s}" for s in SCALES] + [
        "sp_bnr_ratio_3_9", "sp_radial", "sp_radial_sq", "sp_angle_sin", "sp_angle_cos",
        "sp_edge_flag", "sp_dir_n", "sp_dir_s", "sp_dir_w", "sp_dir_e", "sp_dir_asym",
        "sp_longest_run", "sp_wafer_yield", "sp_wafer_n_dies", "sp_hazard",
        "sp_sig_clean", "sp_sig_nearfull", "sp_sig_scratch", "sp_sig_edgeloc", "sp_sig_edgering",
        "sp_sig_center", "sp_sig_donut", "sp_sig_loc", "sp_sig_random",
    ]
    return names
