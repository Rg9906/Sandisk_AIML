"""
Loading, wafer-level splitting and the leakage assertions that keep our results honest.

Two hard constraints from CLAUDE.md are enforced here and nowhere else:
  1. Splits are BY WAFER. A wafer's dies never appear in two splits.
  2. Train and eval use ONLY dies with old_label == 0. Pre-test failures are INPUT
     (they carry the spatial signal), never targets.

Note on the provided data: `validation.parquet` is literally `test.parquet` with the
label column dropped -- the SAME wafers. It is not a third split. We therefore carve a
validation set out of TRAIN by wafer, and keep `test` as the untouched holdout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/cache")
SEED = 42


@dataclass
class Split:
    """One split's dies plus its row-aligned block memmap."""

    name: str
    dies: pd.DataFrame           # ALL dies, including old_label == 1
    blocks: np.ndarray | None    # (n_dies, k) float16 memmap, row-aligned to `dies`
    feature_cols: list[str]

    @property
    def n_wafers(self) -> int:
        return self.dies["wafer_id"].nunique()

    @property
    def eligible(self) -> np.ndarray:
        """Boolean mask of dies we are allowed to train/evaluate on."""
        return (self.dies["old_label"].to_numpy() == 0)

    @property
    def wafer_ids(self) -> set[str]:
        return set(self.dies["wafer_id"].astype(str).unique())

    def y(self) -> np.ndarray:
        assert "label" in self.dies.columns, f"split '{self.name}' has no label column"
        return self.dies["label"].to_numpy().astype(int)

    def subset(self, mask: np.ndarray, name: str | None = None) -> "Split":
        """Row-subset that keeps the dies/blocks alignment intact."""
        idx = np.flatnonzero(mask)
        return Split(
            name=name or f"{self.name}[{mask.sum()}]",
            dies=self.dies.iloc[idx].reset_index(drop=True),
            blocks=self.blocks[idx] if self.blocks is not None else None,
            feature_cols=self.feature_cols,
        )

    def describe(self) -> str:
        n = len(self.dies)
        el = self.eligible
        s = f"{self.name:12s} dies={n:>7,}  wafers={self.n_wafers:>4}  eligible={el.sum():>7,}"
        if "label" in self.dies.columns:
            y = self.y()
            s += (f"  pre-test fails={int((self.dies['old_label'] == 1).sum()):>6,}"
                  f"  NEW fails={int(y[el].sum()):>6,} ({100 * y[el].mean():.2f}% of eligible)")
        return s


def load_manifest(dataset: str = "smoke") -> dict:
    path = CACHE_DIR / dataset / "manifest.json"
    assert path.exists(), (
        f"No cache at {path}. Run:  python -m src.ingest --dataset {dataset}"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(dataset: str, split: str, *, with_blocks: bool = True) -> Split:
    """Load one split from the cache. Blocks are memory-mapped, never fully read."""
    out_dir = CACHE_DIR / dataset
    dies = pd.read_parquet(out_dir / f"{split}_dies.parquet")
    dies["wafer_id"] = dies["wafer_id"].astype(str)

    blocks = None
    bpath = out_dir / f"{split}_blocks.f16.npy"
    if with_blocks and bpath.exists():
        blocks = np.load(bpath, mmap_mode="r")
        assert len(blocks) == len(dies), (
            f"{split}: block rows {len(blocks)} != die rows {len(dies)} -- cache is corrupt"
        )

    feature_cols = [c for c in dies.columns if c.startswith("feature_")]
    return Split(name=split, dies=dies, blocks=blocks, feature_cols=feature_cols)


def wafer_split(split: Split, val_frac: float = 0.25, seed: int = SEED) -> tuple[Split, Split]:
    """
    Carve `split` into (train, val) BY WAFER.

    Wafers are stratified by their eligible-die failure rate so both sides see a mix of
    quiet and heavy wafers -- the per-wafer rate is an exponential draw, so a naive
    random split can hand validation nothing but quiet wafers.
    """
    dies = split.dies
    el = split.eligible
    per_wafer = (
        pd.DataFrame({"wafer_id": dies["wafer_id"], "el": el,
                      "y": dies["label"] if "label" in dies else 0})
        .assign(y=lambda d: d["y"] * d["el"])
        .groupby("wafer_id", observed=True)
        .agg(fails=("y", "sum"), n=("el", "sum"))
    )
    per_wafer["rate"] = per_wafer["fails"] / per_wafer["n"].clip(lower=1)

    # Interleave wafers ordered by rate, then take every 1/val_frac-th for validation.
    order = per_wafer.sort_values("rate").index.to_numpy()
    rng = np.random.default_rng(seed)
    stride = max(int(round(1 / val_frac)), 2)
    offset = int(rng.integers(stride))
    val_wafers = set(order[offset::stride])
    assert val_wafers and len(val_wafers) < len(order), "degenerate wafer split"

    is_val = dies["wafer_id"].isin(val_wafers).to_numpy()
    tr = split.subset(~is_val, name="train")
    va = split.subset(is_val, name="val")
    assert_no_wafer_leakage(tr, va)
    return tr, va


def assert_no_wafer_leakage(*splits: Split) -> None:
    """CLAUDE.md rule 1. Called after every split operation."""
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = a.wafer_ids & b.wafer_ids
            assert not overlap, (
                f"WAFER LEAKAGE: {len(overlap)} wafers in both '{a.name}' and '{b.name}': "
                f"{sorted(overlap)[:5]}"
            )


def assert_eligible_only(y: np.ndarray, old_label: np.ndarray, where: str) -> None:
    """CLAUDE.md rule 2. Called before every fit and every score."""
    assert len(y) == len(old_label), f"{where}: length mismatch"
    n_bad = int((old_label == 1).sum())
    assert n_bad == 0, (
        f"{where}: {n_bad} dies with old_label==1 reached train/eval. "
        "Pre-test failures are INPUT, never targets (CLAUDE.md rule 2)."
    )


def load_all(dataset: str = "smoke", val_frac: float = 0.25,
             seed: int = SEED) -> tuple[Split, Split, Split]:
    """The standard entry point: returns (train, val, test) with all assertions passed."""
    full_train = load_split(dataset, "train")
    test = load_split(dataset, "test")
    tr, va = wafer_split(full_train, val_frac=val_frac, seed=seed)
    assert_no_wafer_leakage(tr, va, test)
    return tr, va, test


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    tr, va, te = load_all(ds)
    print(f"dataset = {ds}\n")
    for s in (tr, va, te):
        print(" ", s.describe())
    print("\n  wafer-leakage assertions: PASSED")
    print(f"  blocks memmap: {tr.blocks.shape} {tr.blocks.dtype} (lazy)")
