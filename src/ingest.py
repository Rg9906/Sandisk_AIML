"""
One-time materialisation of the generated parquet into a fast on-disk cache.

The kit writes `block_readings` as a space-separated string of 2000 floats. Parsing
that on every run would dominate the whole pipeline, so we do it exactly once:

    data/cache/<split>_dies.parquet   die features + labels + positions (float32)
    data/cache/<split>_blocks.f16.npy (n_dies, 2000) float16 memmap, row-aligned

Row i of the memmap is row i of the parquet. That alignment is the contract.

Usage:
    python -m src.ingest --dataset smoke
    python -m src.ingest --dataset full
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CACHE_DIR = Path("data/cache")
ID_COLS = ["wafer_id", "die_row", "die_col"]
LABEL_COLS = ["old_label", "label"]


def _parse_batch(strings: list[str], k: int) -> np.ndarray:
    """Parse a batch of space-separated float strings into an (n, k) float32 array."""
    # Joining the batch and parsing in one C-level call is ~10x faster than per-row.
    flat = np.fromstring(" ".join(strings), dtype=np.float32, sep=" ")
    if flat.size != len(strings) * k:
        # Ragged input -- fall back to the safe per-row path.
        flat = np.concatenate([np.fromstring(s, dtype=np.float32, sep=" ") for s in strings])
    return flat.reshape(len(strings), k)


def ingest_split(src: Path, split: str, out_dir: Path, batch_size: int = 4096) -> dict:
    """Convert one generated parquet into (dies.parquet, blocks.f16.npy). Returns a manifest."""
    t0 = time.time()
    pf = pq.ParquetFile(src)
    n_rows = pf.metadata.num_rows
    all_cols = [f.name for f in pf.schema_arrow]
    has_blocks = "block_readings" in all_cols
    die_cols = [c for c in all_cols if c != "block_readings"]
    feature_cols = [c for c in die_cols if c.startswith("feature_")]
    n_features = len(feature_cols)

    # ---- die table (everything but the block string) ----
    dies = pq.read_table(src, columns=die_cols).to_pandas()
    for c in feature_cols:
        dies[c] = dies[c].astype(np.float32)
    for c in [c for c in LABEL_COLS if c in dies]:
        dies[c] = dies[c].astype(np.int8)
    for c in ("die_row", "die_col"):
        dies[c] = dies[c].astype(np.int16)
    dies["wafer_id"] = dies["wafer_id"].astype("category")

    dies_path = out_dir / f"{split}_dies.parquet"
    dies.to_parquet(dies_path, index=False)
    n_wafers = int(dies["wafer_id"].nunique())

    # Release the die frame before touching blocks. The kit writes ONE row group, so the
    # block column can only be decoded whole (~14 KB of text per die); holding a 500-column
    # frame alongside it is what pushed this into swap on a 16 GB machine.
    del dies
    gc.collect()

    # ---- block memmap ----
    k = 0
    blocks_path = None
    if has_blocks:
        # STREAM the block column. `read_table` would materialise all ~2.3 GB of decoded
        # strings for the train split at once and OOM a 16 GB machine; iter_batches
        # decodes incrementally even though the kit writes a single row group.
        probe = next(pq.ParquetFile(src).iter_batches(
            batch_size=1, columns=["block_readings"]))
        k = len(probe.column(0)[0].as_py().split())
        del probe

        blocks_path = out_dir / f"{split}_blocks.f16.npy"
        mm = np.lib.format.open_memmap(
            blocks_path, mode="w+", dtype=np.float16, shape=(n_rows, k)
        )
        written = 0
        for batch in pq.ParquetFile(src).iter_batches(
            batch_size=batch_size, columns=["block_readings"]
        ):
            strings = batch.column(0).to_pylist()
            mm[written:written + len(strings)] = _parse_batch(strings, k).astype(np.float16)
            written += len(strings)
            del strings, batch
            if written % (batch_size * 8) == 0:
                gc.collect()
            print(f"    {split}: blocks {written:,}/{n_rows:,}", end="\r", flush=True)
        assert written == n_rows, f"wrote {written} block rows, expected {n_rows}"
        mm.flush()
        del mm
        gc.collect()
        print()

    manifest = {
        "split": split,
        "source": str(src),
        "n_rows": int(n_rows),
        "n_wafers": n_wafers,
        "n_features": n_features,
        "n_blocks": int(k),
        "has_label": "label" in die_cols,
        "dies_path": str(dies_path),
        "blocks_path": str(blocks_path) if blocks_path else None,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  {split}: {n_rows:,} dies x {len(feature_cols)} features x {k} blocks "
          f"({manifest['n_wafers']} wafers, {manifest['seconds']}s)")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialise generated parquet into a fast cache")
    ap.add_argument("--dataset", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--src-dir", default=None)
    ap.add_argument("--split", default=None,
                    help="ingest only this split (runs each in its own process)")
    args = ap.parse_args()

    src_dir = Path(args.src_dir or f"data/generated/{args.dataset}")
    out_dir = CACHE_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    wanted = [args.split] if args.split else ["train", "test", "validation"]
    if args.split and (out_dir / "manifest.json").exists():
        manifests = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    for split in wanted:
        src = src_dir / f"{split}.parquet"
        if not src.exists():
            print(f"  {split}: MISSING at {src} -- skipped")
            continue
        manifests[split] = ingest_split(src, split, out_dir)

    (out_dir / "manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"\nCache ready at {out_dir}")


if __name__ == "__main__":
    main()
