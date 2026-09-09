# WaferLens OS — the demo console

Mission-control front end for the die yield pipeline. Built to the
`stitch_waferlens_os_mission_control` mockups.

## Run it

```bash
python -m webapp.server                  # http://127.0.0.1:8000
python -m webapp.server --port 9000 --dataset smoke
```

Standard library only — no new dependency. Artifacts load once at start; the whole
test split stays resident and the 2,000-reading block waveforms are pulled from the
float16 memmap on demand.

It needs `results/preds_<dataset>.pkl`, `data/cache/<dataset>/` and `results/results.csv`.
If predictions are missing the server says so instead of starting empty:

```bash
python -m src.report --dataset full
```

## What is wired to what

| View | Reads |
|---|---|
| Live Wafer Matrix | `preds_*.pkl`, `test_dies.parquet`, block memmap |
| Die Micro-Inspector | per-die attribution, signed SHAP, conformal interval |
| Spatial Anomaly Maps | `src.explain.ROOT_CAUSE_TABLE`, per-die signatures |
| 2000-Block Telemetry | block memmap, cached MIL attention |
| AUC-PR Benchmarks | `results.csv` (baselines, ablation, imbalance ladder) |
| Robotics Interlock | derived work order, per-wafer quarantine list |
| A–Z Pipeline | `results/recon.md` |

Nothing on the page is a placeholder. Every figure traces to an artifact the pipeline
wrote, and any endpoint whose artifact is missing returns an error rather than a
plausible-looking number.

## API

`/api/summary` `/api/wafers` `/api/wafer/<id>` `/api/die/<wafer>/<row>/<col>`
`/api/benchmarks` `/api/imbalance` `/api/calibration` `/api/economics`
`/api/workorder/<wafer>` `/api/recon` `/api/rootcauses` `/api/focus` `/api/health`

`/api/economics` takes `cost_ratio`, `die_cost`, `wafers_per_month` and `scrap_ceiling`,
and re-sweeps the threshold on every call. That is what the war-room sliders drive.

## Two things worth knowing

**`y_pred` is the authority, not `threshold`.** The contract's `p_fail` is the
isotonic-calibrated probability, while `metrics["threshold"]` was tuned on the raw
score. Comparing one against the other under-counts flags (539 against the true 909)
and reports recall 0.368 where the pipeline reports 0.446. The server uses each die's
own `y_pred` and recovers the calibrated-scale cut from where that decision splits, so
the console's confusion matrix reproduces `results.csv` exactly.

**The wafer risk ramp is gamma-compressed for display.** Base rate is ~4% and the
distribution is heavily right-skewed, so a linear ramp renders nearly every die flat
black. The colour is compressed; every number on the panel stays linear.
