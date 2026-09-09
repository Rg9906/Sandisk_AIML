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
| Command Deck | `/api/summary`, `/api/wafers`, `/api/benchmarks`, `/api/provenance` |
| Live Wafer Matrix | `preds_*.pkl`, `test_dies.parquet`, block memmap |
| Die Micro-Inspector | per-die attribution, signed SHAP, conformal interval |
| Spatial Anomaly Maps | `src.explain.ROOT_CAUSE_TABLE`, per-die signatures |
| 2000-Block Telemetry | block memmap, cached MIL attention |
| AUC-PR Benchmarks | `results.csv` (baselines, ablation, imbalance ladder) |
| Robotics Interlock | derived work order, per-wafer quarantine list |
| A–Z Pipeline | `results/recon.md` |
| Risk Observatory | `/api/scores` — ROC, capture, KS and deciles are computed in the browser |
| Calibration Lab | `/api/calibration` |
| Signature Atlas | `/api/rootcauses` |
| Waveform Bank | four live dies off the active wafer; FFT and spectrogram computed in the browser |
| Neural Core | the ablation waterfall from `results.csv` |

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

## The hyperdeck layer

`hyper.css` and `hyper.js` add six decks, a boot sequence and the ambient FX on top of
`app.js` without editing it. They are additive: every selector is prefixed `hx-` or `fx-`,
and the new sections are injected into `<main>` at load, so `app.js` keeps owning the
original eleven views.

- **Command Deck** — the reactor core is a real radial bar chart: one spoke per held-out
  wafer, length proportional to that wafer's mean predicted risk. The green containment
  arc is the fraction of new failures caught.
- **Risk Observatory** — ships the 32,598 calibrated risks and truth labels once, then
  recomputes ROC, cumulative capture, the KS separation and the decile table client-side.
  Dragging the cut moves every number on the page.
- **Waveform Bank** — four dies chosen by outcome (caught failure, false alarm, escape,
  clean reference), each showing all 2,000 block readings. The FFT and the spectrogram
  are computed in the browser from those same readings.
- **Neural Core** — the ablation chain as a signal path. Edge thickness is the measured
  AUC-PR gain of the component it carries; a component worth nothing is drawn thin.
- **Calibration Lab** and **Signature Atlas** — reliability, conformal width, and the
  nine wafer-scale signatures with the physical fault each one traces to.

The look is a single dark theme defined at the end of `hyper.css`: a near-black ground,
opaque panels, and colour used only as signal. Two rules matter if you touch it. The
scanline overlay must never use `mix-blend-mode: screen` — that lifts every black on the
page to grey and is what made the first pass look washed out. And the vignette sits behind
`.shell`, so it darkens the ground rather than muting the content on top of it.

The Material Symbols font is fetched from Google. If that fetch is slow or blocked every
icon renders as its literal name, so the glyphs stay hidden until `document.fonts` confirms
the face actually loaded.

`Ctrl-K` opens the palette, number keys 1-9 jump between decks, and the `FX` button in
the top bar drops scanlines, vignette and the particle field for a low-motion display.
Rendering is capped at 30fps, panels on hidden views are skipped, and anything static is
rasterised once into an offscreen canvas rather than rebuilt per frame.
