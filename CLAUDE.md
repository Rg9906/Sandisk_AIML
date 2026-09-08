# CLAUDE.md — WaferLens

Read `PROJECT.md` before doing anything. This file is the short list of things that must never break.

## Project in one line
Predict per-die failure probability on semiconductor wafers at two resolutions (Model A = die + spatial, Model B = + 2000 block readings), and explain every prediction by resolution. 24-hour hackathon, 4 people.

## Working style — READ THIS FIRST
**We build in waves, not in phases.** Every wave must end with the FULL pipeline running end to end
(data → features → model → metrics → explanation → dashboard) and a tagged commit.

- Never build one component to high quality while another doesn't exist yet.
- If asked to improve something, first confirm `python run_pipeline.py --wave N` still completes.
- Prefer a working ugly version now over a clean version in two hours.
- When blocked for more than ~15 minutes, implement the dumb fallback and leave a `# TODO(wave3):` marker.

## Hard constraints — violating any of these invalidates our results

1. **Split train/val/test BY WAFER, never by die.** Dies from one wafer must never appear in two splits. Assert this in code.
2. **Train and evaluate ONLY on dies where `old_label == 0`.** Pre-test failures are INPUT (spatial signal), not targets. The task is predicting which green dies turn orange.
3. **Never select models on accuracy.** Use AUC-PR (primary) and F1 on the failure class. Accuracy is reported only for the deliverable table.
4. **Never leave the decision threshold at 0.5.** Tune it on the validation PR curve, store it, apply it at test time.
5. **No per-die Python loops over the wafer.** Use `scipy.ndimage` / vectorised numpy. A nested loop over dies will not finish in time.
6. **Every model emits the output contract** in `PROJECT.md` §4. No exceptions — the dashboard and the explainer depend on it.
7. **Never fit anything (scalers, thresholds, imputers, PCA, class weights) on validation or test data.** Fit on train, apply elsewhere.
8. **Log every run to `results/results.csv`.** An experiment that wasn't logged didn't happen.
9. **Set seeds everywhere** (`numpy`, `random`, `torch`, model `random_state`). Judges may ask us to rerun.
10. **Do not delete or rewrite `generate_data.py` or anything shipped in the hackathon zip.** Read it, wrap it, never modify it.

## Style
- Python 3.10+, type hints on public functions, docstrings short.
- One responsibility per module. Keep `src/` files under ~300 lines.
- No new dependency without adding it to `requirements-ours.txt`.
- Print metrics as a markdown table so they can be pasted straight into the deck.

## What NOT to spend time on
- Wafer-map CNN classifiers from scratch (saturated; use as a feature only, from an existing repo)
- Hyperparameter tuning before wave 3
- Refactors, abstractions, or config frameworks
- Anything not traceable to a deliverable in `PROJECT.md` §2
