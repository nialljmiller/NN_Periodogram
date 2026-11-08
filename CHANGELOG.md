# Changelog

## 1.1.0 (2026-11-07)

- **More accurate by default:** the network now scores every candidate fold four ways
  (as is, magnitudes flipped, time reversed, both) and averages them (`tta=4`). On the
  validation stars this recovers 0.7 points more periods (p = 0.011, paired test), with
  gains on every survey. All four versions come from one binning, so the extra cost is the
  network alone (about 2.5-3x on a CPU). `tta=1` gives the 1.0 behaviour.
- **Search FAP recalibrated** for the new default on 10,500 pure-noise searches.
- **Command-line tool:** `nn-periodogram star.csv`, many files at once (`--out table.csv`),
  `--check P` and `--plot fold.png`; columns are recognised by name or position, CSV, text
  and FITS are read.
- **`Periodogram.peaks` lists distinct periods** (at least 1% apart); before, the runners-up
  could be near-copies of the best period.
- **Ensembles:** comma-separated `weights=` average several networks; the model registry
  can describe ensembles and a per-model `tta`.
- Training: quasi-periodic injections and season offsets (`--quasi`), `--width`/`--blocks`;
  error analysis, proposer and FAP-validation scripts in `training/`.

## 1.0.0 (2026-11-06)

First release of the PyTorch rewrite: `periods-v4`, hybrid search, `check_period`,
calibrated search FAP, the 2022 network ported from Keras.
