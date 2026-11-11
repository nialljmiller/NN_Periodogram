# Changelog

## 1.2.0 (2026-11-10)

- **A second stage that compares the finalists.** After the network scores every
  candidate, a small ranker looks at the best ten together: each one's fold summary from
  the network, its fold FAP and periodogram powers, and how the candidates relate (2x, 1/2x,
  3x, 1/3x, 3/2, 2/3, daily and yearly aliases). It was trained on the shortlists of 84,000
  training stars (10% held out to choose when to stop) to pick the catalogue period, and to keep the first-pass choice when the
  truth is not on the shortlist. The network itself is unchanged.
- **Held-out test stars:** OGLE 93.9 -> 96.4%, Kepler 79.4 -> 82.6%, TESS 78.9 -> 80.8%,
  VVV 53.0 -> 54.6%, ASAS-SN 29.4 -> 32.1% (strictly periodic 44.6 -> 47.5%). On validation:
  +2.6 points overall (89 stars fixed, 28 broken, p = 1e-8), fewer harmonic errors on every
  survey.
- The ranker (1 MB) downloads with the model and is checked against its SHA-256. The search
  FAP is recalibrated for it.
- Investigated and not adopted (no better than 1.1 on validation): retraining without the
  VVV-derived labels, quasi-periodic training signals, a wider network, ensembles, training
  with contact-binary hard negatives, and a learned per-bin point encoder.

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
