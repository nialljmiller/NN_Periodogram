# Usage

## Install

```bash
pip install nn-periodogram            # numpy + PyTorch
pip install "nn-periodogram[astropy]" # if you pass astropy Time / Quantity
```

PyTorch picks your hardware; for a CPU-only install use
`pip install torch --index-url https://download.pytorch.org/whl/cpu` first.
Weights (a few MB) download on first use to `~/.cache/nn_periodogram`
(set `NN_PERIODOGRAM_CACHE` to move it) and are checked against a SHA-256.

## Inputs

Times, magnitudes (or fluxes) and optional errors, as arrays of any length from
about ten points to hundreds of thousands. Nothing is resampled or truncated.
Times are days; astropy `Time` objects and `Quantity` arrays are converted for
you. Without errors, they are estimated from point-to-point scatter.

## Three ways to use it

**`periodogram(time, mag, err)`**, standard mode (`method="hybrid"`).
Fast Lomb-Scargle and 3-harmonic Lomb-Scargle periodograms run on the full
frequency grid; their 40 strongest peaks, and each one's 2x, 1/2x, 3x and 1/3x,
are passed to the network, which sees every candidate folded at P, 2P and P/2
side by side and picks the period whose fold looks most like a real signal.
`proposers=("ls", "mhls", "pdm")` adds phase-dispersion candidates.

**`periodogram(..., method="full")`**: the network scores every trial frequency
itself, then refines the best peaks. Independent of the classical methods, and
slower.

**`check_period(time, mag, period, err)`**: for a period you already have, from
a catalogue or another method. Compares P with 2P, P/2, 3P, P/3, 3P/2, 2P/3 and
the daily and yearly sampling aliases, and says which one the network prefers:

```text
>>> nnp.check_period(t, m, 0.2538, e)
PeriodCheck(0.253800 d -> 2P = 0.507600 d)
  2P               0.507600 d   FAP 0.0021
  P                0.253800 d   FAP 0.31
  ...
```

**`fap(time, mag, period, err)`**: the false-alarm probability of one or many
given periods.

## What the FAP means

`periodogram()` returns two numbers for the best period:

* `best_fap`: the false-alarm probability **of the search**. Out of light curves with no
  periodic signal and the same number of points, the fraction whose best fold would score
  this well somewhere among the hundreds of candidates tried. It is calibrated on 10,500
  searches of pure noise (white, red and outlier-ridden, on real cadences from five
  surveys). Use this to decide whether a star is periodic.
* `best_fap_fold`: the network's score of the best fold alone. Too optimistic for
  detection after a search (the best of many tries always looks good), but the right
  quantity when the period was fixed in advance, which is what `fap()` returns.

Both are near 0 for "this looks like a real period" and near 1 for "it does not".
