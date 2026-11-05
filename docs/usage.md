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
Lomb-Scargle and phase-dispersion periodograms run on the full frequency grid
(cheap, on the GPU or CPU); their strongest peaks, and each one's 2x, 1/2x, 3x
and 1/3x, are passed to the network, which picks the period whose fold looks
most like a real signal. Costs about as much as running LS and PDM.

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

The network is trained to tell folds of genuinely periodic light curves at
their true period from folds of non-periodic light curves, of periodic ones at
wrong periods, and at harmonics. Its output is reported as a false-alarm
probability: near 0 means "this looks like a real period", near 1 means it
does not. It is a classifier score, not a calibrated p-value of a noise model.
