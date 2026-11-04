# NN_Periodogram

Neural-network false-alarm probabilities and periodograms for astronomical time
series: one package, one API, GPU-ready. It brings together
[NN_FAP](https://github.com/nialljmiller/NN_FAP) (Miller et al. 2022) and the
earlier NN_Periodogram code, which is kept at the `legacy-2025` tag.

```python
import nn_periodogram

p = nn_periodogram.periodogram(time, mag)       # search all trial periods
p.best_period, p.best_fap, p.peaks, p.note
p.plot()

nn_periodogram.fap(time, mag, 0.50775)          # score one period (or an array of them)
```

A low FAP means the light curve folded at that period looks like a real signal.
Folding at twice the true period also gives a coherent curve and scores about
the same; `periodogram()` reports the shorter period in that case and says so
in `p.note`.

## Install

```bash
pip install git+https://github.com/nialljmiller/NN_Periodogram
```

Model weights download on first use to `~/.cache/nn_periodogram` (override with
`NN_PERIODOGRAM_CACHE`) from the GitHub release, falling back to
data.nialljmiller.com, and are checked against a SHA-256 before use. They are
also archived on Zenodo. Runs on CUDA, Apple MPS or CPU.

## Models

| name | what | size |
|---|---|---|
| `legacy-2022` | The original NN_FAP network (14 × GRU-1024), ported exactly from Keras. Same outputs as the 2022 TensorFlow model to 1e-5. | 343 MB |
| `periods-v1` | Circular 1-D CNN over a 64-bin phase profile, trained on PRIMVS (VVV) light curves and injected signals. | 3 MB |

Choose with `model=`. Benchmark: see `training/` and the results table below.

## How the models are trained and tested

Labels never come from an NN FAP (PRIMVS's `best_fap` is the 2022 network's
output). Real positives are PRIMVS stars whose LS, PDM and CE periods agree;
injections put sines, RR Lyrae-like, eclipsing and random Fourier signals into
real VVV light curves of quiet stars; negatives are quiet stars, wrong periods,
nearby incoherent periods and daily aliases. The benchmark is a full
periodogram search on PRIMVS stars matched to ASAS-SN, scored against the
ASAS-SN period, none of which are used in training. `training/manifest.json`
records every selection rule, seed and split.

## Citation

Miller et al. — NN_FAP (2022) and NN_Periodogram (2026). See `CITATION.cff`.
