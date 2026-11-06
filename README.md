# NN_Periodogram

[![PyPI](https://img.shields.io/pypi/v/nn-periodogram)](https://pypi.org/project/nn-periodogram/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24022627.svg)](https://doi.org/10.5281/zenodo.24022627)
[![tests](https://github.com/nialljmiller/NN_Periodogram/actions/workflows/tests.yml/badge.svg)](https://github.com/nialljmiller/NN_Periodogram/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-online-blue)](https://nialljmiller.github.io/NN_Periodogram/)

**Find the periods of variable stars in any survey's light curves, and tell real periods
from harmonics and aliases.** A fast periodogram proposes candidate periods; a neural
network, which sees each light curve folded at P, 2P and P/2 side by side, decides which
one is real. Runs on a GPU or a CPU, on light curves from ten to hundreds of thousands of
points.

```python
import nn_periodogram as nnp

p = nnp.periodogram(time, mag, err)        # find the period
p.best_period, p.best_fap                  # period and false-alarm probability of the search
p.peaks                                    # runners-up: [(period, fold FAP), ...]
p.plot()

nnp.check_period(time, mag, 0.3127, err)   # is a known period right, or really 2P, P/2, an alias?
nnp.fap(time, mag, [0.6254, 0.3127], err)  # false-alarm probability of given periods
```

```bash
pip install nn-periodogram
```

## How well it works

Held-out stars from five surveys (never used for training or for choosing the model),
scored against each catalogue's own period: **correct** within 1%, otherwise **harmonic**
(k/n × the true period, k, n ≤ 4) or wrong. Every method searches the same frequency grid.

| Survey (stars) | **NN_Periodogram** | PDM | Lomb-Scargle |
|---|---|---|---|
| OGLE: RR Lyrae, Cepheids, δ Scuti, eclipsing (559) | **92.8%** | 79.1% | 66.0% |
| Kepler eclipsing binaries (281) | **79.7%** | 47.3% | 5.0% |
| TESS eclipsing binaries (459) | **78.6%** | 49.2% | 9.4% |
| VVV: near-infrared, sparse (421; ASAS-SN periods) | **52.5%** | 39.7% | 37.1% |
| ASAS-SN / VSX, strictly periodic classes (204) | **44.1%** | 36.8% | 29.4% |
| ASAS-SN / VSX, all classes (585) | **28.9%** | 23.8% | 17.1% |

Lomb-Scargle numbers are identical for astropy, nifty-ls and an exact implementation. Most
of the gain is in harmonics: Lomb-Scargle reports half the orbital period of an eclipsing
binary, and phase dispersion often a multiple; NN_Periodogram's harmonic errors are 2–16%
where theirs are 11–68%. Every difference from PDM is significant (paired McNemar test,
p < 10⁻³; p = 0.03 for the 204 strictly periodic ASAS-SN stars). ASAS-SN is low for every
method because many VSX classes (semi-regular, rotational, miscellaneous) have catalogue
periods that a fold need not confirm, and for a third of its stars no periodogram puts the
true period among its strongest peaks.

**Against the 2022 network.** On the same 100 test stars per survey, the original NN FAP
network (`legacy-2022`, exhaustive search) recovers 52% (OGLE), 22% (Kepler), 10% (TESS),
36% (VVV) and 16% (ASAS-SN) of periods, where NN_Periodogram recovers 91, 82, 81, 50 and 28%;
it also needs 6–14 s per star on a GPU, about 200× longer.

**Robustness** (synthetic light curves with known periods; the same light curves for every
method):

| Case | NN_Periodogram | Lomb-Scargle | PDM |
|---|---|---|---|
| 100 / 300 / 1000 points | **82 / 93 / 99%** | 69 / 71 / 63% | 49 / 67 / 75% |
| 5% outliers (5–15σ) | **71%** | 43% | 1% |
| red noise and a trend | **89%** | 65% | 50% |
| ground-based seasons, nightly sampling | **81%** | 67% | 53% |
| real OGLE / ASAS-SN / Kepler / TESS / VVV cadences | **85 / 81 / 99 / 77 / 67%** | 65 / 65 / 67 / 53 / 58% | 67 / 57 / 87 / 59 / 31% |

Below about 50 points at signal = noise, no method finds periods reliably.

**False-alarm probability.** `best_fap` is the probability that a light curve with *no*
periodic signal, of the same size, would produce as good a best fold somewhere in the
search. It is calibrated on 10,500 searches of pure noise (white, red and outlier-ridden,
on real cadences from all five surveys), because the best of hundreds of folds always looks
good by chance. On fresh noise, `best_fap < 0.01` flags 0.2% of light curves and
`best_fap < 0.1` about 6%. The network's FAP of the best fold alone is `best_fap_fold`.

**Speed** (median per star; same stars and grid; Apple M-series CPU and an RTX 3070 Ti):

| Light curves | NN_Periodogram CPU | NN_Periodogram GPU | PDM (CPU) | exact LS (CPU) | astropy LS | nifty-ls |
|---|---|---|---|---|---|---|
| ~400 points (OGLE, ASAS-SN, VVV) | 0.24–0.29 s | ~0.05 s | 0.09–0.11 s | 0.04–0.08 s | 0.05–0.09 s | 0.004 s |
| ~65,000 points (Kepler, TESS) | 0.36 s | ~0.07 s | 9.9–17.7 s | 7.2–8.5 s | 0.08–0.12 s | 0.007 s |

A Lomb-Scargle periodogram is the first step of NN_Periodogram's search, so it cannot be
faster than the fastest LS; it costs a fraction of a second per star and does not grow
with the number of points.

## Ways to use it

* **`periodogram(time, mag, err)`**: standard mode. Fast Lomb-Scargle and 3-harmonic
  Lomb-Scargle periodograms (Press & Rybicki FFT method, in PyTorch) propose their 40
  strongest peaks, plus 2×, ½×, 3× and ⅓× each; the network scores them all and the best
  is refined. `proposers=("ls", "mhls", "pdm")` adds phase-dispersion candidates (slower,
  about one point more accurate on eclipsing binaries).
* **`periodogram(..., method="full")`**: the network scores every trial frequency itself.
* **`check_period(time, mag, period, err)`**: for a period from a catalogue or another
  method. Compares P with 2P, P/2, 3P, P/3, 3P/2, 2P/3 and the daily and yearly sampling
  aliases and says which one the network prefers.
* **`fap(time, mag, period, err)`**: the network's FAP for periods chosen in advance.

Times are days (astropy `Time` and `Quantity` inputs are converted); magnitudes or fluxes;
errors optional (estimated from point-to-point scatter if absent). Choose the hardware with
`device="cuda"`, `"mps"` or `"cpu"`; the default is the best available.

## Models

Weights download on first use to `~/.cache/nn_periodogram` (or `$NN_PERIODOGRAM_CACHE`)
from the GitHub release, with data.nialljmiller.com as a mirror, and are checked against a
SHA-256.

| name | | size |
|---|---|---|
| `periods-v4` (default) | Circular 1-D CNN over multi-resolution binned phase profiles of the folds at P, 2P and P/2, with global light-curve features. | 3 MB |
| `legacy-2022` | The 2022 NN_FAP network (14 × GRU-1024; Miller et al. 2024), ported exactly from Keras. Exhaustive search only. | 343 MB |

## Training

About 380,000 light curves of known periodic variables, each labelled with the catalogue's
own period refined by phase dispersion, and each kept only if our own Lomb-Scargle or
phase-dispersion analysis independently recovers that period (or its double or half):
OGLE-IV (179,000 RR Lyrae, Cepheids, δ Scuti, eclipsing binaries), ASAS-SN/VSX (96,000),
VVV/PRIMVS (97,000), Kepler (2,500) and TESS (4,300) eclipsing binaries; 10% of the stars
of every survey are held out for testing and 10% for validation. Batches mix real stars at
their period (positives) or at wrong, nearby, harmonic and alias periods (negatives);
signals injected into quiet VVV stars and into synthetic red, trended and outlier-ridden
noise on every survey's cadences; magnitude-shuffled light curves; and pure noise. Each
example is randomly thinned (10 points to all), flipped, time-reversed, cut by a season
and given rescaled errors. Everything is in `training/`.

## Citing

Please cite the software, [doi:10.5281/zenodo.24022627](https://doi.org/10.5281/zenodo.24022627) (all versions; see `CITATION.cff`), and

Miller N., Lucas P. W., Sun Y., Guo Z., Cooper W. J., Morris C., 2024, *The verification
of periodicity with the use of recurrent neural networks*, RAS Techniques and Instruments,
3, 224. [doi:10.1093/rasti/rzae015](https://doi.org/10.1093/rasti/rzae015)

The pre-2026 code is at the `legacy-2025` tag.
