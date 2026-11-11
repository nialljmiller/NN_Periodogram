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
nn-periodogram star.csv                     # or from the command line (CSV, text, FITS)
nn-periodogram lcs/*.csv --out periods.csv  # many light curves -> one table
```

## How well it works

Held-out stars from five surveys (never used for training or for choosing the model),
scored against each catalogue's own period: **correct** within 1%, otherwise **harmonic**
(k/n × the true period, k, n ≤ 4) or wrong. Every method searches the same frequency grid.

| Survey (stars) | **NN_Periodogram** | PDM | Lomb-Scargle |
|---|---|---|---|
| OGLE: RR Lyrae, Cepheids, δ Scuti, eclipsing (559) | **96.4%** | 79.1% | 66.0% |
| Kepler eclipsing binaries (281) | **82.6%** | 47.3% | 5.0% |
| TESS eclipsing binaries (459) | **80.8%** | 49.2% | 9.4% |
| VVV: near-infrared, sparse (421; ASAS-SN periods) | **54.6%** | 39.7% | 37.1% |
| ASAS-SN / VSX, strictly periodic classes (204) | **47.5%** | 36.8% | 29.4% |
| ASAS-SN / VSX, all classes (585) | **32.1%** | 23.8% | 17.1% |

Lomb-Scargle numbers are identical for astropy, nifty-ls and an exact implementation. Most
of the gain is in harmonics: Lomb-Scargle reports half the orbital period of an eclipsing
binary, and phase dispersion often a multiple; NN_Periodogram's harmonic errors are 1–16%
where theirs are 11–68%. Every difference from PDM is significant (paired McNemar test,
p < 10⁻³). ASAS-SN is low for every
method because many VSX classes (semi-regular, rotational, miscellaneous) have catalogue
periods that a fold need not confirm, and for a third of its stars no periodogram puts the
true period among its strongest peaks.

**Against the 2022 network.** On the same 100 test stars per survey, the original NN FAP
network (`legacy-2022`, exhaustive search) recovers 52% (OGLE), 22% (Kepler), 10% (TESS),
36% (VVV) and 16% (ASAS-SN) of periods, where NN_Periodogram recovers 96, 85, 82, 56 and 31%;
it also needs 6–15 s per star on a GPU, 100–200× longer.

**Robustness** (synthetic light curves with known periods; the same light curves for every
method; measured with version 1.0):

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
search. It is calibrated (for the default search) on 10,500 searches of pure noise (white, red and outlier-ridden,
on real cadences from all five surveys), because the best of hundreds of folds always looks
good by chance. On fresh noise, `best_fap < 0.01` flags 0.2% of light curves and
`best_fap < 0.1` about 5%. The network's FAP of the best fold alone is `best_fap_fold`.

**Every mode, by survey and length.** Percentage right (number of stars), in the top 5,
and harmonic; median time per light curve on 8 cores of an AMD EPYC 7763 and an RTX A6000.
The same light curves for every method in a row; bold is the best in the row (rows under 10
stars are not compared). NN times cover the whole search; classical times the periodogram.
The table scrolls sideways; the [project page](https://nialljmiller.com/projects/NN_Periodogram/)
has a pivotable version.

<!-- modes-table -->
| Survey | Points | NN hybrid LS right | NN hybrid LS top-5 | NN hybrid LS harm. | NN hybrid LS CPU | NN hybrid LS GPU | NN hybrid PDM right | NN hybrid PDM top-5 | NN hybrid PDM harm. | NN hybrid PDM CPU | NN hybrid PDM GPU | NN hybrid LS+PDM right | NN hybrid LS+PDM top-5 | NN hybrid LS+PDM harm. | NN hybrid LS+PDM CPU | NN hybrid LS+PDM GPU | NN full right | NN full top-5 | NN full harm. | NN full CPU | NN full GPU | PDM (ours) right | PDM (ours) top-5 | PDM (ours) harm. | PDM (ours) CPU | PDM (ours) GPU | PDM (PyAstronomy) CPU | astropy LS right | astropy LS top-5 | astropy LS harm. | astropy LS CPU | nifty-ls right | nifty-ls top-5 | nifty-ls harm. | nifty-ls CPU | nifty-ls GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OGLE | 10–100 | **87% (100)** | 92% | 1% | 3.1 s | 76 ms | 82% (100) | 86% | 1% | 1.5 s | 33 ms | **87% (100)** | 92% | 2% | 4.6 s | 84 ms | 83% (100) | 93% | 1% | 376 s | 3.2 s | 56% (100) | 77% | 16% | 17 ms | 1 ms | 14.0 s | 77% (100) | 92% | 3% | 869 ms | 77% (100) | 92% | 3% | 7 ms | 6 ms |
|  | 100–1,000 | **100% (100)** | 100% | 0% | 3.1 s | 79 ms | **100% (100)** | 100% | 0% | 2.1 s | 42 ms | **100% (100)** | 100% | 0% | 5.5 s | 99 ms | 95% (100) | 96% | 2% | 398 s | 4.4 s | 78% (100) | 97% | 20% | 337 ms | 9 ms | 23.1 s | 79% (100) | 80% | 19% | 1.6 s | 79% (100) | 80% | 19% | 16 ms | 7 ms |
|  | 1,000–10,000 | **98% (100)** | 99% | 1% | 3.5 s | 87 ms | **98% (100)** | 99% | 1% | 11.5 s | 219 ms | **98% (100)** | 99% | 1% | 15.3 s | 281 ms | 90% (100) | 93% | 4% | 1200 s | 10.3 s | 63% (100) | 96% | 37% | 7.4 s | 184 ms | 83.3 s | 18% (100) | 20% | 76% | 3.0 s | 18% (100) | 20% | 76% | 25 ms | 7 ms |
|  | > 10,000 | **100% (100)** | 100% | 0% | 3.2 s | 80 ms | 99% (100) | 100% | 1% | 20.2 s | 412 ms | **100% (100)** | 100% | 0% | 22.2 s | 461 ms | **100% (100)** | 100% | 0% | 609 s | 5.7 s | **100% (100)** | 100% | 0% | 12.7 s | 372 ms | 108 s | **100% (100)** | 100% | 0% | 1.7 s | **100% (100)** | 100% | 0% | 17 ms | 7 ms |
| Kepler | 1,000–10,000 | **89% (18)** | 94% | 6% | 3.0 s | 76 ms | 83% (18) | 89% | 6% | 2.3 s | 46 ms | **89% (18)** | 94% | 6% | 5.3 s | 98 ms | 72% (18) | 83% | 17% | 22.9 s | 458 ms | 28% (18) | 67% | 56% | 639 ms | 16 ms | 5.5 s | 0% (18) | 44% | 72% | 19 ms | 0% (18) | 44% | 72% | 3 ms | 6 ms |
|  | > 10,000 | 78% (100) | 85% | 6% | 3.4 s | 83 ms | **80% (100)** | 86% | 7% | 35.5 s | 1.0 s | **80% (100)** | 87% | 6% | 39.9 s | 1.1 s | 70% (100) | 76% | 9% | 334 s | 2.6 s | 33% (100) | 69% | 46% | 23.0 s | 1.0 s | 163 s | 4% (100) | 33% | 61% | 2.0 s | 4% (100) | 33% | 61% | 19 ms | 6 ms |
| TESS | > 10,000 | **82% (100)** | 91% | 12% | 3.3 s | 84 ms | **82% (100)** | 93% | 11% | 21.4 s | 726 ms | **82% (100)** | 93% | 11% | 23.1 s | 793 ms | 72% (100) | 82% | 16% | 357 s | 2.4 s | 47% (100) | 82% | 43% | 13.0 s | 700 ms | 88.9 s | 10% (100) | 21% | 71% | 1.2 s | 10% (100) | 21% | 71% | 14 ms | 6 ms |
| VVV | 10–100 | **40% (10)** | 50% | 0% | 3.0 s | 80 ms | 30% (10) | 30% | 0% | 1.6 s | 36 ms | **40% (10)** | 50% | 0% | 4.7 s | 92 ms | 30% (10) | 40% | 0% | 757 s | 6.5 s | 20% (10) | 30% | 10% | 86 ms | 3 ms | 29.4 s | 10% (10) | 10% | 30% | 4.1 s | 10% (10) | 10% | 30% | 16 ms | 7 ms |
|  | 100–1,000 | 50% (100) | 68% | 23% | 2.9 s | 81 ms | 44% (100) | 64% | 23% | 2.0 s | 40 ms | 46% (100) | 68% | 26% | 4.9 s | 97 ms | **51% (100)** | 66% | 19% | 796 s | 6.7 s | 39% (100) | 63% | 25% | 225 ms | 6 ms | 31.0 s | 40% (100) | 43% | 29% | 3.6 s | 40% (100) | 43% | 29% | 17 ms | 7 ms |
|  | 1,000–10,000 | 100% (1) | 100% | 0% | 3.2 s | 79 ms | 100% (1) | 100% | 0% | 3.5 s | 60 ms | 100% (1) | 100% | 0% | 6.6 s | 116 ms | 100% (1) | 100% | 0% | 582 s | 4.8 s | 100% (1) | 100% | 0% | 1.1 s | 27 ms | 30.5 s | 100% (1) | 100% | 0% | 1.8 s | 100% (1) | 100% | 0% | 18 ms | 6 ms |
| ASAS-SN | 10–100 | **2% (100)** | 7% | 5% | 2.8 s | 70 ms | **2% (100)** | 8% | 5% | 1.2 s | 31 ms | **2% (100)** | 8% | 5% | 4.1 s | 82 ms | 0% (100) | 5% | 11% | 203 s | 2.5 s | 1% (100) | 4% | 3% | 22 ms | 1 ms | 8.7 s | **2% (100)** | 5% | 7% | 717 ms | **2% (100)** | 5% | 7% | 10 ms | 7 ms |
|  | 100–1,000 | **35% (100)** | 44% | 11% | 2.9 s | 71 ms | 31% (100) | 40% | 9% | 1.7 s | 36 ms | 34% (100) | 43% | 11% | 4.8 s | 91 ms | 29% (100) | 34% | 6% | 682 s | 4.8 s | 27% (100) | 40% | 10% | 219 ms | 6 ms | 22.0 s | 29% (100) | 32% | 17% | 1.7 s | 29% (100) | 32% | 17% | 12 ms | 7 ms |
|  | 1,000–10,000 | **47% (100)** | 56% | 7% | 3.2 s | 80 ms | 46% (100) | 57% | 9% | 3.8 s | 78 ms | 46% (100) | 56% | 9% | 7.6 s | 133 ms | 44% (100) | 53% | 5% | 671 s | 5.2 s | 31% (100) | 52% | 19% | 2.1 s | 43 ms | 37.6 s | 21% (100) | 27% | 32% | 2.0 s | 21% (100) | 27% | 32% | 16 ms | 7 ms |
<!-- /modes-table -->

The default (NN hybrid LS) is within 2 points of the best method in every row with at least
10 stars; adding PDM proposals changes little and costs more on long light curves;
`method="full"` is no more accurate and 40–100× slower. `tta=1` is about 2× faster on a CPU for a fraction of a point.

## Ways to use it

* **`periodogram(time, mag, err)`**: standard mode. Fast Lomb-Scargle and 3-harmonic
  Lomb-Scargle periodograms (Press & Rybicki FFT method, in PyTorch) propose their 40
  strongest peaks, plus 2×, ½×, 3× and ⅓× each; the network scores them all; a small
  ranker then compares the ten best side by side (their folds, scores and how they relate:
  2×, ½×, aliases) and picks one, which is refined. `proposers=("ls", "mhls", "pdm")` adds phase-dispersion candidates (slower,
  about one point more accurate on eclipsing binaries).
* **`periodogram(..., method="full")`**: the network scores every trial frequency itself.
* **`check_period(time, mag, period, err)`**: for a period from a catalogue or another
  method. Compares P with 2P, P/2, 3P, P/3, 3P/2, 2P/3 and the daily and yearly sampling
  aliases and says which one the network prefers.
* **`fap(time, mag, period, err)`**: the network's FAP for periods chosen in advance.
* **`nn-periodogram FILES`** on the command line: the same search for one or many files
  (CSV, whitespace text or FITS; columns found by name or position), with `--check P`,
  `--out table.csv` and `--plot fold.png`.

Each candidate fold is scored four ways (as is, magnitudes flipped, time reversed, both)
and the network's outputs averaged (`tta=4`, the default since 1.1): a little more
accurate on every survey. `tta=1` is about 2× faster on a CPU.

Times are days (astropy `Time` and `Quantity` inputs are converted); magnitudes or fluxes;
errors optional (estimated from point-to-point scatter if absent). Choose the hardware with
`device="cuda"`, `"mps"` or `"cpu"`; the default is the best available.

## Models

Weights download on first use to `~/.cache/nn_periodogram` (or `$NN_PERIODOGRAM_CACHE`)
from the GitHub release, with data.nialljmiller.com as a mirror, and are checked against a
SHA-256.

| name | | size |
|---|---|---|
| `periods-v4` (default) | Circular 1-D CNN over multi-resolution binned phase profiles of the folds at P, 2P and P/2, with global light-curve features; plus the candidate ranker (since 1.2), a 2-layer transformer over the shortlist with relation-aware attention. | 3 MB + 1 MB |
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
