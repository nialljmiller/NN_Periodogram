# Training and benchmarks

Everything used to build the training set, train `periods-v4`, calibrate its search
FAP and produce the numbers in the main README. One consumer GPU (an RTX 3070 Ti) trains
the model in about 1.5 hours; the data take a few days to download and verify.

## 1. Light curves → shards

Each source writes `NAME.npz` (concatenated `time`, `mag`, `err` with `offsets`) and
`NAME_meta.json` (one record per light curve: catalogue period(s), class, survey, and a
fixed `split`: 80% train, 10% val, 10% test by hash of the star's ID).

| script | data |
|---|---|
| `build_dataset.py` | VVV light curves from PRIMVS (`$PRIMVS_DIR`): periodic candidates, ASAS-SN cross-matches, quiet (non-variable) stars |
| `sources/ogle.py` | OGLE-IV Collection of Variable Stars (RR Lyrae, Cepheids, δ Scuti, eclipsing binaries, ...) |
| `sources/asassn.py` | ASAS-SN Sky Patrol light curves of VSX periodic variables |
| `sources/kepler_tess.py` | Kepler and TESS eclipsing binary catalogues (Villanova) via lightkurve |

## 2. Verify, then label

* `verify_periods.py` (`verify_all.sh` for every shard): keeps a star only if our own
  generalised Lomb-Scargle (a top-5 peak within 1/3T of 1, 2 or ½ × the catalogue period,
  Baluev FAP < 10⁻³) or phase dispersion (better than 2000 random frequencies) recovers it.
* `relabel.py`: the label is the **catalogue** period, refined by phase dispersion within
  ±1/T of exactly 1×. (Verification must not set the label: for an eclipsing binary the
  matching Lomb-Scargle peak is at half the orbital period.) VVV candidates have no
  independent truth, so only stars verified at the candidate itself are kept.

## 3. Train

`train.py --data SHARDS --out runs/v4 --views 1 0.5 2 --workers 8`. Batches are built
in worker processes (see the main README for what they contain); the best validation
loss is kept.

## 4. Evaluate

* `benchmark_surveys.py`: per-survey accuracy (correct / harmonic / wrong) on the val or
  test split against catalogue periods, with candidate recall and paired McNemar tests;
  `--summary rows.jsonl` prints the tables. Methods: `hybrid:MODEL:WEIGHTS[:proposers]`,
  `full:MODEL:WEIGHTS`, `ls`, `pdm`, `astropy`, `nifty`.
* `export_bench_sample.py`: the benchmark's stars in one small file, for the scripts below.
* `robustness.py`: synthetic stress tests (few points, outliers, red noise, seasons,
  real cadences, pure noise).
* `calibrate_fap.py`, `validate_fap.py`: the noise calibration of the search FAP
  (written to `nn_periodogram/data/`) and its check on fresh noise and signals.
* `timing_cpu.py`: CPU time per star of every method; `proposer_study.py`: candidate
  recall of proposer settings.
* `nifty_worker.py`: runs nifty-ls in its own interpreter (it and PyTorch each ship an
  OpenMP runtime, which clash on macOS).
