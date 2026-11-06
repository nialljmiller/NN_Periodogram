#!/usr/bin/env python3
"""Wall-clock time per star of each method on the CPU only, same stars and grid
(2/T to 10/day, 3 samples per 1/T) as the benchmark.

    python timing_cpu.py --sample bench_val_sample.npz --per-survey 50 --out timing_cpu.json
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nn_periodogram as nnp  # noqa: E402
from nn_periodogram.classical import gls, pdm_theta  # noqa: E402

FMAX = 10.0


def grid(t):
    span = np.ptp(t); df = 1 / (3 * span); f0 = 2 / span
    return f0, df, int((FMAX - f0) / df)


def run(method, t, m, e):
    f0, df, nf = grid(t)
    if method == "nn_periodogram":
        return nnp.periodogram(t, m, e, fmax=FMAX, device="cpu").best_period
    if method == "nifty-ls":
        import nifty_ls
        p = nifty_ls.lombscargle(t, m, e, fmin=f0, fmax=f0 + df * (nf - 1), Nf=nf).power
    elif method == "astropy LS (fast)":
        from astropy.timeseries import LombScargle
        p = LombScargle(t, m, e).power(f0 + df * np.arange(nf), method="fast")
    elif method == "LS (exact)":
        p = gls(t, m, e, f0 + df * np.arange(nf), torch.device("cpu"))
    elif method == "PDM":
        p = -pdm_theta(t, m, f0 + df * np.arange(nf), torch.device("cpu"))
    return 1 / (f0 + df * int(np.argmax(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--per-survey", type=int, default=50); ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    d = np.load(a.sample); meta = json.loads(str(d["meta"])); off = d["offsets"]
    rng = np.random.default_rng(5)
    methods = ["nn_periodogram", "nifty-ls", "astropy LS (fast)", "LS (exact)", "PDM"]
    out = {"cpu": platform.processor() or platform.machine(), "torch_threads": torch.get_num_threads(), "rows": {}}
    t0 = d["time"][off[0]:off[1]]
    run("nn_periodogram", t0, d["mag"][off[0]:off[1]].astype(float), d["err"][off[0]:off[1]].astype(float))  # warm up
    for survey in ("OGLE", "ASAS-SN", "Kepler", "TESS", "VVV"):
        idx = rng.choice([i for i, r in enumerate(meta) if r["survey"] == survey], a.per_survey, replace=False)
        stars = [(d["time"][off[i]:off[i + 1]].astype(float), d["mag"][off[i]:off[i + 1]].astype(float),
                  d["err"][off[i]:off[i + 1]].astype(float)) for i in idx]
        npts = int(np.median([len(s[0]) for s in stars]))
        for method in methods:
            ts = []
            for t, m, e in stars:
                s = time.perf_counter(); run(method, t, m, e); ts.append(time.perf_counter() - s)
            out["rows"][f"{survey}|{method}"] = {"median_ms": 1000 * float(np.median(ts)), "mean_ms": 1000 * float(np.mean(ts)),
                                                 "median_points": npts}
            print(f"{survey:<8} (median N={npts:6d})  {method:<20} median {1000 * np.median(ts):9.1f} ms   "
                  f"mean {1000 * np.mean(ts):9.1f} ms", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
