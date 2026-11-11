#!/usr/bin/env python3
"""Every search mode against the standard tools, by survey and light-curve length.

Methods
  nn-hybrid-ls      periodogram(): fast LS + multi-harmonic LS propose, network + ranker pick (default)
  nn-hybrid-pdm     periodogram(proposers=("pdm",)): PDM proposes
  nn-hybrid-lspdm   periodogram(proposers=("ls", "mhls", "pdm"))
  nn-full           periodogram(method="full"): the network scores every trial frequency
  astropy-ls        astropy LombScargle(...).power(method="fast")          (CPU only)
  nifty-ls          nifty-ls: finufft on the CPU, cufinufft on the GPU
  pdm               phase dispersion: PyAstronomy on the CPU (widely used), our PyTorch PDM on
                    the GPU (and cuvarbase's PDM when it is installed)

Metrics per method x survey x length bin
  accuracy   best period within 1% of the catalogue period
  recovery   catalogue period among the method's 5 best periods
  harmonic   best period is k/n x the catalogue period (k, n <= 4)
  time       median seconds per light curve, separately on CPU and GPU

Accuracy does not depend on the hardware, so it is measured once (--pass accuracy, on the
GPU, up to --n-fast per cell, --n-slow for the slow methods); speed is measured on a small
subset on each device (--pass speed --device cpu|cuda). Every row is written as it is done,
so an interrupted run resumes where it stopped.

    python benchmark_modes.py --data SHARDS --pass accuracy --device cuda --out modes.jsonl
    python benchmark_modes.py --data SHARDS --pass speed --device cpu --out modes.jsonl
    python benchmark_modes.py --summary modes.jsonl
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_periodogram as nnp  # noqa: E402
from benchmark import verdict  # noqa: E402
from benchmark_surveys import SURVEYS, stars  # noqa: E402
from nn_periodogram.classical import pdm_theta  # noqa: E402
from nn_periodogram.periodogram import _top  # noqa: E402

FMAX, OVERSAMPLE = 10.0, 3.0
BINS = [(10, 100), (100, 1000), (1000, 10000), (10000, 10 ** 9)]
BIN_NAMES = ["10-100", "100-1k", "1k-10k", ">10k"]
METHODS = ["nn-hybrid-ls", "nn-hybrid-pdm", "nn-hybrid-lspdm", "nn-full", "astropy-ls", "nifty-ls", "pdm"]
NIFTY = None


def grid(t):
    span = np.ptp(t); df = 1 / (OVERSAMPLE * span); f0 = 2 / span
    return f0, df, int((FMAX - f0) / df)


def slow(method, n):
    """Methods whose cost grows with the full grid x points get the smaller sample."""
    return method == "nn-full" or (method in ("pdm", "nn-hybrid-pdm", "nn-hybrid-lspdm") and n > 10000)


def top_periods(power, f0, df, k=5):
    return [1 / (f0 + df * i) for i in _top(power, k)]


def run(method, t, m, e, device):
    """(best period, five best periods, seconds)."""
    dev = torch.device(device)
    s = time.perf_counter()
    if method.startswith("nn-"):
        kw = {"device": dev}
        if method == "nn-hybrid-pdm":
            kw["proposers"] = ("pdm",)
        elif method == "nn-hybrid-lspdm":
            kw["proposers"] = ("ls", "mhls", "pdm")
        elif method == "nn-full":
            kw["method"] = "full"
        r = nnp.periodogram(t, m, e, fmax=FMAX, **kw)
        return r.best_period, [p for p, _ in r.peaks[:5]], time.perf_counter() - s
    f0, df, nf = grid(t)
    if method == "astropy-ls":
        from astropy.timeseries import LombScargle
        p = LombScargle(t, m, e).power(f0 + df * np.arange(nf), method="fast")
    elif method == "nifty-ls":
        p, inside = NIFTY.power(t, m, e, f0, df, nf, "cufinufft" if dev.type == "cuda" else "finufft")
        tops = top_periods(p, f0, df)
        return tops[0], tops, inside
    elif method == "pdm":
        if dev.type == "cpu":
            p = -pyastronomy_pdm(t, m, f0 + df * np.arange(nf))
        else:
            p = -pdm_theta(t, m, f0 + df * np.arange(nf), dev)
    else:
        raise ValueError(method)
    dt = time.perf_counter() - s
    tops = top_periods(p, f0, df)
    return tops[0], tops, dt


def pyastronomy_pdm(t, m, freqs):
    """PyAstronomy's PDM (Stellingwerf 1978; 10 bins, 3 covers): theta at each frequency."""
    from PyAstronomy.pyTiming import pyPDM
    scanner = pyPDM.Scanner(minVal=float(freqs[0]), maxVal=float(freqs[-1]), dVal=float(freqs[1] - freqs[0]), mode="frequency")
    p = pyPDM.PyPDM(t, m)
    f, theta = p.pdmEquiBinCover(10, 3, scanner)
    return np.interp(freqs, f, theta)


def sample(data, n_per_cell, seed):
    """Test-split stars of every survey, grouped by length bin."""
    cells = defaultdict(list)
    for survey, prefix in SURVEYS.items():
        for name, i, lo, hi, p_true, cls, d in stars(data, prefix, 10 ** 9, seed, "test"):
            n = hi - lo
            t = np.asarray(d["time"][lo:hi], float)
            if n < 10 or p_true > np.ptp(t) / 3 or 1 / p_true > FMAX:
                continue
            b = next(k for k, (a, z) in enumerate(BINS) if a <= n < z)
            if len(cells[(survey, b)]) < n_per_cell:
                cells[(survey, b)].append((name, int(i), lo, hi, p_true, cls, d))
    return cells


def export(cells, path):
    """The sampled light curves in one file, so the benchmark can run without the shards."""
    T, M, E, off, meta = [], [], [], [0], []
    for (survey, b), rows in sorted(cells.items()):
        for name, i, lo, hi, p_true, cls, d in rows:
            T.append(np.asarray(d["time"][lo:hi], float)); M.append(np.asarray(d["mag"][lo:hi], np.float32))
            E.append(np.asarray(d["err"][lo:hi], np.float32)); off.append(off[-1] + hi - lo)
            meta.append({"survey": survey, "bin": b, "shard": name, "index": int(i), "p_true": p_true, "cls": cls})
    np.savez(path, time=np.concatenate(T), mag=np.concatenate(M), err=np.concatenate(E), offsets=np.array(off),
             meta=json.dumps(meta))
    print(f"exported {len(meta)} light curves, {off[-1]} points -> {path}")


def load_sample(path):
    z = np.load(path); meta = json.loads(str(z["meta"])); off = z["offsets"]
    d = {"time": z["time"], "mag": z["mag"], "err": z["err"]}
    cells = defaultdict(list)
    for k, r in enumerate(meta):
        cells[(r["survey"], r["bin"])].append((r["shard"], r["index"], int(off[k]), int(off[k + 1]), r["p_true"], r["cls"], d))
    return cells


def main():
    global NIFTY
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--summary", nargs="+")
    ap.add_argument("--pass", dest="mode", choices=["accuracy", "speed"], default="accuracy")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--n-fast", type=int, default=100); ap.add_argument("--n-slow", type=int, default=20)
    ap.add_argument("--n-speed", type=int, default=10); ap.add_argument("--n-speed-slow", type=int, default=3)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--export", help="write the sampled light curves to this file and stop")
    ap.add_argument("--sample", help="read light curves from an --export file instead of the shards")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = default)")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary)
    if a.export:
        return export(sample(a.data, a.n_fast, a.seed), a.export)
    if a.threads:
        torch.set_num_threads(a.threads)
    from nifty_worker import Nifty
    NIFTY = Nifty()
    cap = a.n_fast if a.mode == "accuracy" else a.n_speed
    cells = load_sample(a.sample) if a.sample else sample(a.data, cap, a.seed)
    cells = {k: v[:cap] for k, v in cells.items()}
    done = set()
    if Path(a.out).exists():
        done = {(r["pass"], r["device"], r["method"], r["shard"], r["index"]) for r in map(json.loads, open(a.out))}
    if a.device == "cuda" and "astropy-ls" in a.methods and a.mode == "speed":
        a.methods = [m for m in a.methods if m != "astropy-ls"]        # astropy has no GPU path
    with open(a.out, "a") as fout:
        for method in a.methods:
            for (survey, b), rows in sorted(cells.items()):
                n_max = (a.n_slow if a.mode == "accuracy" else a.n_speed_slow) if any(slow(method, r[3] - r[2]) for r in rows) else cap
                t_cell = []
                for name, i, lo, hi, p_true, cls, d in rows[:n_max]:
                    if (a.mode, a.device, method, name, i) in done:
                        continue
                    t, m, e = (np.asarray(d[k][lo:hi], float) for k in ("time", "mag", "err"))
                    try:
                        best, tops, dt = run(method, t, m, e, a.device)
                    except Exception as exc:                          # record and carry on
                        fout.write(json.dumps({"pass": a.mode, "device": a.device, "method": method, "survey": survey,
                                               "bin": BIN_NAMES[b], "shard": name, "index": i, "error": str(exc)[:200]}) + "\n")
                        continue
                    row = {"pass": a.mode, "device": a.device, "method": method, "survey": survey, "bin": BIN_NAMES[b],
                           "shard": name, "index": i, "cls": cls, "n": hi - lo, "p_true": p_true, "p": best,
                           "v": verdict(best, p_true), "recovered": bool(any(abs(q / p_true - 1) < 0.01 for q in tops)),
                           "seconds": dt}
                    fout.write(json.dumps(row) + "\n"); fout.flush()
                    t_cell.append(dt)
                if t_cell:
                    print(f"{a.mode:8s} {a.device:4s} {method:16s} {survey:8s} {BIN_NAMES[b]:7s} n={len(t_cell):3d} "
                          f"median {np.median(t_cell):8.3f} s", flush=True)
    NIFTY.close()


def summary(paths):
    rows = [json.loads(line) for path in paths for line in open(path) if "error" not in line]
    acc = defaultdict(list); spd = defaultdict(list)
    for r in rows:
        key = (r["method"], r["survey"], r["bin"])
        if r["pass"] == "accuracy":
            acc[key].append(r)
        spd[key + (r["device"],)].append(r["seconds"])
    print(f"{'method':16s} {'survey':8s} {'points':7s} {'n':>4s} {'acc%':>6s} {'recov%':>7s} {'harm%':>6s} {'CPU s':>8s} {'GPU s':>8s}")
    for key in sorted(acc):
        rs = acc[key]; n = len(rs)
        c = 100 * sum(r["v"] == "correct" for r in rs) / n
        rec = 100 * sum(r["recovered"] for r in rs) / n
        h = 100 * sum(r["v"] == "harmonic" for r in rs) / n
        cpu = np.median(spd.get(key + ("cpu",), [np.nan])); gpu = np.median(spd.get(key + ("cuda",), [np.nan]))
        print(f"{key[0]:16s} {key[1]:8s} {key[2]:7s} {n:4d} {c:6.1f} {rec:7.1f} {h:6.1f} {cpu:8.3f} {gpu:8.3f}")


if __name__ == "__main__":
    main()
