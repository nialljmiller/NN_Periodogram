#!/usr/bin/env python3
"""Robustness tests on synthetic light curves with known periods.

Each case draws signals (sine, RR Lyrae-like, eclipsing binary) with random
periods, then stresses one thing at a time, and every method sees the same
light curves:

  points    N = 10 ... 1000 points over 1000 d
  outliers  5% of points thrown 5-15 sigma
  red       red noise (AR(1), 1-d correlation time) plus a linear trend
  seasons   ground-based: nightly sampling (1-d alias) in 4-month seasons
  cadence   real cadences and errors from each survey's benchmark stars
  noise     no signal at all (white, red, trend): how often is a period claimed?

Methods: nn_periodogram (default model), Lomb-Scargle (nifty-ls) and PDM.
Verdicts: correct (1%), harmonic, wrong. For pure noise: the distribution of
the reported FAP (NN) and of the Baluev FAP (LS).

    python robustness.py --sample bench_val_sample.npz --out robustness.json
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
from nn_periodogram.classical import pdm_theta  # noqa: E402
from train import signal  # noqa: E402

FMAX = 10.0
NIFTY = None   # nifty-ls worker process (see nifty_worker.py)


def make_signal(t, P, shape, rng):
    phase = np.mod(t / P + rng.uniform(), 1.0)
    return signal(shape, phase, rng)


def classical(t, m, e, method):
    span = np.ptp(t); df = 1 / (3 * span); f0 = 2 / span; nf = int((FMAX - f0) / df)
    if method == "ls":
        p, _ = NIFTY.power(t, m, e, f0, df, nf)
        return 1 / (f0 + df * int(np.argmax(p))), float(np.max(p))
    th = pdm_theta(t, m, f0 + df * np.arange(nf), torch.device("cpu"))
    return 1 / (f0 + df * int(np.argmin(th))), float(np.min(th))


def run_case(name, draws, methods, device):
    rows = []
    for t, m, e, P, shape in draws:
        row = {"case": name, "n": len(t), "P": P, "shape": shape}
        for meth in methods:
            t0 = time.time()
            if meth == "nn":
                res = nnp.periodogram(t, m, e, fmax=FMAX, device=device)
                p, score = res.best_period, res.best_fap
            else:
                p, score = classical(t, m, e, meth)
            row[meth] = {"p": p, "score": score, "v": verdict(p, P) if P else None, "s": time.time() - t0}
            if meth == "ls" and P is None:     # Baluev FAP of the highest LS peak, for the noise case
                from astropy.timeseries import LombScargle
                row[meth]["fap"] = float(LombScargle(t, m, e).false_alarm_probability(score, maximum_frequency=FMAX))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--device", default="cpu")
    ap.add_argument("--snr", type=float, default=1.0, help="signal rms / noise rms")
    a = ap.parse_args()
    global NIFTY
    from nifty_worker import Nifty
    NIFTY = Nifty()
    rng = np.random.default_rng(2026)
    methods = ["nn", "ls", "pdm"]
    cases = {}

    def white(n, span=1000.0):
        return np.sort(rng.uniform(0, span, n))

    def draw(t, err=None, extra=None, P=None, shape=None):
        err = np.ones(len(t)) if err is None else err
        shape = shape or rng.choice(["sine", "rr", "eb"])
        span = np.ptp(t)
        P = P or float(np.exp(rng.uniform(np.log(1.5 / FMAX), np.log(span / 4))))
        sig = make_signal(t, P, shape, rng); sig = (sig - sig.mean()) / (sig.std() + 1e-12)
        m = 15 + a.snr * np.median(err) * sig + rng.normal(0, 1, len(t)) * err
        if extra:
            m = extra(t, m, err)
        return t, m, err, P, shape

    for n in (10, 20, 30, 50, 100, 300, 1000):
        cases[f"points N={n}"] = [draw(white(n)) for _ in range(a.n)]

    def outliers(t, m, e):
        k = rng.random(len(t)) < 0.05
        m = m.copy(); m[k] += rng.choice([-1, 1], k.sum()) * rng.uniform(5, 15, k.sum()) * e[k]
        return m
    cases["outliers 5%"] = [draw(white(200), extra=outliers) for _ in range(a.n)]

    def red(t, m, e):
        r = np.zeros(len(t)); tau = 1.0
        for i in range(1, len(t)):
            c = np.exp(-(t[i] - t[i - 1]) / tau)
            r[i] = c * r[i - 1] + np.sqrt(1 - c * c) * rng.normal()
        return m + 0.7 * np.median(e) * r + rng.normal(0, 0.5) * np.median(e) * (t - t.mean()) / np.ptp(t)
    cases["red noise + trend"] = [draw(white(200), extra=red) for _ in range(a.n)]

    def seasonal(n):
        nights = np.arange(0, 1500)
        nights = nights[(nights % 365) < 120]                                 # 4-month seasons
        t = rng.choice(nights, n) + rng.normal(0.0, 0.08, n)                 # observed near midnight
        return np.sort(t)
    cases["seasons + nightly"] = [draw(seasonal(150)) for _ in range(a.n)]
    cases["seasons + nightly, P 0.3-3 d"] = [draw(seasonal(150), P=float(np.exp(rng.uniform(np.log(0.3), np.log(3)))))
                                            for _ in range(a.n)]

    d = np.load(a.sample); meta = json.loads(str(d["meta"])); off = d["offsets"]
    for survey in ("OGLE", "ASAS-SN", "Kepler", "TESS", "VVV"):
        idx = [i for i, r in enumerate(meta) if r["survey"] == survey]
        draws = []
        for i in rng.choice(idx, min(a.n, len(idx)), replace=False):
            t = np.asarray(d["time"][off[i]:off[i + 1]], float); e = np.asarray(d["err"][off[i]:off[i + 1]], float)
            if len(t) > 20000:
                k = np.sort(rng.choice(len(t), 20000, replace=False)); t, e = t[k], e[k]
            draws.append(draw(t, err=np.clip(e, 1e-4, None)))
        cases[f"cadence {survey}"] = draws

    def noise_only(kind):
        t = white(200); e = np.ones(len(t)); m = 15 + rng.normal(0, 1, len(t))
        if kind == "red":
            m = red(t, m, e)
        return t, m, e, None, kind
    cases["noise white"] = [noise_only("white") for _ in range(a.n)]
    cases["noise red + trend"] = [noise_only("red") for _ in range(a.n)]

    all_rows = []
    for name, draws in cases.items():
        rows = run_case(name, draws, methods, a.device)
        all_rows += rows
        if draws[0][3] is None:
            nn_f = np.array([r["nn"]["score"] for r in rows]); ls_f = np.array([r["ls"]["fap"] for r in rows])
            print(f"{name:<30} claimed periodic at FAP<0.01: NN {100 * np.mean(nn_f < 0.01):5.1f}%  LS(Baluev) "
                  f"{100 * np.mean(ls_f < 0.01):5.1f}%   at FAP<0.1: NN {100 * np.mean(nn_f < 0.1):5.1f}%  "
                  f"LS {100 * np.mean(ls_f < 0.1):5.1f}%", flush=True)
        else:
            out = [f"{name:<30}"]
            for meth in methods:
                v = [r[meth]["v"] for r in rows]
                out.append(f"{meth} {100 * v.count('correct') / len(v):5.1f}% ({100 * v.count('harmonic') / len(v):4.1f}% harm)")
            print("  ".join(out), flush=True)
    json.dump(all_rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
