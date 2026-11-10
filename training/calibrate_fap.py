#!/usr/bin/env python3
"""Calibrate periodogram()'s best FAP for the search (look-elsewhere effect).

The network's FAP is per fold. periodogram() reports the best of several hundred
candidates, and the best of many tries on pure noise is always low. Here the
default search runs on pure-noise light curves (real cadences and errors from every
survey; white noise, red noise with trends, outliers; 10 to 20000 points) and the
lowest per-fold FAP of each is recorded. The calibrated FAP of a real search is the
fraction of noise searches, in the same light-curve-size bin, whose best fold scored
at least as well, with an exponential tail fit beyond the simulated range.

Writes nn_periodogram/data/calibration_<model>.json.

    python calibrate_fap.py --sample bench_val_sample.npz --per-bin 1500 --procs 8
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EDGES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]     # log10(number of points) bins
CADENCES = None


OPTS = {}


def _init(sample, opts):
    global CADENCES
    import torch
    torch.set_num_threads(1)
    OPTS.update(opts)
    d = dict(np.load(sample)); off = d["offsets"]
    CADENCES = [(d["time"][off[i]:off[i + 1]], d["err"][off[i]:off[i + 1]]) for i in range(len(off) - 1)]


def null_search(args):
    import importlib
    import nn_periodogram as nnp
    seed, lo, hi, model = args
    if OPTS.get("ranker"):
        P = importlib.import_module("nn_periodogram.periodogram")
        import torch as _t
        net = P._model(model, _t.device(OPTS.get("device", "cpu")), OPTS.get("weights"))
        if getattr(net, "ranker", None) is None:
            net.ranker = nnp.hub.load_ranker(OPTS["ranker"]).to(OPTS.get("device", "cpu"))
    rng = np.random.default_rng(seed)
    n = int(round(10 ** rng.uniform(lo, hi)))
    pool = [c for c in CADENCES if len(c[0]) >= n] or CADENCES
    t, e = pool[rng.integers(len(pool))]
    k = np.sort(rng.choice(len(t), min(n, len(t)), replace=False))
    t, e = t[k].astype(float), np.clip(e[k].astype(float), 1e-4, None)
    m = rng.normal(0, 1, len(t)) * e
    kind = rng.choice(["white", "red", "outliers"], p=[0.5, 0.25, 0.25])
    if kind == "red":
        r = np.zeros(len(t)); tau = float(np.exp(rng.uniform(np.log(0.3), np.log(30))))
        for i in range(1, len(t)):
            c = np.exp(-(t[i] - t[i - 1]) / tau)
            r[i] = c * r[i - 1] + np.sqrt(1 - c * c) * rng.normal()
        m = m + rng.uniform(0.3, 1.0) * np.median(e) * r + rng.normal(0, 0.5) * np.median(e) * (t - t.mean()) / np.ptp(t)
    elif kind == "outliers":
        o = rng.random(len(t)) < 0.03
        m[o] += rng.choice([-1, 1], o.sum()) * rng.uniform(5, 15, o.sum()) * e[o]
    res = nnp.periodogram(t, 15 + m, e, model=model, calibrated=False, device=OPTS.get("device", "cpu"),
                          weights=OPTS.get("weights"), tta=OPTS.get("tta"))
    return len(t), float(res.best_fap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True); ap.add_argument("--model", default="periods-v4")
    ap.add_argument("--per-bin", type=int, default=1500); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--tta", type=int); ap.add_argument("--weights")
    ap.add_argument("--name", help="calibration file name (default: the model name)"); ap.add_argument("--ranker")
    a = ap.parse_args()
    jobs = [(1000 * b + j, EDGES[b], EDGES[b + 1], a.model) for b in range(len(EDGES) - 1) for j in range(a.per_bin)]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")                     # CUDA cannot be shared by forked workers
    with ctx.Pool(a.procs, initializer=_init, initargs=(a.sample, {"device": a.device, "tta": a.tta, "weights": a.weights, "ranker": a.ranker})) as pool:
        res = pool.map(null_search, jobs, chunksize=8)
    bins = []
    for b in range(len(EDGES) - 1):
        n = np.array([r[0] for r in res]); f = np.array([r[1] for r in res])
        sel = (np.log10(n) >= EDGES[b]) & (np.log10(n) < EDGES[b + 1])
        s = np.sort(-np.log10(np.clip(f[sel], 1e-12, 1)))[::-1]          # significance, best first
        # exponential tail beyond the simulated range, fitted to the top 10% of null searches
        top = s[: max(10, len(s) // 10)]
        rate = 1.0 / max(np.mean(top - top[-1]), 1e-3)
        bins.append({"log10_n": [EDGES[b], EDGES[b + 1]], "count": int(sel.sum()), "null_sig": s.round(5).tolist(),
                     "tail": {"start": float(top[-1]), "frac": len(top) / len(s), "rate": float(rate)}})
        print(f"log10 N {EDGES[b]:.1f}-{EDGES[b + 1]:.1f}: {sel.sum():5d} noise searches; best fold FAP median "
              f"{10 ** -np.median(s):.3g}, 1% quantile {10 ** -np.quantile(s, 0.99):.3g}", flush=True)
    out = ROOT / "nn_periodogram" / "data" / f"calibration_{a.name or a.model}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({"model": a.name or a.model, "tta": a.tta, "ranker": bool(a.ranker), "bins": bins}, open(out, "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
