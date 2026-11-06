#!/usr/bin/env python3
"""Which cheap proposers give the network the true period? Candidate recall and CPU time
per proposer set, on the exported benchmark sample (export_bench_sample.py)."""
import argparse, json, sys, time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nn_periodogram.classical import gls, gls_fast, mhls_fast  # noqa: E402

SIDEREAL = 1 / 0.99726957
MULT = (1.0, 2.0, 0.5, 3.0, 1 / 3)


def top(score, n, sep=5):
    picked = []
    for i in np.argsort(score)[::-1]:
        if all(abs(i - j) > sep for j in picked):
            picked.append(int(i))
        if len(picked) == n:
            break
    return picked


def propose(t, m, e, spec, fmax=10.0, oversample=3.0):
    """spec: list of (method, n_peaks) with method in ls, mh2, mh3; flags: 'alias'."""
    span = np.ptp(t); df = 1 / (oversample * span); f0 = 2 / span; nf = int((fmax - f0) / df)
    cands = []
    for method, n in spec["proposers"]:
        p = gls_fast(t, m, e, f0, df, nf) if method == "ls" else mhls_fast(t, m, e, f0, df, nf, int(method[2:]))
        for i in top(p, n):
            if 0 < i < nf - 1:           # parabola through the grid peak: sub-grid precision for free
                a, b, c = p[i - 1], p[i], p[i + 1]
                den = a - 2 * b + c
                i = i + (0.5 * (a - c) / den if den < 0 else 0.0)
            cands.append(f0 + df * i)
    cands = np.array(cands)
    if spec.get("alias"):
        cands = np.concatenate([cands, np.abs(cands + SIDEREAL), np.abs(cands - SIDEREAL)])
    cands = np.array([c * r for c in cands for r in MULT])
    return np.unique(np.round(cands[(cands >= f0) & (cands <= fmax)], 12))


SPECS = {
    "ls10": {"proposers": [("ls", 10)]},
    "ls20": {"proposers": [("ls", 20)]},
    "ls40": {"proposers": [("ls", 40)]},
    "ls10+mh2_10": {"proposers": [("ls", 10), ("mh2", 10)]},
    "ls10+mh3_10": {"proposers": [("ls", 10), ("mh3", 10)]},
    "ls20+mh3_20": {"proposers": [("ls", 20), ("mh3", 20)]},
    "ls10+alias": {"proposers": [("ls", 10)], "alias": True},
    "ls10+mh3_10+alias": {"proposers": [("ls", 10), ("mh3", 10)], "alias": True},
}


def work(args):
    torch.set_num_threads(1)
    t, m, e, p_true, names = args
    out = {}
    for name in names:
        t0 = time.time(); c = propose(t, m, e, SPECS[name]); dt = time.time() - t0
        out[name] = (bool(np.any(np.abs(1 / c / p_true - 1) < 0.01)), dt, len(c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True); ap.add_argument("--specs", nargs="+", default=list(SPECS))
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--out")
    a = ap.parse_args()
    d = np.load(a.sample); meta = json.loads(str(d["meta"])); off = d["offsets"]
    T, M, E = d["time"], d["mag"].astype(float), d["err"].astype(float)
    jobs = [(T[off[i]:off[i + 1]], M[off[i]:off[i + 1]], E[off[i]:off[i + 1]], r["p_true"], a.specs) for i, r in enumerate(meta)]
    with Pool(a.procs) as pool:
        res = pool.map(work, jobs, chunksize=4)
    STRICT = __import__("benchmark_surveys").STRICT
    agg = defaultdict(list)
    for r, o in zip(meta, res):
        for name, v in o.items():
            agg[(r["survey"], name)].append(v)
            if r["survey"] == "ASAS-SN" and r["cls"] in STRICT:
                agg[("ASAS-SN strict", name)].append(v)
    for survey in ("OGLE", "ASAS-SN strict", "Kepler", "TESS", "VVV"):
        print(f"== {survey}")
        for name in a.specs:
            v = agg[(survey, name)]
            print(f"  {name:<20} recall {100 * np.mean([x[0] for x in v]):5.1f}%   {1000 * np.mean([x[1] for x in v]):7.1f} ms/star (1 CPU core)   {np.mean([x[2] for x in v]):5.0f} candidates")
    if a.out:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in agg.items()}, open(a.out, "w"))


if __name__ == "__main__":
    main()
