#!/usr/bin/env python3
"""Full-search benchmark on ASAS-SN-matched PRIMVS stars (never used in training).

Runs periodogram() with a given model on the same stars and grid and scores the
reported period against ASAS-SN's: correct (within 1%), harmonic (within 1% of
k/n x P, k, n <= 4), or wrong.

    python benchmark.py --data /media/data4/nn_periodogram_data --model legacy-2022 --weights w.pt --n 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nn_periodogram  # noqa: E402
from nn_periodogram import hub  # noqa: E402


def verdict(p, p_true):
    if abs(p / p_true - 1) < 0.01:
        return "correct"
    for k in range(1, 5):
        for n in range(1, 5):
            if (k, n) != (1, 1) and abs(p / (p_true * k / n) - 1) < 0.01:
                return "harmonic"
    return "wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--fmax", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--method", choices=["full", "hybrid"], default=None)
    a = ap.parse_args()
    d = dict(np.load(f"{a.data}/asassn.npz")); meta = json.load(open(f"{a.data}/asassn_meta.json"))
    off = d["offsets"]
    idx = np.random.default_rng(a.seed).choice(len(meta), min(a.n, len(meta)), replace=False)
    rows, t0 = [], time.time()
    t_search = 0.0
    for k, i in enumerate(idx):
        t, m = d["time"][off[i]:off[i + 1]], d["mag"][off[i]:off[i + 1]]
        p_true = meta[i]["period"]
        if p_true > np.ptp(t) / 3:
            continue
        ts = time.time()
        res = nn_periodogram.periodogram(t, m, fmax=a.fmax, model=a.model, weights=a.weights, device="cuda",
                                         method=a.method, err=d["err"][off[i]:off[i + 1]] if a.method == "hybrid" else None)
        t_search += time.time() - ts
        rows.append({"sourceid": meta[i]["sourceid"], "p_true": p_true, "p": res.best_period,
                     "fap": res.best_fap, "n": len(t), "verdict": verdict(res.best_period, p_true),
                     "aliased_note": bool(res.note)})
        if k % 20 == 0:
            print(f"{k}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)
    json.dump(rows, open(a.out, "w"))
    v = [r["verdict"] for r in rows]
    print({x: f"{v.count(x)} ({100 * v.count(x) / len(v):.1f}%)" for x in ("correct", "harmonic", "wrong")},
          f"of {len(v)} stars; search time {t_search:.0f}s")


if __name__ == "__main__":
    main()
