#!/usr/bin/env python3
"""Classical baselines on exactly the stars and grid of benchmark.py: Lomb-Scargle and PDM.

Same star selection (seed), same frequency grid (2/T to fmax, 3 samples per 1/T),
same refinement (+-1 grid step, 41 points around the best 5 peaks) and the same
verdicts, so the numbers sit directly beside the NN models'.

    python benchmark_classical.py --data DIR --method ls --out bench_ls.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import verdict  # noqa: E402
from verify_periods import gls, pdm_theta  # noqa: E402


def search(t, m, e, method, fmax, device, oversample=3.0, n_peaks=5, refine=20):
    span = np.ptp(t)
    df = 1 / (oversample * span)
    f = np.arange(2 / span, fmax, df)
    score = gls(t, m, e, f, device) if method == "ls" else -pdm_theta(t, m, f, device)
    order, picked = np.argsort(score)[::-1], []
    for i in order:
        if all(abs(i - j) > 5 for j in picked):
            picked.append(i)
        if len(picked) == n_peaks:
            break
    best = None
    for i in picked:
        fine = np.linspace(f[i] - df, f[i] + df, 2 * refine + 1)
        s = gls(t, m, e, fine, device) if method == "ls" else -pdm_theta(t, m, fine, device)
        k = int(np.argmax(s))
        if best is None or s[k] > best[1]:
            best = (fine[k], s[k])
    return 1 / best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--method", choices=["ls", "pdm"], required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--fmax", type=float, default=5.0); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = dict(np.load(f"{a.data}/asassn.npz")); meta = json.load(open(f"{a.data}/asassn_meta.json"))
    off = d["offsets"]
    idx = np.random.default_rng(a.seed).choice(len(meta), min(a.n, len(meta)), replace=False)
    rows, t0 = [], time.time()
    for i in idx:
        t, m, e = d["time"][off[i]:off[i + 1]], d["mag"][off[i]:off[i + 1]].astype(float), d["err"][off[i]:off[i + 1]].astype(float)
        p_true = meta[i]["period"]
        if p_true > np.ptp(t) / 3:
            continue
        p = search(t, m, e, a.method, a.fmax, device)
        rows.append({"sourceid": meta[i]["sourceid"], "p_true": p_true, "p": p, "n": len(t), "verdict": verdict(p, p_true)})
    json.dump(rows, open(a.out, "w"))
    v = [r["verdict"] for r in rows]
    print(a.method, {x: f"{v.count(x)} ({100 * v.count(x) / len(v):.1f}%)" for x in ("correct", "harmonic", "wrong")},
          f"of {len(v)} stars, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
