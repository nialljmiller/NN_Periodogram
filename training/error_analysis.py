#!/usr/bin/env python3
"""Why does the hybrid search miss? Every star of the exported validation sample is run
through periodogram(); each miss is classified:

  missing      the true period is not among the scored candidates (proposer problem)
  harmonic     a k/n multiple of the true period was preferred (k, n <= 4)
  daily alias  |f_found - f_true| ~ m / sidereal day (m = 1, 2)
  yearly alias |f_found - f_true| ~ 1 / year
  other        anything else (often the true period scored, but lower)

and, for misses where the truth was a candidate, how its fold FAP compared with the winner's.

    python error_analysis.py --sample bench_val_sample.npz --out errors.json [--proposers ls mhls]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_surveys import STRICT  # noqa: E402

SID = 1 / 0.99726957
DATA = None


def _init(sample):
    global DATA
    import torch
    torch.set_num_threads(1)
    d = dict(np.load(sample))
    DATA = (d, json.loads(str(d["meta"])))


def classify(f_found, f_true, span):
    tol = 1.5 / (3 * span) + 0.01 * f_true
    if abs(f_found / f_true - 1) < 0.01:
        return "correct"
    for k in range(1, 5):
        for n in range(1, 5):
            if (k, n) != (1, 1) and abs(f_found - f_true * k / n) < 0.01 * f_true * k / n:
                return "harmonic"
    d = abs(f_found - f_true)
    if any(abs(d - m * SID) < tol for m in (1, 2)):
        return "daily alias"
    if abs(d - 1 / 365.25) < tol:
        return "yearly alias"
    return "other"


def job(args):
    import nn_periodogram as nnp
    i, kwargs = args
    d, meta = DATA
    off = d["offsets"]
    t, m, e = (d[k][off[i]:off[i + 1]].astype(float) for k in ("time", "mag", "err"))
    r = meta[i]; f_true = 1 / r["p_true"]
    res = nnp.periodogram(t, m, e, device="cpu", **kwargs)
    span = np.ptp(t)
    v = classify(1 / res.best_period, f_true, span)
    hit = np.abs(res.freq / f_true - 1) < 0.01
    out = {"i": i, "survey": r["survey"], "cls": r["cls"], "n": len(t), "span": float(span), "p_true": r["p_true"],
           "p": res.best_period, "fap": res.best_fap, "v": v, "in_cands": bool(hit.any()), "n_cands": len(res.freq)}
    if hit.any():
        out["true_fold_fap"] = float(res.fap[hit].min())
        out["best_fold_fap"] = float(res.best_fap_fold)
    if v != "correct" and not hit.any():
        out["v"] = "missing"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--proposers", nargs="+", default=None); ap.add_argument("--n-candidates", type=int, default=None)
    ap.add_argument("--weights"); ap.add_argument("--model", default=None)
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--surveys", nargs="+")
    a = ap.parse_args()
    kwargs = {}
    if a.proposers:
        kwargs["proposers"] = tuple(a.proposers)
    if a.n_candidates:
        kwargs["n_candidates"] = a.n_candidates
    if a.weights:
        kwargs["weights"] = a.weights
    if a.model:
        kwargs["model"] = a.model
    meta = json.loads(str(np.load(a.sample)["meta"]))
    with Pool(a.procs, initializer=_init, initargs=(a.sample,)) as p:
        idx = [i for i, r in enumerate(meta) if not a.surveys or r["survey"] in a.surveys]
        rows = p.map(job, [(i, kwargs) for i in idx], chunksize=4)
    json.dump(rows, open(a.out, "w"))
    groups = defaultdict(list)
    for r in rows:
        groups[r["survey"]].append(r)
        if r["survey"] == "ASAS-SN" and r["cls"] in STRICT:
            groups["ASAS-SN strict"].append(r)
    for s in ("OGLE", "Kepler", "TESS", "VVV", "ASAS-SN strict", "ASAS-SN"):
        rs = groups[s]; c = Counter(r["v"] for r in rs); n = len(rs)
        if not n:
            continue
        print(f"{s:<15} n={n:4d}  " + "  ".join(f"{k} {100 * c[k] / n:5.1f}%" for k in
              ("correct", "missing", "harmonic", "daily alias", "yearly alias", "other")), flush=True)


if __name__ == "__main__":
    main()
