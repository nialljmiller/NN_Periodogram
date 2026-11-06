#!/usr/bin/env python3
"""Write the benchmark's star sample (same split/seed/filters as benchmark_surveys) to one small npz,
so proposer studies can run anywhere without the full shards."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_surveys import SURVEYS, stars  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=600)
ap.add_argument("--seed", type=int, default=11); ap.add_argument("--fmax", type=float, default=10.0)
a = ap.parse_args()
T, M, E, off, meta = [], [], [], [0], []
for survey, prefix in SURVEYS.items():
    for name, i, lo, hi, p_true, cls, d in stars(a.data, prefix, a.n, a.seed, a.split):
        t = np.asarray(d["time"][lo:hi], float)
        if len(t) >= 10 and p_true < np.ptp(t) / 3 and 1 / p_true < a.fmax:
            T.append(t); M.append(np.asarray(d["mag"][lo:hi], float)); E.append(np.asarray(d["err"][lo:hi], float))
            off.append(off[-1] + len(t)); meta.append({"survey": survey, "shard": name, "index": int(i), "cls": cls, "p_true": p_true})
np.savez(a.out, time=np.concatenate(T), mag=np.concatenate(M).astype(np.float32), err=np.concatenate(E).astype(np.float32),
         offsets=np.array(off), meta=json.dumps(meta))
print(len(meta), "stars,", off[-1], "points")
