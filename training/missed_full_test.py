#!/usr/bin/env python3
"""Are the hybrid's 'missing' stars recoverable from the same light curve? Run v4's full
search (network scores every grid frequency, no Lomb-Scargle proposals) on the validation
stars whose true period was not among the hybrid's candidates."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_periodogram as nnp  # noqa: E402
from benchmark import verdict  # noqa: E402

sample, errors, W = sys.argv[1], sys.argv[2], sys.argv[3]
d = dict(np.load(sample)); off = d["offsets"]
rows = [r for r in json.load(open(errors)) if r["v"] == "missing"]
dev = torch.device("cuda")
by = {}
for r in rows:
    i = r["i"]; t, m, e = (d[k][off[i]:off[i + 1]].astype(float) for k in ("time", "mag", "err"))
    if len(t) > 20000:
        continue
    res = nnp.periodogram(t, m, e, device=dev, weights=W, method="full", tta=1)
    by.setdefault(r["survey"], Counter())[verdict(res.best_period, r["p_true"])] += 1
for s, c in by.items():
    n = sum(c.values())
    print(f"{s:8s} {n:4d} hybrid-missed stars: full search correct {100 * c['correct'] / n:5.1f}%  harmonic {100 * c['harmonic'] / n:5.1f}%")
