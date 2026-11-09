#!/usr/bin/env python3
"""Does combining a star's light curves help? ASAS-SN validation stars observed in both V and g:
v4 on each band alone vs on the two bands merged (each band median-subtracted and scaled to
unit robust scatter, so the shapes line up)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_periodogram as nnp  # noqa: E402
from benchmark import verdict  # noqa: E402
from train import mapped  # noqa: E402

D = sys.argv[1]; W = sys.argv[2]
meta = json.load(open(f"{D}/asassn_vsx_meta.json")); d = mapped(f"{D}/asassn_vsx"); off = d["offsets"]
by_star = defaultdict(dict)
for i, r in enumerate(meta):
    if r["split"] == "val" and r.get("candidate_freqs"):
        sid, band = r["sourceid"].rsplit(":", 1)
        by_star[sid][band] = i
pairs = [(s, b) for s, b in by_star.items() if "V" in b and "g" in b]
rng = np.random.default_rng(1); rng.shuffle(pairs); pairs = pairs[:400]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tally = defaultdict(lambda: defaultdict(int)); n = 0
for sid, bands in pairs:
    lcs = {}
    for band, i in bands.items():
        t, m, e = (np.asarray(d[k][off[i]:off[i + 1]], float) for k in ("time", "mag", "err"))
        lcs[band] = (t, m, e)
    p_true = 1 / meta[bands["V"]]["candidate_freqs"][0]
    cls = meta[bands["V"]].get("cls", "")
    span = max(np.ptp(np.concatenate([lcs["V"][0], lcs["g"][0]])), 1)
    if p_true > span / 3 or 1 / p_true > 10:
        continue
    out = {}
    for band in ("V", "g"):
        t, m, e = lcs[band]
        out[band] = verdict(nnp.periodogram(t, m, e, device=dev, weights=W).best_period, p_true) if len(t) >= 10 else "wrong"
    ts, ms, es = [], [], []
    for band in ("V", "g"):
        t, m, e = lcs[band]
        s = 1.4826 * np.median(np.abs(m - np.median(m))) or 1.0
        ts.append(t); ms.append((m - np.median(m)) / s); es.append(e / s)
    t, m, e = np.concatenate(ts), np.concatenate(ms), np.concatenate(es)
    o = np.argsort(t)
    out["V+g"] = verdict(nnp.periodogram(t[o], m[o], e[o], device=dev, weights=W).best_period, p_true)
    out["best single"] = "correct" if "correct" in (out["V"], out["g"]) else out["V"]
    n += 1
    for k, v in out.items():
        tally[k][v] += 1
        if cls in ("E", "EA", "EB", "EW", "RRAB", "RRC", "DCEP", "DSCT", "HADS", "ROT", "SR"):
            tally[f"{k} [{cls}]"][v] += 0
print(f"{n} ASAS-SN validation stars with both V and g light curves")
for k in ("V", "g", "best single", "V+g"):
    c = tally[k]; print(f"  {k:12s} correct {100 * c['correct'] / n:5.1f}%   harmonic {100 * c['harmonic'] / n:5.1f}%")
