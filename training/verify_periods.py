#!/usr/bin/env python3
"""Verify candidate periods with our own Lomb-Scargle, on the GPU, and write labels.

For each star in a shard from build_dataset.py, a generalised (floating-mean)
Lomb-Scargle periodogram is computed on the quality-cut Ks light curve over
1/300 to 20 per day, 5 samples per 1/T. A candidate is verified when one of
our five highest peaks (separated by > 2/T) lies within 1/(3T) of 1, 2 or 1/2
times a candidate frequency and its Baluev false-alarm probability is below
1e-3. The label is our refined peak frequency, so a label is always a fold
that our own periodogram supports; the relation to the catalogue (1, 2, 1/2)
is recorded.

    python verify_periods.py --data DIR --shard real      # writes real_verified.json
    python verify_periods.py --data DIR --shard asassn    # records agreement only
    python verify_periods.py --check                      # compare with astropy
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

P_MAX, F_MAX = 300.0, 20.0
RATIOS = (1.0, 2.0, 0.5)


def gls(t, y, dy, freqs, device, chunk=32768):
    """Generalised Lomb-Scargle power (standard normalisation, floating mean)."""
    w = 1.0 / dy ** 2
    w = w / w.sum()
    y = y - (w * y).sum()
    tt = torch.tensor(t - t.min(), dtype=torch.float64, device=device)
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    yy = float((w * y * y).sum())
    out = []
    for s in range(0, len(freqs), chunk):
        f = torch.tensor(freqs[s:s + chunk], dtype=torch.float64, device=device)
        ph = torch.remainder(f[:, None] * tt[None, :], 1.0).float() * (2 * np.pi)
        c, sn = torch.cos(ph), torch.sin(ph)
        C, S = c @ wt, sn @ wt
        YC, YS = c @ (wt * yt), sn @ (wt * yt)
        CC = (c * c) @ wt - C * C
        SS = 1.0 - (c * c) @ wt - S * S
        CS = (c * sn) @ wt - C * S
        D = CC * SS - CS * CS
        out.append(((SS * YC ** 2 + CC * YS ** 2 - 2 * CS * YC * YS) / (yy * D)).cpu().numpy())
    return np.concatenate(out)


def baluev(z, t, y, dy):
    from astropy.timeseries.periodograms.lombscargle._statistics import false_alarm_probability
    return float(false_alarm_probability(z, F_MAX, t, y, dy, normalization="standard", method="baluev"))


def verify_one(t, y, dy, cands, device):
    span = np.ptp(t)
    freqs = np.arange(1 / P_MAX, F_MAX, 1 / (5 * span))
    p = gls(t, y, dy, freqs, device)
    top = []
    for i in np.argsort(p)[::-1]:
        if all(abs(freqs[i] - freqs[j]) > 2 / span for j in top):
            top.append(i)
        if len(top) == 5:
            break
    for rank, i in enumerate(top):
        for fc in cands:
            for r in RATIOS:
                if abs(freqs[i] - r * fc) < 1 / (3 * span):
                    fine = np.linspace(freqs[i] - 1 / span, freqs[i] + 1 / span, 401)
                    pf = gls(t, y, dy, fine, device)
                    k = int(np.argmax(pf))
                    return {"freq": float(fine[k]), "fap": baluev(pf[k], t, y, dy), "ratio": r, "rank": rank}
    return None


def check(device):
    from astropy.timeseries import LombScargle
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(55000, 58400, 150)); dy = rng.uniform(0.01, 0.05, 150)
    y = 14 + 0.2 * np.sin(2 * np.pi * t / 0.5078) + rng.normal(0, dy)
    f = np.linspace(0.01, 5, 20000)
    a = LombScargle(t, y, dy).power(f)
    b = gls(t, y, dy, f, device)
    print(f"max |astropy - ours| = {np.abs(a - b).max():.2e}; peak {1 / f[np.argmax(b)]:.5f} d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--shard"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.check:
        return check(device)
    d = np.load(Path(a.data) / f"{a.shard}.npz")
    meta = json.load(open(Path(a.data) / f"{a.shard}_meta.json"))
    off = d["offsets"]
    out, t0 = [], time.time()
    for i, row in enumerate(meta):
        s, e = off[i], off[i + 1]
        res = verify_one(d["time"][s:e], d["mag"][s:e].astype(float), d["err"][s:e].astype(float),
                         row["candidate_freqs"], device)
        ok = res is not None and res["fap"] < 1e-3
        out.append({"index": i, "sourceid": row["sourceid"], "verified": ok, **(res or {})})
        if i % 5000 == 0:
            n_ok = sum(o["verified"] for o in out)
            print(f"{i}/{len(meta)}  verified {n_ok}  {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(Path(a.data) / f"{a.shard}_verified.json", "w"))
    n_ok = sum(o["verified"] for o in out)
    print(f"done: {n_ok}/{len(out)} verified; ratios {dict(zip(*np.unique([o['ratio'] for o in out if o['verified']], return_counts=True)))}")


if __name__ == "__main__":
    main()
