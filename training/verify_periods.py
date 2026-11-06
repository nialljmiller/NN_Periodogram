#!/usr/bin/env python3
"""Verify candidate periods with our own Lomb-Scargle, on the GPU, and write labels.

Two independent checks; a star is verified if either passes:

  LS  : a generalised (floating-mean) Lomb-Scargle periodogram over 1/300 to
        20 per day (5 samples per 1/T); one of the five highest peaks
        (separated by > 2/T) lies within 1/(3T) of 1, 2 or 1/2 times a
        candidate frequency, with Baluev false-alarm probability < 1e-3.
  PDM : phase dispersion (within-bin variance / total, 16 and 32 bins, the
        lower) at 1, 2 and 1/2 times each candidate, refined within +-1/T,
        must be lower than at every one of 2000 random trial frequencies
        (empirical p < 1/2000). Shape-agnostic, so it catches eclipses that a
        sine fit misses, and it prefers the true orbital period.

The label is the refined frequency of the check that passed (LS first), so a
label is always a fold our own analysis supports; method and ratio are recorded.

    python verify_periods.py --data DIR --shard real      # writes real_verified.json
    python verify_periods.py --data DIR --shard asassn    # records agreement only
    python verify_periods.py --check                      # compare with astropy
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nn_periodogram.classical import gls, pdm_theta  # noqa: E402

P_MAX, F_MAX = 300.0, 20.0
RATIOS = (1.0, 2.0, 0.5)


class ThermalGuard:
    """Pause at 80 C until the card cools to 70 C (same rule as training and the weather station)."""

    def __init__(self, device):
        self.device, self.next = device, 0.0

    def check(self):
        if self.device.type != "cuda" or time.time() < self.next:
            return
        try:
            if torch.cuda.temperature(self.device) >= 80:
                print(f"GPU at {torch.cuda.temperature(self.device)} C, pausing", flush=True)
                while torch.cuda.temperature(self.device) > 70:
                    time.sleep(10)
        except Exception:
            pass
        self.next = time.time() + 2


def baluev(z, t, y, dy):
    from astropy.timeseries.periodograms.lombscargle._statistics import false_alarm_probability
    return float(false_alarm_probability(z, F_MAX, t, y, dy, normalization="standard", method="baluev"))


def pdm_check(t, y, cands, device, n_random=2000, seed=0):
    span = np.ptp(t)
    rng = np.random.default_rng(seed)
    rand = np.exp(rng.uniform(np.log(1 / P_MAX), np.log(F_MAX), n_random))
    grids, tags = [], []
    for fc in cands:
        for r in RATIOS:
            fine = np.linspace(r * fc - 1 / span, r * fc + 1 / span, 201)
            fine = fine[(fine > 1 / P_MAX) & (fine < F_MAX)]
            if len(fine):                                # candidates outside the search range give nothing
                grids.append(fine); tags += [r] * len(fine)
    if not grids:
        return None
    fine = np.concatenate(grids)
    th = pdm_theta(t, y, np.concatenate([rand, fine]), device)       # one batched call
    null, th = th[:n_random], th[n_random:]
    k = int(np.argmin(th))
    return {"freq": float(fine[k]), "theta": float(th[k]), "ratio": tags[k],
            "p": float((np.sum(null <= th[k]) + 1) / (n_random + 1))}


def verify_one(t, y, dy, cands, device):
    span = np.ptp(t)
    freqs = np.arange(1 / P_MAX, F_MAX, 1 / (5 * span))
    p = gls(t, y, dy, freqs, device)
    top = []
    cand = np.argpartition(p, -400)[-400:]           # partial sort: only the strongest bins matter
    for i in cand[np.argsort(p[cand])[::-1]]:
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
    ap.add_argument("--part", default="0/1", help="k/n: process every n-th star starting at k")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.check:
        return check(device)
    d = dict(np.load(Path(a.data) / f"{a.shard}.npz"))   # load arrays once: indexing an NpzFile re-reads the whole array
    meta = json.load(open(Path(a.data) / f"{a.shard}_meta.json"))
    off = d["offsets"]
    k, n = (int(x) for x in a.part.split("/"))
    out, t0 = [], time.time()
    guard = ThermalGuard(device)
    for i, row in enumerate(meta):
        if i % n != k:
            continue
        guard.check()
        s, e = off[i], off[i + 1]
        t, y, dy = d["time"][s:e], d["mag"][s:e].astype(float), d["err"][s:e].astype(float)
        if len(t) > 20000:     # checking a period needs far fewer points than a 2-min TESS light curve has
            keep = np.sort(np.random.default_rng(i).choice(len(t), 20000, replace=False))
            t, y, dy = t[keep], y[keep], dy[keep]
        res = verify_one(t, y, dy, row["candidate_freqs"], device)
        if res is not None and res["fap"] < 1e-3:
            rec = {"verified": True, "method": "ls", **res}
        else:
            pd_ = pdm_check(t, y, row["candidate_freqs"], device)
            ok = pd_ is not None and pd_["p"] < 1 / 2000
            rec = {"verified": ok, "method": "pdm" if ok else None, **(pd_ or {}), "ls": res}
        out.append({"index": i, "sourceid": row["sourceid"], **rec})
        if i % 5000 == 0:
            n_ok = sum(o["verified"] for o in out)
            print(f"{i}/{len(meta)}  verified {n_ok}  {time.time() - t0:.0f}s", flush=True)
    suffix = "" if n == 1 else f".part{k}of{n}"
    json.dump(out, open(Path(a.data) / f"{a.shard}_verified{suffix}.json", "w"))
    n_ok = sum(o["verified"] for o in out)
    meth = [o["method"] for o in out if o["verified"]]
    print(f"done: {n_ok}/{len(out)} verified (ls {meth.count('ls')}, pdm {meth.count('pdm')})")


if __name__ == "__main__":
    main()
