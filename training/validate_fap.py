#!/usr/bin/env python3
"""Check the calibrated search FAP on fresh noise (seeds not used for calibration) and on
injected signals: false-positive rate should match the threshold; power should stay high."""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_fap as C  # noqa: E402


def job(args):
    import nn_periodogram as nnp
    from train import signal
    seed, lo, hi, snr = args
    rng = np.random.default_rng(seed)
    n = int(round(10 ** rng.uniform(lo, hi)))
    pool = [c for c in C.CADENCES if len(c[0]) >= n] or C.CADENCES
    t, e = pool[rng.integers(len(pool))]
    k = np.sort(rng.choice(len(t), min(n, len(t)), replace=False))
    t, e = t[k].astype(float), np.clip(e[k].astype(float), 1e-4, None)
    m = 15 + rng.normal(0, 1, len(t)) * e
    if snr:
        P = float(np.exp(rng.uniform(np.log(0.15), np.log(np.ptp(t) / 4))))
        s = signal(rng.choice(["sine", "rr", "eb"]), np.mod(t / P + rng.uniform(), 1), rng)
        m += snr * np.median(e) * (s - s.mean()) / (s.std() + 1e-12)
    r = nnp.periodogram(t, m, e, device="cpu")
    return lo, snr, r.best_fap, r.best_fap_fold


if __name__ == "__main__":
    jobs = [(500000 + 100 * b + j, C.EDGES[b], C.EDGES[b + 1], snr)
            for b in range(len(C.EDGES) - 1) for snr in (0.0, 0.5, 1.0) for j in range(60)]
    with Pool(8, initializer=C._init, initargs=(sys.argv[1], {})) as p:
        res = p.map(job, jobs, chunksize=4)
    for snr in (0.0, 0.5, 1.0):
        print(f"\n== {'pure noise' if snr == 0 else f'signal, rms = {snr} x noise'}")
        for b in range(len(C.EDGES) - 1):
            f = np.array([r[2] for r in res if r[0] == C.EDGES[b] and r[1] == snr])
            g = np.array([r[3] for r in res if r[0] == C.EDGES[b] and r[1] == snr])
            print(f"  log10 N {C.EDGES[b]:.1f}-{C.EDGES[b + 1]:.1f}: search FAP<0.01 {100 * np.mean(f < 0.01):5.1f}%  <0.1 {100 * np.mean(f < 0.1):5.1f}%"
                  f"   | fold FAP<0.01 {100 * np.mean(g < 0.01):5.1f}%  <0.1 {100 * np.mean(g < 0.1):5.1f}%")
