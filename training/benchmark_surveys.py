#!/usr/bin/env python3
"""Per-survey benchmark on held-out stars (test split, never trained on).

For each survey, up to --n stars from the test split of its shards; truth is the
catalogue's own period (OGLE, Kepler, TESS and VSX periods), not our LS/PDM-verified
label, so the classical methods get no head start. Each method searches the same
grid (2/T to fmax, 3 samples per 1/T) and is scored correct (within 1%), harmonic
(k/n x P, k, n <= 4) or wrong. Search time is reported per method.

    python benchmark_surveys.py --data SHARDS --methods ls pdm hybrid:periods-v1:weights.pt --out results.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_periodogram  # noqa: E402
from benchmark import verdict  # noqa: E402
from benchmark_classical import search  # noqa: E402

SURVEYS = {
    "OGLE": "ogle_",
    "ASAS-SN": "asassn_vsx",
    "Kepler": "kepler_eb",
    "TESS": "tess_eb",
    "VVV (ASAS-SN periods)": "vvv_asassn",
}


def stars(data, prefix, n, seed):
    """(t, m, e, true period, class) for up to n test-split stars from shards starting with prefix."""
    out = []
    for meta_path in sorted(Path(data).glob(f"{prefix}*_meta.json")):
        name = meta_path.name[:-len("_meta.json")]
        if name.endswith("_verified") or not (Path(data) / f"{name}.npz").exists():
            continue
        meta = json.load(open(meta_path))
        idx = [i for i, r in enumerate(meta) if r["split"] == "test" and r.get("candidate_freqs")]
        if not idx:
            continue
        d = dict(np.load(Path(data) / f"{name}.npz"))
        off = d["offsets"]
        for i in idx:
            out.append((d["time"][off[i]:off[i + 1]], d["mag"][off[i]:off[i + 1]].astype(float),
                        d["err"][off[i]:off[i + 1]].astype(float), 1 / meta[i]["candidate_freqs"][0],
                        meta[i].get("cls", meta[i].get("type", name))))
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(out))[:n]
    return [out[k] for k in pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["ls", "pdm"])
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--fmax", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = json.load(open(a.out)) if Path(a.out).exists() else {}
    for survey, prefix in SURVEYS.items():
        sample = [s for s in stars(a.data, prefix, a.n, a.seed) if s[3] < np.ptp(s[0]) / 3 and 1 / s[3] < a.fmax]
        if not sample:
            continue
        for method in a.methods:
            key = f"{survey} | {method.split(':')[0]}{':' + method.split(':')[1] if ':' in method else ''}"
            if key in results:
                continue
            tally, t_total = defaultdict(int), 0.0
            for t, m, e, p_true, _ in sample:
                t0 = time.time()
                if method in ("ls", "pdm"):
                    p = search(t, m, e, method, a.fmax, device)
                else:
                    _, model, weights = method.split(":")
                    p = nn_periodogram.periodogram(t, m, err=e, fmax=a.fmax, model=model, weights=weights,
                                                   device=device, method="hybrid").best_period
                t_total += time.time() - t0
                tally[verdict(p, p_true)] += 1
            n = len(sample)
            results[key] = {"n": n, "time_s": round(t_total, 1),
                            **{k: round(100 * tally[k] / n, 1) for k in ("correct", "harmonic", "wrong")}}
            print(f"{key:<40} n={n:<5} correct {results[key]['correct']:5.1f}%  harmonic {results[key]['harmonic']:5.1f}%"
                  f"  wrong {results[key]['wrong']:5.1f}%  {t_total:7.1f}s", flush=True)
            json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
