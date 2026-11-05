#!/usr/bin/env python3
"""Per-survey benchmark on held-out stars, one row per star and method.

Stars come from the --split ("val" for choosing a model, "test" once for the
paper) of each survey's shards; truth is the catalogue's own period (OGLE,
Kepler, TESS and VSX periods; ASAS-SN periods for the VVV stars), never our
LS/PDM-verified label. Every method searches the same grid (2/T to fmax,
3 samples per 1/T). Verdicts: correct (within 1%), harmonic (k/n x P, k, n <= 4)
or wrong.

Methods:
    ls, pdm                         classical, best refined peak
    hybrid:NAME:WEIGHTS[:ls]        nn_periodogram hybrid (":ls" = LS-only proposals)

For each hybrid star the rows record both tie-break settings (prefer_shorter
off and on, from the same scores) and the candidate recall (was the true period,
or 2P or P/2, among the candidates the network chose from?).

    python benchmark_surveys.py --data SHARDS --split val --out rows.jsonl --methods ls pdm hybrid:periods-v1:best.pt
    python benchmark_surveys.py --summary rows.jsonl
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
from train import mapped  # noqa: E402

SURVEYS = {
    "OGLE": "ogle_",
    "ASAS-SN": "asassn_vsx",
    "Kepler": "kepler_eb",
    "TESS": "tess_eb",
    "VVV": "vvv_asassn",
}
# Strictly periodic VSX/GCVS classes: the headline ASAS-SN numbers use only these
# (SR, ROT, MISC, L etc. have catalogue periods that a fold need not confirm).
STRICT = ("E", "EA", "EB", "EW", "EC", "ED", "ESD", "RR", "RRAB", "RRC", "RRD", "DCEP", "CEP", "DSCT", "HADS",
          "CW", "CWA", "CWB", "ACEP", "T2CEP", "RRLYR", "ECL", "DPV", "BCEP", "SXPHE")


def stars(data, prefix, n, seed, split):
    out = []
    for meta_path in sorted(Path(data).glob(f"{prefix}*_meta.json")):
        name = meta_path.name[:-len("_meta.json")]
        if name.endswith("_verified") or not (Path(data) / f"{name}.npz").exists():
            continue
        meta = json.load(open(meta_path))
        idx = [i for i, r in enumerate(meta) if r["split"] == split and r.get("candidate_freqs")]
        if not idx:
            continue
        d = mapped(Path(data) / name)
        off = d["offsets"]
        for i in idx:
            cls = str(meta[i].get("cls") or name.split("_")[-1]).upper()
            out.append((name, i, off[i], off[i + 1], 1 / meta[i]["candidate_freqs"][0], cls, d))
    pick = np.random.default_rng(seed).permutation(len(out))[:n]
    return [out[k] for k in pick]


def near(f_cands, p, tol=0.01):
    return bool(np.any(np.abs(1 / np.asarray(f_cands) / p - 1) < tol))


def run(a):
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    done = set()
    if Path(a.out).exists():
        done = {(r["survey"], r["method"]) for r in map(json.loads, open(a.out))}
    for survey, prefix in SURVEYS.items():
        sample = []
        for name, i, lo, hi, p_true, cls, d in stars(a.data, prefix, a.n, a.seed, a.split):
            t = np.asarray(d["time"][lo:hi], float)
            if len(t) >= 10 and p_true < np.ptp(t) / 3 and 1 / p_true < a.fmax:
                sample.append((name, i, t, np.asarray(d["mag"][lo:hi], float), np.asarray(d["err"][lo:hi], float),
                               p_true, cls))
        for method in a.methods:
            if (survey, method) in done or not sample:
                continue
            t_total, tally = 0.0, defaultdict(int)
            with open(a.out, "a") as fout:
                for name, i, t, m, e, p_true, cls in sample:
                    row = {"survey": survey, "method": method, "shard": name, "index": int(i), "cls": cls,
                           "n": len(t), "p_true": p_true}
                    t0 = time.time()
                    if method in ("ls", "pdm"):
                        row["p"] = search(t, m, e, method, a.fmax, device)
                    else:
                        parts = method.split(":")
                        proposers = ("ls",) if len(parts) > 3 and parts[3] == "ls" else ("ls", "pdm")
                        res = nn_periodogram.periodogram(t, m, err=e, fmax=a.fmax, model=parts[1], weights=parts[2],
                                                         device=device, method="hybrid", prefer_shorter=False,
                                                         proposers=proposers)
                        best, best_f = res.peaks[0]
                        half = [q for q in res.peaks if abs(q[0] - best / 2) < 0.01 * best / 2]
                        row["p"] = best
                        row["p_tie"] = half[0][0] if half and half[0][1] <= 1.25 * best_f else best
                        row["recall"] = near(res.freq, p_true)
                        row["recall_harm"] = row["recall"] or near(res.freq, 2 * p_true) or near(res.freq, p_true / 2)
                        row["v_tie"] = verdict(row["p_tie"], p_true)
                    row["time_s"] = time.time() - t0
                    row["v"] = verdict(row["p"], p_true)
                    t_total += row["time_s"]; tally[row["v"]] += 1
                    fout.write(json.dumps(row) + "\n")
            n = len(sample)
            print(f"{survey:<8} {method:<55} n={n:<5} correct {100 * tally['correct'] / n:5.1f}%  "
                  f"harmonic {100 * tally['harmonic'] / n:5.1f}%  {t_total:7.1f}s", flush=True)


def mcnemar(a, b):
    """Exact two-sided McNemar p-value for paired correct/incorrect outcomes."""
    from math import comb
    n01 = sum(x and not y for x, y in zip(a, b)); n10 = sum(y and not x for x, y in zip(a, b))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def summary(path):
    rows = [json.loads(line) for line in open(path)]
    by = defaultdict(dict)
    for r in rows:
        by[(r["survey"], r["method"])][(r["shard"], r["index"])] = r
    surveys = list(dict.fromkeys(r["survey"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    for survey in surveys:
        for subset in ("all", "strict") if survey == "ASAS-SN" else ("all",):
            print(f"\n== {survey}{' (strictly periodic classes)' if subset == 'strict' else ''}")
            ref = by.get((survey, "pdm"), {})
            for method in methods:
                rs = by.get((survey, method))
                if not rs:
                    continue
                keys = [k for k, r in rs.items() if subset == "all" or r["cls"] in STRICT]
                variants = [("", "v")] + ([(" +tie", "v_tie")] if "v_tie" in next(iter(rs.values())) else [])
                for label, vk in variants:
                    v = [rs[k][vk] for k in keys]; n = len(v)
                    c, h = v.count("correct") / n, v.count("harmonic") / n
                    extra = ""
                    if "recall" in rs[keys[0]] and not label:
                        extra = (f"  recall {100 * np.mean([rs[k]['recall'] for k in keys]):5.1f}%"
                                 f" (incl. 2P, P/2 {100 * np.mean([rs[k]['recall_harm'] for k in keys]):5.1f}%)")
                    if ref and method != "pdm":
                        common = [k for k in keys if k in ref]
                        p = mcnemar([ref[k]["v"] == "correct" for k in common], [rs[k][vk] == "correct" for k in common])
                        extra += f"  vs PDM p={p:.2g}"
                    t = np.mean([rs[k]["time_s"] for k in keys])
                    print(f"  {method + label:<60} n={n:<5} correct {100 * c:5.1f}%  harmonic {100 * h:5.1f}%  "
                          f"{1000 * t:7.0f} ms/star{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--out")
    ap.add_argument("--summary")
    ap.add_argument("--methods", nargs="+", default=["ls", "pdm"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--fmax", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--device")
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
    else:
        run(a)


if __name__ == "__main__":
    main()
