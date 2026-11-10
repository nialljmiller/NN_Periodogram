#!/usr/bin/env python3
"""Training / validation data for the candidate ranker.

For each star: the package's own candidate stage (_hybrid_candidates) and shortlist
features (shortlist_features), exactly as periodogram() computes them, plus the index
of the shortlisted candidate within 1% of the catalogue period (-1 if none).

    python ranker_data.py --data SHARDS --split train --per-shard 3000 --out ranker_train.npz
    python ranker_data.py --sample bench_val_sample.npz --out ranker_val.npz
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
P = importlib.import_module("nn_periodogram.periodogram")
from nn_periodogram import hub  # noqa: E402
from train import mapped  # noqa: E402

K = 10


def stars_from_shards(data, split, per_shard, seed):
    rng = np.random.default_rng(seed)
    for lpath in sorted(Path(data).glob("*_labels.json")):
        name = lpath.name[:-len("_labels.json")]
        if name == "vvv_real":                         # LS-derived labels: P/2 for contact binaries
            continue
        meta = json.load(open(Path(data) / f"{name}_meta.json"))
        labels = [r for r in json.load(open(lpath)) if meta[r["index"]]["split"] == split]
        rng.shuffle(labels)
        d = mapped(Path(data) / name); off = d["offsets"]
        for r in labels[:per_shard]:
            i = r["index"]
            yield name, (np.asarray(d["time"][off[i]:off[i + 1]], float), np.asarray(d["mag"][off[i]:off[i + 1]], float),
                         np.asarray(d["err"][off[i]:off[i + 1]], float)), r["freq"]


def stars_from_sample(path):
    d = dict(np.load(path)); meta = json.loads(str(d["meta"])); off = d["offsets"]
    for i, r in enumerate(meta):
        yield r["survey"], tuple(d[k][off[i]:off[i + 1]].astype(float) for k in ("time", "mag", "err")), 1 / r["p_true"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--split", default="train"); ap.add_argument("--per-shard", type=int, default=3000)
    ap.add_argument("--sample"); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights"); ap.add_argument("--device", default="cuda"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    device = torch.device(a.device)
    net = P._model(hub.DEFAULT, device, a.weights)
    source = stars_from_sample(a.sample) if a.sample else stars_from_shards(a.data, a.split, a.per_shard, a.seed)
    EMB, SC, REL, MASK, LAB, SRC, FT, FR = [], [], [], [], [], [], [], []
    t0, n = time.time(), 0
    for name, (t, m, e), f_true in source:
        ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
        t, m, e = t[ok], m[ok], e[ok]
        if len(t) < 10 or f_true > 10 or 1 / f_true > np.ptp(t) / 3:
            continue
        with torch.no_grad():
            c = P._hybrid_candidates(t, m, e, None, 10.0, 3.0, 40, net, device, 1024)
            freqs = np.array([1 / p for p, _ in c["peaks"]]); faps = np.array([q for _, q in c["peaks"]])
            f = P.shortlist_features(c["t"], c["m"], c["e"], freqs, faps, c["powers"], c["fmin"], c["df"], net, device)
        k = len(freqs)
        emb = np.zeros((K, f["emb"].shape[1]), np.float16); emb[:k] = f["emb"]
        sc = np.zeros((K, f["scalar"].shape[1]), np.float32); sc[:k] = f["scalar"]
        rel = np.zeros((K, K, f["rel"].shape[2]), np.uint8); rel[:k, :k] = f["rel"]
        mask = np.zeros(K, bool); mask[:k] = True
        hit = np.flatnonzero(np.abs(freqs / f_true - 1) < 0.01)
        fr = np.zeros(K); fr[:k] = freqs
        EMB.append(emb); SC.append(sc); REL.append(rel); MASK.append(mask); LAB.append(int(hit[0]) if len(hit) else -1)
        SRC.append(name); FT.append(f_true); FR.append(fr)
        n += 1
        if a.limit and n >= a.limit:
            break
        if n % 1000 == 0:
            lab = np.array(LAB)
            print(f"{n} stars, {time.time() - t0:.0f}s; truth shortlisted {np.mean(lab >= 0):.1%}, "
                  f"first-pass right {np.mean(lab == 0):.1%}", flush=True)
    np.savez_compressed(a.out, emb=np.array(EMB), scalar=np.array(SC), rel=np.array(REL), mask=np.array(MASK),
                        label=np.array(LAB), source=np.array(SRC), f_true=np.array(FT), freqs=np.array(FR))
    lab = np.array(LAB)
    print(f"done: {n} stars -> {a.out}; truth shortlisted {np.mean(lab >= 0):.1%}, first-pass right {np.mean(lab == 0):.1%}")


if __name__ == "__main__":
    main()
