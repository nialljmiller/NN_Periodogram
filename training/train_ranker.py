#!/usr/bin/env python3
"""Train the candidate ranker on shortlists from ranker_data.py.

Target: the shortlisted candidate within 1% of the catalogue period; when the truth is not
on the shortlist, the first-pass winner (index 0), so the ranker only moves away from the
first pass when it has reason to. Reports, per survey on the validation stars, how often the
pick is right for the first pass and for the ranker; keeps the best epoch.

    python train_ranker.py --train train.npz --val val.npz --out ranker.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nn_periodogram.models import CandidateRanker  # noqa: E402


def load(path, device):
    d = np.load(path)
    t = {k: torch.tensor(d[k].astype(np.float32), device=device) for k in ("emb", "scalar", "rel")}
    t["mask"] = torch.tensor(d["mask"], device=device)
    t["label"] = torch.tensor(d["label"], device=device)
    right = np.abs(d["freqs"] / d["f_true"][:, None] - 1) < 0.01                  # (N, K): which picks are right
    t["right"] = torch.tensor(right & d["mask"], device=device)
    t["source"] = d["source"]
    return t


def picks(model, d, bs=4096):
    out = []
    with torch.no_grad():
        for s in range(0, len(d["mask"]), bs):
            sc = model(d["emb"][s:s + bs], d["scalar"][s:s + bs], d["rel"][s:s + bs], d["mask"][s:s + bs])
            out.append(sc.argmax(1))
    return torch.cat(out)


def report(d, pick, label=""):
    right = d["right"][torch.arange(len(pick)), pick].cpu().numpy()
    base = d["right"][:, 0].cpu().numpy()
    src = np.array([str(s) for s in d["source"]])
    rows = []
    for s in sorted(set(src)):
        k = src == s
        rows.append(f"{s}: {100 * base[k].mean():.1f} -> {100 * right[k].mean():.1f}")
    return right.mean(), base.mean(), "  ".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = load(a.train, device), load(a.val, device)
    model = CandidateRanker(n_emb=tr["emb"].shape[2], n_scalar=tr["scalar"].shape[2]).to(device)
    m = tr["mask"][..., None].float()
    mean = (tr["scalar"] * m).sum((0, 1)) / m.sum((0, 1))
    std = (((tr["scalar"] - mean) ** 2 * m).sum((0, 1)) / m.sum((0, 1))).sqrt().clamp_min(1e-3)
    model.scalar_mean.copy_(mean); model.scalar_std.copy_(std)
    target = torch.where(tr["label"] >= 0, tr["label"], torch.zeros_like(tr["label"]))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n = len(target); steps = a.epochs * ((n + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    best = -1
    base_val = va["right"][:, 0].float().mean().item()
    print(f"validation: first pass right {100 * base_val:.2f}% of {len(va['mask'])} stars; truth on shortlist "
          f"{100 * va['right'].any(1).float().mean().item():.2f}%", flush=True)
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, a.batch):
            i = perm[s:s + a.batch]
            sc = model(tr["emb"][i], tr["scalar"][i], tr["rel"][i], tr["mask"][i])
            loss = torch.nn.functional.cross_entropy(sc, target[i])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        r, b, rows = report(va, picks(model, va))
        rt, bt, _ = report(tr, picks(model, tr))
        print(f"epoch {ep + 1:2d}  train {100 * bt:.2f}->{100 * rt:.2f}%  val {100 * b:.2f}->{100 * r:.2f}%   {rows}", flush=True)
        if r > best:
            best = r; torch.save(model.state_dict(), a.out)
    print(f"best validation: {100 * best:.2f}% right (first pass {100 * base_val:.2f}%) -> {a.out}")


if __name__ == "__main__":
    main()
