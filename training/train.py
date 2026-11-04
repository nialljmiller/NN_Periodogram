#!/usr/bin/env python3
"""Train PeriodsNet on the shards from build_dataset.py (run on cirrus's GPU).

Each batch is built on the fly, half positives and half negatives:
  * real stars folded at their agreed period (with jitter well inside 1/T);
  * signals injected into quiet stars: sines, RR Lyrae-like sawtooths,
    eclipses and random Fourier shapes, with amplitude/noise from 0.5 to 20,
    folded at the injected period;
  * negatives: quiet stars at random periods; real and injected stars at
    random periods, nearby incoherent periods (3-30/T away) and daily
    aliases (f +- 1, 2 per day).
Folds within 0.5/T of k/n times the true frequency (k, n <= 3) are left out
of the negatives, since they are partly coherent; periodogram() deals with them.

    python train.py --data /media/data4/nn_periodogram_data --out runs/v1
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nn_periodogram.features import binned_profile, robust_normalise  # noqa: E402
from nn_periodogram.models import PeriodsNet  # noqa: E402

L = 512            # max points per light curve in a batch (random subset beyond that)
F_MAX = 20.0       # highest trial frequency, 1/day


class Shard:
    def __init__(self, path, split):
        d = np.load(f"{path}.npz")
        meta = json.load(open(f"{path}_meta.json"))
        periods = {i: r.get("period", np.nan) for i, r in enumerate(meta)}
        vpath = Path(f"{path}_verified.json")
        if vpath.exists():                 # real stars: keep only verified ones, label = our LS peak
            periods = {v["index"]: 1 / v["freq"] for v in json.load(open(vpath)) if v["verified"]}
        keep = [i for i, r in enumerate(meta) if r["split"] == split and i in periods]
        off = d["offsets"]
        self.t, self.m, self.e = [], [], []
        self.period = []
        for i in keep:
            a, b = off[i], off[i + 1]
            self.t.append(d["time"][a:b]); self.m.append(d["mag"][a:b]); self.e.append(d["err"][a:b])
            self.period.append(periods[i])
        self.period = np.array(self.period)

    def __len__(self):
        return len(self.t)


def pad(rows, rng):
    """List of 1-D arrays -> (B, L) array + mask, random subset if longer than L."""
    out = np.zeros((len(rows), L)); w = np.zeros((len(rows), L))
    for k, r in enumerate(rows):
        if len(r) > L:
            r = r[np.sort(rng.choice(len(r), L, replace=False))]
        out[k, :len(r)] = r; w[k, :len(r)] = 1
    return out, w


def signal(shape, phase, rng):
    """Unit-amplitude light-curve shape at the given phases (magnitudes, brighter = negative)."""
    if shape == "sine":
        return -np.sin(2 * np.pi * phase)
    if shape == "rr":                                   # fast rise, slow decline
        x = (phase - rng.uniform()) % 1
        return np.where(x < 0.15, 1 - 2 * x / 0.15, -1 + 2 * (x - 0.15) / 0.85)
    if shape == "eb":
        w1, w2 = rng.uniform(0.02, 0.1), rng.uniform(0.02, 0.1)
        d2 = rng.uniform(0, 1)
        x = (phase - rng.uniform()) % 1
        dist1 = np.minimum(x, 1 - x); dist2 = np.abs(x - 0.5)
        return np.exp(-0.5 * (dist1 / w1) ** 2) + d2 * np.exp(-0.5 * (dist2 / w2) ** 2) - 0.5
    k = np.arange(1, 5)                                 # random Fourier series
    a = rng.normal(0, 1, 4) / k; p = rng.uniform(0, 2 * np.pi, 4)
    y = (a[None] * np.sin(2 * np.pi * k[None] * phase[:, None] + p[None])).sum(1)
    return y / (np.ptp(y) / 2 + 1e-9)


def negative_freq(f_true, span, rng):
    """A wrong frequency for a star whose true frequency is f_true (nan for none)."""
    for _ in range(100):
        kind = rng.uniform()
        if np.isfinite(f_true) and kind < 0.3:
            f = f_true + rng.choice([-1, 1]) * rng.uniform(3, 30) / span      # nearby, incoherent
        elif np.isfinite(f_true) and kind < 0.45:
            f = f_true + rng.choice([-2, -1, 1, 2])                          # daily alias
        else:
            f = math.exp(rng.uniform(math.log(2 / span), math.log(F_MAX)))
        if f <= 0:
            continue
        if np.isfinite(f_true):
            ratios = [k / n for k in (1, 2, 3) for n in (1, 2, 3)]
            if any(abs(f - r * f_true) < 0.5 / span for r in ratios):
                continue
        return f
    return 1.234


def batch(real, quiet, B, rng, device):
    times, mags, freqs, labels = [], [], [], []
    for _ in range(B):
        positive = rng.uniform() < 0.5
        src = rng.uniform()
        if src < 0.4:                                   # real variable
            i = rng.integers(len(real)); t, m = real.t[i], real.m[i]; f_true = 1 / real.period[i]
        elif src < 0.8:                                 # injection into a quiet star
            i = rng.integers(len(quiet)); t, m = quiet.t[i], quiet.m[i].astype(float)
            span = np.ptp(t)
            f_true = math.exp(rng.uniform(math.log(3 / span), math.log(F_MAX)))
            sig = np.std(m) if np.std(m) > 0 else 0.01
            amp = sig * math.exp(rng.uniform(math.log(0.5), math.log(20)))
            m = m + amp * signal(rng.choice(["sine", "rr", "eb", "fourier"]), (t * f_true) % 1, rng)
        else:                                           # quiet star: negatives only
            i = rng.integers(len(quiet)); t, m = quiet.t[i], quiet.m[i]; f_true = np.nan
            positive = False
        span = np.ptp(t)
        f = f_true + rng.uniform(-0.05, 0.05) / span if positive else negative_freq(f_true, span, rng)
        times.append(t); mags.append(np.asarray(m, float)); freqs.append(f); labels.append(float(positive))
    T, w = pad(times, rng); M, _ = pad(mags, rng)
    phase = np.mod(T * np.array(freqs)[:, None], 1.0)           # float64 on the CPU
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    z = robust_normalise(torch.tensor(M, dtype=torch.float32, device=device), w_t)
    x = binned_profile(torch.tensor(phase, dtype=torch.float32, device=device), z, w_t)
    return x, torch.tensor(labels, device=device)


class ThermalGuard:
    """Pause at 80 C, resume at 70 C; same rule as the weather-station training."""

    def __init__(self, device):
        self.device, self.next = device, 0.0

    def check(self):
        if self.device.type != "cuda" or time.time() < self.next:
            return
        try:
            temp = torch.cuda.temperature(self.device)
        except Exception:
            return
        if temp >= 80:
            print(f"GPU at {temp} C, pausing", flush=True)
            while torch.cuda.temperature(self.device) > 70:
                time.sleep(10)
        self.next = time.time() + 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tr_real, tr_quiet = Shard(f"{a.data}/real", "train"), Shard(f"{a.data}/quiet", "train")
    va_real, va_quiet = Shard(f"{a.data}/real", "val"), Shard(f"{a.data}/quiet", "val")
    print(f"train: {len(tr_real)} real, {len(tr_quiet)} quiet; val: {len(va_real)} real, {len(va_quiet)} quiet", flush=True)

    model = PeriodsNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    vrng = np.random.default_rng(999)
    val = [batch(va_real, va_quiet, a.batch, vrng, device) for _ in range(20)]   # fixed validation set
    guard, best, log = ThermalGuard(device), 1e9, []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        guard.check()
        model.train()
        x, y = batch(tr_real, tr_quiet, a.batch, rng, device)
        loss = loss_fn(model.logits(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                vl = np.mean([loss_fn(model.logits(vx), vy).item() for vx, vy in val])
                acc = np.mean([((model(vx) > 0.5).float() == vy).float().mean().item() for vx, vy in val])
            log.append({"step": step, "train_loss": loss.item(), "val_loss": vl, "val_acc": acc})
            print(f"step {step}  train {loss.item():.4f}  val {vl:.4f}  acc {acc:.4f}  {time.time() - t0:.0f}s", flush=True)
            if vl < best:
                best = vl; torch.save(model.state_dict(), out / "best.pt")
            json.dump(log, open(out / "log.json", "w"))


if __name__ == "__main__":
    main()
