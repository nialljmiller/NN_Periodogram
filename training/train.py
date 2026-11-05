#!/usr/bin/env python3
"""Train PeriodsNet on the shards from build_dataset.py (run on windybox's 3070 Ti).

Light curves are used whole, from 10 to 10,000+ points: batches are padded to
the longest one with a mask. Half the time a light curve is randomly thinned to
10..all points so the model learns how confidence should grow with data.

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
from nn_periodogram.features import binned_profile, global_features, robust_normalise  # noqa: E402
from nn_periodogram.models import PeriodsNet  # noqa: E402

F_MAX = 20.0       # highest trial frequency, 1/day


class Shard:
    def __init__(self, path, split):
        # VVV labels came from LS/PDM agreement, so its harmonics are not trustworthy negatives
        self.trusted = not Path(path).name.startswith("vvv_")
        d = dict(np.load(f"{path}.npz"))  # load arrays once: indexing an NpzFile re-reads the whole array
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


def pad(rows):
    """List of 1-D arrays -> (B, L) array + mask, L = the longest row. No points are dropped."""
    L = max(len(r) for r in rows)
    out = np.zeros((len(rows), L)); w = np.zeros((len(rows), L))
    for k, r in enumerate(rows):
        out[k, :len(r)] = r; w[k, :len(r)] = 1
    return out, w


MAX_TRAIN_POINTS = 5000


def thin(t, m, e, rng, min_points=10):
    """Augmentation: half the time keep a random subset of 10..all points (log-uniform),
    so the model sees each star at many data volumes. Light curves longer than
    MAX_TRAIN_POINTS (Kepler, TESS) get a fresh random subset of that size each draw,
    to bound GPU memory in a padded batch; inference always uses every point."""
    n = len(t)
    if n > MAX_TRAIN_POINTS:
        keep = np.sort(rng.choice(n, MAX_TRAIN_POINTS, replace=False))
        t, m, e, n = t[keep], m[keep], e[keep], MAX_TRAIN_POINTS
    if n <= min_points or rng.uniform() < 0.5:
        return t, m, e
    k = int(round(math.exp(rng.uniform(math.log(min_points), math.log(n)))))
    keep = np.sort(rng.choice(n, k, replace=False))
    return t[keep], m[keep], e[keep]


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


HARMONICS = (2.0, 0.5, 3.0, 1 / 3, 2 / 3, 1.5)


def negative_freq(f_true, span, rng, harmonics=False):
    """A wrong frequency for a star whose true frequency is f_true (nan for none).

    With harmonics=True (only where the truth is exact: injections and the
    high-quality catalogues), a quarter of negatives are k/n multiples of the
    true frequency, so the model learns P from 2P and P/2. Otherwise folds near
    any small-integer ratio are kept out of the negatives.
    """
    if harmonics and np.isfinite(f_true) and rng.uniform() < 0.25:
        return f_true * HARMONICS[rng.integers(len(HARMONICS))] + rng.uniform(-0.05, 0.05) / span
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


def synthetic_noise(t, e, rng):
    """Noise on a real cadence: white noise from the quoted errors, plus (sometimes) red noise,
    a slow trend and outliers -- the things that fool periodograms."""
    m = rng.normal(0, e)
    kind = rng.uniform()
    if kind < 0.25:                                     # random walk (red noise) on the time sampling
        steps = rng.normal(0, 1, len(t)) * np.sqrt(np.diff(t, prepend=t[0]) / max(np.ptp(t), 1e-9))
        m = m + np.cumsum(steps) * np.median(e) * math.exp(rng.uniform(0, math.log(10)))
    elif kind < 0.4:                                    # slow polynomial trend
        x = (t - t.mean()) / max(np.ptp(t), 1e-9)
        m = m + np.polyval(rng.normal(0, 1, 3), x) * np.median(e) * rng.uniform(1, 10)
    if rng.uniform() < 0.2:                             # outliers
        k = rng.random(len(t)) < rng.uniform(0.01, 0.05)
        m = m + k * rng.normal(0, 10 * np.median(e), len(t))
    return m


def inject(t, m, rng):
    """Add a known periodic signal; returns new mags and its frequency."""
    span = np.ptp(t)
    f_true = math.exp(rng.uniform(math.log(3 / span), math.log(F_MAX)))
    sig = 1.4826 * np.median(np.abs(m - np.median(m))) or 0.01
    amp = sig * math.exp(rng.uniform(math.log(0.5), math.log(20)))
    return m + amp * signal(rng.choice(["sine", "rr", "eb", "fourier"]), (t * f_true) % 1, rng), f_true


def batch(reals, quiet, B, rng, device):
    """Mix per example (positives ~half):
      35%  real variable from any survey, at its verified period or a wrong one
      30%  signal injected into a quiet VVV star (real noise and systematics)
      20%  signal injected into synthetic noise on the cadence/errors of a random star
           from any survey (Kepler, Gaia, OGLE, ... sampling)
      15%  pure noise (quiet star or synthetic red noise / trend / outliers): negatives only
    """
    times, mags, errs, freqs, labels = [], [], [], [], []
    for _ in range(B):
        positive = rng.uniform() < 0.5
        src = rng.uniform()
        if src < 0.35:
            real = reals[rng.integers(len(reals))]
            i = rng.integers(len(real)); t, m, e = real.t[i], real.m[i], real.e[i]; f_true = 1 / real.period[i]
            exact = real.trusted
        elif src < 0.65:
            i = rng.integers(len(quiet)); t, e = quiet.t[i], quiet.e[i]
            m, f_true = inject(t, quiet.m[i].astype(float), rng); exact = True
        elif src < 0.85:
            real = reals[rng.integers(len(reals))]; i = rng.integers(len(real))
            t, e = real.t[i], np.asarray(real.e[i], float)
            m, f_true = inject(t, synthetic_noise(t, e, rng), rng); exact = True
        else:
            positive, f_true, exact = False, np.nan, False
            if rng.uniform() < 0.5:
                i = rng.integers(len(quiet)); t, m, e = quiet.t[i], quiet.m[i], quiet.e[i]
            else:
                real = reals[rng.integers(len(reals))]; i = rng.integers(len(real))
                t, e = real.t[i], np.asarray(real.e[i], float); m = synthetic_noise(t, e, rng)
        span = np.ptp(t)
        f = f_true + rng.uniform(-0.05, 0.05) / span if positive else negative_freq(f_true, span, rng, exact)
        t, m, e = thin(np.asarray(t), np.asarray(m, float), np.asarray(e, float), rng)
        times.append(t); mags.append(m); errs.append(e); freqs.append(f); labels.append(float(positive))
    T, w = pad(times); M, _ = pad(mags); E, _ = pad(errs)
    phase = np.mod(T * np.array(freqs)[:, None], 1.0)           # float64 on the CPU
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    z, ze = robust_normalise(torch.tensor(M, dtype=torch.float32, device=device),
                             torch.tensor(np.where(w > 0, E, 1.0), dtype=torch.float32, device=device), w_t)
    x = binned_profile(torch.tensor(phase, dtype=torch.float32, device=device), z, ze, w_t)
    return x, global_features(z, ze, w_t), torch.tensor(labels, device=device)


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
    ap.add_argument("--real", nargs="+", default=["auto"],
                    help="shards of verified real variables; 'auto' = every shard with a *_verified.json")
    ap.add_argument("--quiet", default="vvv_quiet")
    ap.add_argument("--steps", type=int, default=60000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    if a.real == ["auto"]:
        a.real = sorted(p.name[:-len("_verified.json")] for p in Path(a.data).glob("*_verified.json")
                        if "asassn" not in p.name)               # the ASAS-SN-matched VVV stars are the benchmark
    tr_real = [Shard(f"{a.data}/{r}", "train") for r in a.real]
    va_real = [Shard(f"{a.data}/{r}", "val") for r in a.real]
    tr_quiet, va_quiet = Shard(f"{a.data}/{a.quiet}", "train"), Shard(f"{a.data}/{a.quiet}", "val")
    print("train real:", {r: len(s) for r, s in zip(a.real, tr_real)}, "quiet", len(tr_quiet),
          "| val real:", {r: len(s) for r, s in zip(a.real, va_real)}, "quiet", len(va_quiet), flush=True)

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
        x, g, y = batch(tr_real, tr_quiet, a.batch, rng, device)
        loss = loss_fn(model.logits(x, g), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                vl = np.mean([loss_fn(model.logits(vx, vg), vy).item() for vx, vg, vy in val])
                acc = np.mean([((model(vx, vg) > 0.5).float() == vy).float().mean().item() for vx, vg, vy in val])
            log.append({"step": step, "train_loss": loss.item(), "val_loss": vl, "val_acc": acc})
            print(f"step {step}  train {loss.item():.4f}  val {vl:.4f}  acc {acc:.4f}  {time.time() - t0:.0f}s", flush=True)
            if vl < best:
                best = vl; torch.save(model.state_dict(), out / "best.pt")
            json.dump(log, open(out / "log.json", "w"))


if __name__ == "__main__":
    main()
