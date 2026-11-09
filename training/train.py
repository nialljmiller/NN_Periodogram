#!/usr/bin/env python3
"""Train PeriodsNet on the shards from build_dataset.py (one consumer GPU is plenty).

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

    python train.py --data DATA --out runs/v1
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
from nn_periodogram.features import global_features, profile_views, robust_normalise  # noqa: E402
from nn_periodogram.models import PeriodsNet  # noqa: E402

F_MAX = 20.0       # highest trial frequency, 1/day
VIEWS = (1.0,)     # set from --views: folds at f * r for each r


def mapped(path):
    """Arrays of {path}.npz, memory-mapped. The (uncompressed) npz is unpacked once into
    {path}.npyd/ as plain .npy files; only the pages actually read are held in RAM."""
    import zipfile
    d = Path(f"{path}.npyd")
    if not (d / "done").exists():
        d.mkdir(exist_ok=True)
        with zipfile.ZipFile(f"{path}.npz") as z:
            z.extractall(d)
        (d / "done").touch()
    return {f.stem: np.load(f, mmap_mode="r") for f in d.glob("*.npy")}


class Shard:
    def __init__(self, path, split):
        # VVV labels came from LS/PDM agreement, so its harmonics are not trustworthy negatives
        self.trusted = not Path(path).name.startswith("vvv_real")
        d = mapped(path)                  # memory-mapped: 20+ GB of shards in a few GB of RAM
        meta = json.load(open(f"{path}_meta.json"))
        periods = {i: r.get("period", np.nan) for i, r in enumerate(meta)}
        lpath, vpath = Path(f"{path}_labels.json"), Path(f"{path}_verified.json")
        if lpath.exists():                 # relabel.py: catalogue period (refined) for verified stars
            periods = {v["index"]: 1 / v["freq"] for v in json.load(open(lpath))}
        elif vpath.exists():               # older runs: verified frequency (may be a harmonic!)
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
        eb_classes = {"E", "EA", "EB", "EW", "ELL", "ECL", "ESD", "ED", "EC", "KEPLER_EB", "TESS_EB"}
        is_eb_shard = any(k in Path(path).name for k in ("_ecl", "_eb", "_dpv"))
        self.binary = np.array([is_eb_shard or str(meta[i].get("cls", "")).upper().rstrip(":") in eb_classes
                                for i in keep], bool)

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
    if shape == "ew":                                   # contact (W UMa) / ellipsoidal binary
        x = (phase - rng.uniform()) % 1
        delta = rng.uniform(0, 0.15)                    # primary vs secondary depth difference
        oconnell = rng.uniform(-0.05, 0.05)             # unequal maxima
        y = 0.5 * (1 - np.cos(4 * np.pi * x)) + delta * 0.5 * (1 - np.cos(2 * np.pi * x)) \
            + oconnell * np.sin(2 * np.pi * x)
        return y - y.mean()
    if shape == "eb_equal":                             # detached, near-equal eclipses
        w = rng.uniform(0.02, 0.08); d2 = rng.uniform(0.75, 1.0)
        x = (phase - rng.uniform()) % 1
        dist1 = np.minimum(x, 1 - x); dist2 = np.abs(x - 0.5)
        return np.exp(-0.5 * (dist1 / w) ** 2) + d2 * np.exp(-0.5 * (dist2 / (w * rng.uniform(0.8, 1.25))) ** 2) - 0.5
    k = np.arange(1, 5)                                 # random Fourier series
    a = rng.normal(0, 1, 4) / k; p = rng.uniform(0, 2 * np.pi, 4)
    y = (a[None] * np.sin(2 * np.pi * k[None] * phase[:, None] + p[None])).sum(1)
    return y / (np.ptp(y) / 2 + 1e-9)


HARMONICS = (2.0, 0.5, 3.0, 1 / 3, 2 / 3, 1.5)


HARD_EB = 0.0      # set from --hard-eb: share of injections that are binaries with near-equal minima
EB_SHAPES = ("eb", "ew", "eb_equal")


def negative_freq(f_true, span, rng, harmonics=False, binary=False):
    """A wrong frequency for a star whose true frequency is f_true (nan for none).

    With harmonics=True (only where the truth is exact: injections and the
    high-quality catalogues), a quarter of negatives are k/n multiples of the
    true frequency, so the model learns P from 2P and P/2. Otherwise folds near
    any small-integer ratio are kept out of the negatives.
    """
    if HARD_EB and binary and harmonics and np.isfinite(f_true) and rng.uniform() < 0.6:
        # binaries: the fold at P/2 (eclipses stacked) and at 2P are the hard cases
        return f_true * (2.0 if rng.uniform() < 0.6 else 0.5) + rng.uniform(-0.05, 0.05) / span
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


def augment(t, m, e, rng):
    """Label-preserving views of the same light curve: flip, time reversal, a dropped
    season, rescaled errors. None of these changes whether (or at what period) it is periodic."""
    if rng.uniform() < 0.5:
        m = -m                                              # upside down
    if rng.uniform() < 0.25:
        t = t.max() + t.min() - t                           # time reversed
    if rng.uniform() < 0.2 and np.ptp(t) > 730:             # drop one ~year-long stretch
        start = rng.uniform(t.min(), t.max() - 365)
        keep = (t < start) | (t > start + 365)
        if keep.sum() >= 10:
            t, m, e = t[keep], m[keep], e[keep]
    if rng.uniform() < 0.3:
        e = e * math.exp(rng.uniform(math.log(0.5), math.log(2)))
    if QUASI and rng.uniform() < 0.15 and np.ptp(t) > 400:              # season-to-season zero points
        season = np.floor((t - t.min()) / 365.25).astype(int)
        sig = 1.4826 * np.median(np.abs(m - np.median(m))) or float(np.median(e))
        m = m + rng.normal(0, 0.3 * sig, season.max() + 1)[season]
    return t, m, e


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


QUASI = 0.0        # set from --quasi: fraction of injections that are quasi-periodic


def inject(t, m, rng):
    """Add a known periodic signal; returns new mags and its frequency.

    With probability QUASI the signal is quasi-periodic, like spotted rotators,
    semi-regulars and long-period variables: the phase wanders (random walk with a
    coherence time of 5-100 cycles), the amplitude is modulated over seasons, and
    the period may drift by up to 1%. The label stays the mean frequency."""
    span = np.ptp(t)
    f_true = math.exp(rng.uniform(math.log(3 / span), math.log(F_MAX)))
    sig = 1.4826 * np.median(np.abs(m - np.median(m))) or 0.01
    amp = sig * math.exp(rng.uniform(math.log(0.5), math.log(20)))
    phase = t * f_true
    envelope = 1.0
    if rng.uniform() < QUASI:
        order = np.argsort(t); ts = t[order]
        n_coh = math.exp(rng.uniform(math.log(5), math.log(100)))          # coherence time in cycles
        dphi = np.diff(ts, prepend=ts[0]) * f_true / n_coh                 # cycles elapsed / coherence
        walk = np.cumsum(rng.normal(0, 1, len(ts)) * np.sqrt(np.maximum(dphi, 0))) * 0.25
        drift = rng.uniform(-0.01, 0.01) * f_true * (ts - ts.mean()) ** 2 / (2 * span)
        wander = np.empty_like(walk); wander[order] = walk + drift
        phase = phase + wander
        t_mod = math.exp(rng.uniform(math.log(span / 10), math.log(2 * span)))
        envelope = 1 + rng.uniform(0, 0.6) * np.sin(2 * np.pi * t / t_mod + rng.uniform(0, 2 * np.pi))
    if np.ndim(envelope):
        shape = rng.choice(["sine", "fourier"])
    elif HARD_EB and rng.uniform() < HARD_EB:
        shape = rng.choice(["ew", "eb_equal"])
    else:
        shape = rng.choice(["sine", "rr", "eb", "fourier"])
    INJECTED[0] = shape
    return m + amp * envelope * signal(shape, phase % 1, rng), f_true


INJECTED = [None]   # shape of the last injection (read by batch_arrays)


def batch_arrays(reals, quiet, B, rng):
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
            exact = real.trusted; binary = bool(real.binary[i])
        elif src < 0.65:
            i = rng.integers(len(quiet)); t, e = quiet.t[i], quiet.e[i]
            m, f_true = inject(t, quiet.m[i].astype(float), rng); exact = True; binary = INJECTED[0] in EB_SHAPES
        elif src < 0.85:
            real = reals[rng.integers(len(reals))]; i = rng.integers(len(real))
            t, e = real.t[i], np.asarray(real.e[i], float)
            m, f_true = inject(t, synthetic_noise(t, e, rng), rng); exact = True; binary = INJECTED[0] in EB_SHAPES
        else:
            positive, f_true, exact, binary = False, np.nan, False, False
            u = rng.uniform()
            if u < 0.35:                                    # real light curve, magnitudes shuffled in time:
                real = reals[rng.integers(len(reals))]; i = rng.integers(len(real))
                t, m, e = real.t[i], real.m[i].copy(), real.e[i].copy()
                k = rng.permutation(len(m)); m, e = m[k], e[k]          # same cadence and noise, no signal
            elif u < 0.65:
                i = rng.integers(len(quiet)); t, m, e = quiet.t[i], quiet.m[i], quiet.e[i]
            else:
                real = reals[rng.integers(len(reals))]; i = rng.integers(len(real))
                t, e = real.t[i], np.asarray(real.e[i], float); m = synthetic_noise(t, e, rng)
        span = np.ptp(t)
        f = f_true + rng.uniform(-0.05, 0.05) / span if positive else negative_freq(f_true, span, rng, exact, binary)
        t, m, e = thin(np.asarray(t), np.asarray(m, float), np.asarray(e, float), rng)
        t, m, e = augment(t, m, e, rng)
        times.append(t); mags.append(m); errs.append(e); freqs.append(f); labels.append(float(positive))
    T, w = pad(times); M, _ = pad(mags); E, _ = pad(errs)
    return T, w, M, E, np.array(freqs), np.array(labels, np.float32)


def features(arrays, device):
    """Padded arrays from batch_arrays -> network inputs on `device`."""
    T, w, M, E, freqs, labels = (np.asarray(a) for a in arrays)
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    z, ze = robust_normalise(torch.tensor(M, dtype=torch.float32, device=device),
                             torch.tensor(np.where(w > 0, E, 1.0), dtype=torch.float32, device=device), w_t)
    x = profile_views(torch.tensor(T, dtype=torch.float64), torch.tensor(freqs, dtype=torch.float64),
                      z, ze, w_t, VIEWS, device)
    return x, global_features(z, ze, w_t), torch.tensor(labels, device=device)


def batch(reals, quiet, B, rng, device):
    return features(batch_arrays(reals, quiet, B, rng), device)


class Batches(torch.utils.data.IterableDataset):
    """Endless training batches built in worker processes (the light-curve work is NumPy on
    the CPU; with several workers the GPU no longer waits for it)."""

    def __init__(self, reals, quiet, B, seed):
        self.reals, self.quiet, self.B, self.seed = reals, quiet, B, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            yield batch_arrays(self.reals, self.quiet, self.B, rng)


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
    ap.add_argument("--views", type=float, nargs="+", default=[1.0],
                    help="fold views per candidate; 1 0.5 2 = P with 2P and P/2 side by side")
    ap.add_argument("--steps", type=int, default=60000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6, help="processes building batches")
    ap.add_argument("--quasi", type=float, default=0.0, help="fraction of injections that are quasi-periodic")
    ap.add_argument("--width", type=int, default=96); ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--hard-eb", type=float, default=0.0, help="share of injections that are near-equal-minima binaries")
    ap.add_argument("--init", help="start from these weights (fine-tuning)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    if a.real == ["auto"]:
        # every verified shard; benchmarks use only the val/test splits, which training never sees
        a.real = sorted(p.name[:-len("_verified.json")] for p in Path(a.data).glob("*_verified.json"))
    tr_real = [Shard(f"{a.data}/{r}", "train") for r in a.real]
    va_real = [Shard(f"{a.data}/{r}", "val") for r in a.real]
    tr_quiet, va_quiet = Shard(f"{a.data}/{a.quiet}", "train"), Shard(f"{a.data}/{a.quiet}", "val")
    print("train real:", {r: len(s) for r, s in zip(a.real, tr_real)}, "quiet", len(tr_quiet),
          "| val real:", {r: len(s) for r, s in zip(a.real, va_real)}, "quiet", len(va_quiet), flush=True)

    global VIEWS, QUASI, HARD_EB
    VIEWS = tuple(a.views)
    QUASI = a.quasi
    HARD_EB = a.hard_eb
    json.dump({"views": VIEWS, "width": a.width, "blocks": a.blocks, "quasi": a.quasi, "hard_eb": a.hard_eb,
               "init": a.init}, open(out / "config.json", "w"))
    model = PeriodsNet(views=VIEWS, width=a.width, dilations=((1, 2, 4, 8) * 4)[:a.blocks]).to(device)
    if a.init:
        state = torch.load(a.init, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("state_dict", state))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    vrng = np.random.default_rng(999)
    val = [batch(va_real, va_quiet, a.batch, vrng, device) for _ in range(20)]   # fixed validation set
    guard, best, log = ThermalGuard(device), 1e9, []
    loader = iter(torch.utils.data.DataLoader(Batches(tr_real, tr_quiet, a.batch, a.seed), batch_size=None,
                                              num_workers=a.workers, prefetch_factor=4 if a.workers else None,
                                              persistent_workers=a.workers > 0))
    t0 = time.time()
    for step in range(1, a.steps + 1):
        guard.check()
        model.train()
        x, g, y = features(next(loader), device)
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
