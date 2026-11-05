"""The one API: NN false-alarm probabilities and NN periodograms."""

from dataclasses import dataclass, field

import numpy as np
import torch

from . import hub
from .features import (binned_profile, channels_from_phase, fold_phase, global_features, prepare,
                       robust_normalise)

_MODELS = {}


def _model(name, device, path=None):
    key = (name, str(device), path)
    if key not in _MODELS:
        _MODELS[key] = hub.load(name, path=path, device=device)
    return _MODELS[key]


@torch.no_grad()
def _score_legacy(t, m, freqs, model, device, batch, exact=False):
    # exact: features in float64 on the CPU, matching the 2022 NumPy code to ~1e-6.
    # Otherwise float32 on `device`: about 2x faster, and a few window-edge ties in the
    # running-scatter channel can round the other way (largest seen: 0.02 in FAP).
    t, m = prepare(t, m)
    fdev, fdt = (torch.device("cpu"), torch.float64) if exact else (device, torch.float32)
    tt = torch.tensor(t, dtype=torch.float64)
    mm = torch.tensor(m, dtype=fdt, device=fdev)
    out = np.empty(len(freqs), np.float32)
    for s in range(0, len(freqs), batch):
        f = torch.tensor(freqs[s:s + batch], dtype=torch.float64)
        # Phase in float64 on the CPU (time * freq over long baselines needs it).
        ph = fold_phase(tt, f).to(device=fdev, dtype=fdt)
        x = channels_from_phase(ph, mm).to(device=device, dtype=torch.float32)
        out[s:s + len(f)] = model(x).float().cpu().numpy()
    return out


def estimate_errors(t, m):
    """Per-point error when none is given: scatter between time-neighbouring points."""
    d = np.diff(m[np.argsort(t)])
    return np.full(len(m), max(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2), 1e-6))


@torch.no_grad()
def _score_binned(t, m, e, freqs, model, device, max_elements=2 ** 24):
    """Every point is used; batches shrink as light curves grow to bound memory."""
    t, m = np.asarray(t, float), np.asarray(m, float)
    e = estimate_errors(t, m) if e is None else np.asarray(e, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    w = torch.ones(1, len(t), device=device)
    z, ze = robust_normalise(torch.tensor(m, dtype=torch.float32, device=device)[None],
                             torch.tensor(e, dtype=torch.float32, device=device)[None], w)
    g = global_features(z, ze, w)
    tt = torch.tensor(t, dtype=torch.float64)
    batch = max(1, max_elements // len(t))
    out = np.empty(len(freqs), np.float32)
    for s in range(0, len(freqs), batch):
        f = torch.tensor(freqs[s:s + batch], dtype=torch.float64)
        ph = fold_phase(tt, f).to(device=device, dtype=torch.float32)
        B = len(f)
        x = binned_profile(ph, z.expand(B, -1), ze.expand(B, -1), w.expand(B, -1))
        out[s:s + B] = 1.0 - model(x, g.expand(B, -1)).float().cpu().numpy()   # P(real) -> FAP
    return out


def _score(t, m, freqs, model, device, batch, exact=False, err=None):
    if getattr(model, "input_kind", "legacy") == "binned":
        return _score_binned(t, m, err, freqs, model, device)
    return _score_legacy(t, m, freqs, model, device, batch, exact)


def fap(time, mag, period, err=None, model=hub.DEFAULT, device=None, batch=1024, weights=None, exact=False):
    """NN false-alarm probability of `period` (scalar or array) for one light curve.

    Low values mean the fold at that period looks like a real signal.
    """
    device = torch.device(device) if device else hub.default_device()
    t, m = np.asarray(time, float), np.asarray(mag, float)
    periods = np.atleast_1d(np.asarray(period, float))
    out = _score(t, m, 1.0 / periods, _model(model, device, weights), device, batch, exact, err)
    return out if np.ndim(period) else float(out[0])


@dataclass
class Periodogram:
    freq: np.ndarray
    fap: np.ndarray
    peaks: list = field(default_factory=list)      # [(period, fap)] best first, after refinement
    best_period: float = np.nan
    best_fap: float = np.nan
    note: str = ""

    @property
    def period(self):
        return 1.0 / self.freq

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        ax.plot(self.freq, -np.log10(np.clip(self.fap, 1e-6, 1)), lw=0.8)
        ax.axvline(1 / self.best_period, color="k", ls=":", lw=1)
        ax.set_xlabel("frequency (1/day)"); ax.set_ylabel("-log10 FAP")
        return ax


def periodogram(time, mag, err=None, fmin=None, fmax=10.0, oversample=3.0, n_peaks=5, refine=20,
                model=hub.DEFAULT, device=None, batch=1024, weights=None, exact=False):
    """NN periodogram: FAP at every trial frequency, then a finer pass around the best peaks.

    The grid spacing is 1/(oversample * baseline). Folding at twice the true
    period also gives a coherent curve, and the network scores it about the
    same, so when the best peak's half-period is nearly as good, the shorter
    period is reported and `note` says so.
    """
    device = torch.device(device) if device else hub.default_device()
    t, m = np.asarray(time, float), np.asarray(mag, float)
    span = np.ptp(np.asarray(time, float)[np.isfinite(time)])
    df = 1.0 / (oversample * span)
    fmin = fmin or 2.0 / span
    freqs = np.arange(fmin, fmax, df)
    net = _model(model, device, weights)
    f_ap = _score(t, m, freqs, net, device, batch, exact, err)

    # local minima, best first, separated by at least 5 grid steps
    order = np.argsort(f_ap)
    picked = []
    for i in order:
        if all(abs(i - j) > 5 for j in picked):
            picked.append(i)
        if len(picked) == n_peaks:
            break
    peaks = []
    for i in picked:
        fine = np.linspace(freqs[i] - df, freqs[i] + df, 2 * refine + 1)
        ff = _score(t, m, fine, net, device, batch, exact, err)
        k = int(np.argmin(ff))
        peaks.append((1.0 / fine[k], float(ff[k])))
    peaks.sort(key=lambda p: p[1])

    best, best_f = peaks[0]
    note = ""
    half = [p for p in peaks if abs(p[0] - best / 2) < 0.01 * best / 2]
    if half and half[0][1] <= 1.25 * best_f:
        note = f"twice-period alias: {best:.6g} d scores about the same as {half[0][0]:.6g} d; reporting the shorter"
        best, best_f = half[0]
    return Periodogram(freqs, f_ap, peaks, best, best_f, note)
