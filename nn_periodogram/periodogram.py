"""The one API: NN false-alarm probabilities and NN periodograms."""

from dataclasses import dataclass, field

import numpy as np
import torch

from . import hub
from .classical import gls, pdm_theta
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
    batch = max(1, min(4096, max_elements // len(t)))     # bound both the point arrays and the network activations
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
                prefer_shorter=None, method=None, n_candidates=10,
                model=hub.DEFAULT, device=None, batch=1024, weights=None, exact=False):
    """NN periodogram: FAP at every trial frequency, then a finer pass around the best peaks.

    The grid spacing is 1/(oversample * baseline). Folding at twice the true
    period also gives a coherent curve, and the network scores it about the
    same, so when the best peak's half-period is nearly as good, the shorter
    period is reported and `note` says so.
    """
    device = torch.device(device) if device else hub.default_device()
    net = _model(model, device, weights)
    if method is None:          # the 2022 network only works as an exhaustive search
        method = "full" if getattr(net, "input_kind", "legacy") == "legacy" else "hybrid"
    if method == "hybrid":
        return _hybrid(time, mag, err, fmin, fmax, oversample, n_candidates, net, device, batch, prefer_shorter)
    t, m = np.asarray(time, float), np.asarray(mag, float)
    span = np.ptp(np.asarray(time, float)[np.isfinite(time)])
    df = 1.0 / (oversample * span)
    fmin = fmin or 2.0 / span
    freqs = np.arange(fmin, fmax, df)
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
    # A model that cannot tell P from 2P scores them alike; then the shorter of a tied pair is
    # reported (on the benchmark this helps both the 2022 model and periods-v1).
    if prefer_shorter is None:
        prefer_shorter = True
    half = [p for p in peaks if abs(p[0] - best / 2) < 0.01 * best / 2]
    if prefer_shorter and half and half[0][1] <= 1.25 * best_f:
        note = f"twice-period alias: {best:.6g} d scores about the same as {half[0][0]:.6g} d; reporting the shorter"
        best, best_f = half[0]
    return Periodogram(freqs, f_ap, peaks, best, best_f, note)


def _top(score, n, sep=5):
    """Indices of the n highest values, at least `sep` grid steps apart."""
    picked = []
    for i in np.argsort(score)[::-1]:
        if all(abs(i - j) > sep for j in picked):
            picked.append(int(i))
        if len(picked) == n:
            break
    return picked


def _hybrid(time, mag, err, fmin, fmax, oversample, n_candidates, net, device, batch, prefer_shorter):
    """Classical periodograms propose, the network decides.

    Lomb-Scargle and phase dispersion run over the full grid on the GPU (cheap).
    Their best n_candidates peaks each, refined by the same method, plus 2x, 1/2x,
    3x and 1/3x of every one, are scored by the network; the lowest FAP wins.
    About a hundred network evaluations per star instead of one per grid point.
    """
    t, m = np.asarray(time, float), np.asarray(mag, float)
    e = estimate_errors(t, m) if err is None else np.asarray(err, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    span = np.ptp(t)
    df = 1.0 / (oversample * span)
    fmin = fmin or 2.0 / span
    freqs = np.arange(fmin, fmax, df)
    ls = gls(t, m, e, freqs, device)
    pdm = -pdm_theta(t, m, freqs, device)
    cands = []
    for score, fn in ((ls, lambda f: gls(t, m, e, f, device)), (pdm, lambda f: -pdm_theta(t, m, f, device))):
        for i in _top(score, n_candidates):
            fine = np.linspace(freqs[i] - df, freqs[i] + df, 41)
            cands.append(fine[int(np.argmax(fn(fine)))])
    cands = np.array([c * r for c in cands for r in (1.0, 2.0, 0.5, 3.0, 1 / 3)])
    cands = np.unique(np.round(cands[(cands >= fmin) & (cands <= fmax)], 12))
    f_ap = _score(t, m, cands, net, device, batch, False, e)
    order = np.argsort(f_ap)
    peaks = [(1.0 / cands[i], float(f_ap[i])) for i in order[:10]]
    best, best_f = peaks[0]
    note = ""
    if prefer_shorter is None:
        prefer_shorter = True
    half = [p for p in peaks if abs(p[0] - best / 2) < 0.01 * best / 2]
    if prefer_shorter and half and half[0][1] <= 1.25 * best_f:
        note = f"twice-period alias: {best:.6g} d scores about the same as {half[0][0]:.6g} d; reporting the shorter"
        best, best_f = half[0]
    o = np.argsort(cands)
    return Periodogram(cands[o], f_ap[o], peaks, best, best_f, note)
