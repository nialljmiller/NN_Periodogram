"""The one API: NN false-alarm probabilities and NN periodograms."""

from dataclasses import dataclass, field

import numpy as np
import torch

from . import hub
from .classical import gls, gls_fast, mhls_fast, pdm_theta
from .features import (channels_from_phase, fold_phase, global_features, prepare, profile_views,
                       robust_normalise)

_MODELS = {}
# The networks are trained on at most this many points per light curve (a fresh random
# subset each time), so scoring a random subset of this size loses nothing they were
# trained to use and keeps the network step to milliseconds at any length.
MAX_POINTS = 5000
# Test-time augmentation: average the network's logit over 1, 2 or 4 label-preserving
# versions of each fold (as is; magnitudes flipped; time reversed; both), all seen in training.
TTA = 1


def _days(x):
    """Plain float days from numbers, astropy Time (-> JD) or Quantity (-> day); astropy optional."""
    if hasattr(x, "jd") and hasattr(x, "scale"):          # astropy.time.Time
        return np.asarray(x.jd, float)
    if hasattr(x, "unit") and hasattr(x, "to_value"):      # astropy Quantity
        return np.asarray(x.to_value("day"), float)
    return np.asarray(x, float)


def _plain(x):
    """Magnitudes/fluxes and their errors: strip any astropy unit."""
    if x is None:
        return None
    return np.asarray(x.value if hasattr(x, "unit") and hasattr(x, "value") else x, float)


def _freq(x):
    """Frequencies in 1/day from numbers or a Quantity (frequency or period)."""
    if x is None or not hasattr(x, "unit"):
        return x
    try:
        return float(x.to_value("1/day"))
    except Exception:
        return 1.0 / float(x.to_value("day"))


def _model(name, device, path=None):
    key = (name, str(device), path)
    if key not in _MODELS:
        if path and "," in str(path):           # several weight files: an ensemble
            from .models import Ensemble
            _MODELS[key] = Ensemble([hub.load(name, path=p, device=device) for p in str(path).split(",")]).eval()
        else:
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
def _score_binned(t, m, e, freqs, model, device, max_elements=2 ** 24, max_points=MAX_POINTS, tta=None, embed=False):
    """Light curves longer than max_points are scored on a fixed random subset of that size
    (as in training); batches shrink as light curves grow to bound memory."""
    t, m = np.asarray(t, float), np.asarray(m, float)
    e = estimate_errors(t, m) if e is None else np.asarray(e, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    if max_points and len(t) > max_points:
        keep = np.sort(np.random.default_rng(0).choice(len(t), max_points, replace=False))
        t, m, e = t[keep], m[keep], e[keep]
    w = torch.ones(1, len(t), device=device)
    z, ze = robust_normalise(torch.tensor(m, dtype=torch.float32, device=device)[None],
                             torch.tensor(e, dtype=torch.float32, device=device)[None], w)
    g = global_features(z, ze, w)
    tt = torch.tensor(t - t.min(), dtype=torch.float64)   # phase origin = first point: same answer for MJD, JD, ...
    batch = max(1, min(4096, max_elements // len(t)))     # bound both the point arrays and the network activations
    out = np.empty(len(freqs), np.float32)
    views = getattr(model, "views", (1.0,))
    batch = max(1, batch // len(views))
    if tta is None:                                  # the model's own default, else the package's
        tta = getattr(model, "tta", None) or TTA
    variants = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)][:max(1, tta)]
    batch = max(1, batch // len(variants))
    emb_out = None
    for s in range(0, len(freqs), batch):
        f = torch.tensor(freqs[s:s + batch], dtype=torch.float64)
        B = len(f)
        x = profile_views(tt[None, :], f, z.expand(B, -1), ze.expand(B, -1), w.expand(B, -1), views, device)
        # The augmented folds come from this one: flipping the magnitudes negates the mean
        # channels (every 4th), reversing time mirrors the phase bins. The global features
        # depend on z^2 only, so they are unchanged.
        sign = torch.ones(x.shape[1], 1, device=x.device, dtype=x.dtype)
        sign[0::4] = -1.0
        xs = []
        for sign_m, sign_t in variants:
            xv = x * sign if sign_m < 0 else x
            xs.append(xv.flip(-1) if sign_t < 0 else xv)
        if embed:
            em = model.embed(torch.cat(xs), g.expand(B * len(xs), -1)).float()
            head = model.members[0].head if hasattr(model, "members") else model.head
            logit = (model.logits(torch.cat(xs), g.expand(B * len(xs), -1)).float() if hasattr(model, "members")
                     else head(em).squeeze(-1)).view(len(xs), B).mean(0)
            em = em.view(len(xs), B, -1).mean(0).cpu().numpy()
            if emb_out is None:
                emb_out = np.empty((len(freqs), em.shape[1]), np.float32)
            emb_out[s:s + B] = em
        else:
            logit = model.logits(torch.cat(xs), g.expand(B * len(xs), -1)).float().view(len(xs), B).mean(0)
        out[s:s + B] = 1.0 - torch.sigmoid(logit).cpu().numpy()   # P(real) -> FAP
    return (out, emb_out) if embed else out


@torch.no_grad()
def _score_points(t, m, e, freqs, model, device, max_points=MAX_POINTS, tta=None, embed=False):
    """Scoring for PointBinNet (learned binning): the augmented versions are recomputed."""
    from .features import point_inputs
    t, m = np.asarray(t, float), np.asarray(m, float)
    e = estimate_errors(t, m) if e is None else np.asarray(e, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    if max_points and len(t) > max_points:
        keep = np.sort(np.random.default_rng(0).choice(len(t), max_points, replace=False))
        t, m, e = t[keep], m[keep], e[keep]
    w = torch.ones(1, len(t), device=device)
    z, ze = robust_normalise(torch.tensor(m, dtype=torch.float32, device=device)[None],
                             torch.tensor(e, dtype=torch.float32, device=device)[None], w)
    g = global_features(z, ze, w)
    tt = torch.tensor(t - t.min(), dtype=torch.float64)
    if tta is None:
        tta = getattr(model, "tta", None) or TTA
    variants = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)][:max(1, tta)]
    per = len(t) * len(model.views) * len(model.resolutions) * 24      # floats per fold in the point network
    batch = max(1, min(1024, 2 ** 26 // per) // len(variants))
    out = np.empty(len(freqs), np.float32); emb_out = None
    for s in range(0, len(freqs), batch):
        f = torch.tensor(freqs[s:s + batch], dtype=torch.float64)
        B = len(f)
        logit, em = 0.0, 0.0
        for sign_m, sign_t in variants:
            zz = (z * sign_m).expand(B, -1)
            inp = point_inputs(sign_t * tt[None, :], f, zz, ze.expand(B, -1), w.expand(B, -1), model.views, device)
            if sign_t < 0:                                   # reversed time: tn -> 1 - tn
                inp = inp[:4] + (1.0 - inp[4],)
            e_ = model.embed(inp, g.expand(B, -1)).float()
            logit = logit + model.head(e_).squeeze(-1)
            if embed:
                em = em + e_
        out[s:s + B] = 1.0 - torch.sigmoid(logit / len(variants)).cpu().numpy()
        if embed:
            em = (em / len(variants)).cpu().numpy()
            if emb_out is None:
                emb_out = np.empty((len(freqs), em.shape[1]), np.float32)
            emb_out[s:s + B] = em
    return (out, emb_out) if embed else out


def _score(t, m, freqs, model, device, batch, exact=False, err=None, tta=None):
    kind = getattr(model, "input_kind", "legacy")
    if kind == "points":
        return _score_points(t, m, err, freqs, model, device, tta=tta)
    if kind == "binned":
        return _score_binned(t, m, err, freqs, model, device, tta=tta)
    return _score_legacy(t, m, freqs, model, device, batch, exact)


def fap(time, mag, period, err=None, model=hub.DEFAULT, device=None, batch=1024, weights=None, exact=False, tta=None):
    """Neural-network false-alarm probability of one or more trial periods.

    Parameters
    ----------
    time : array-like, astropy Time or Quantity
        Observation times, in days unless they carry units. Any length from ~10 points.
    mag : array-like or Quantity
        Magnitudes (or fluxes; only the shape of the folded curve matters).
    period : float, array-like or Quantity
        Trial period(s), in days unless they carry units.
    err : array-like or Quantity, optional
        Per-point uncertainties. Estimated from point-to-point scatter if omitted.
    model : str
        Registered model name (see ``nn_periodogram.hub.MODELS``).
    device : str or torch.device, optional
        ``"cpu"``, ``"cuda"`` or ``"mps"``. Default: the best available.
    batch : int
        Trial periods per pass (legacy model only; newer models size batches automatically).
    weights : str, optional
        Path to a local weights file, bypassing the download.
    exact : bool
        Legacy model only: compute features in float64 on the CPU, matching the 2022 code to ~1e-6.
    tta : int, optional
        Test-time augmentation: average the network over 1, 2 or 4 label-preserving versions
        of each fold (as is, magnitudes flipped, time reversed, both). More is slightly more
        accurate and proportionally slower. Default: the package setting ``TTA``.

    Returns
    -------
    float or numpy.ndarray
        FAP in [0, 1] per period; low means the fold looks like a real periodic signal.
    """
    device = torch.device(device) if device else hub.default_device()
    t, m, err = _days(time), _plain(mag), _plain(err)
    periods = np.atleast_1d(_days(period))
    out = _score(t, m, 1.0 / periods, _model(model, device, weights), device, batch, exact, err, tta)
    return out if np.ndim(period) else float(out[0])


@dataclass
class Periodogram:
    freq: np.ndarray
    fap: np.ndarray
    peaks: list = field(default_factory=list)      # [(period, fold FAP)] distinct periods, best first
    best_period: float = np.nan
    best_fap: float = np.nan        # FAP of the search: corrected for the many periods tried
    note: str = ""
    best_fap_fold: float = np.nan   # the network's FAP of the best fold alone

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
                prefer_shorter=None, method=None, n_candidates=40, proposers=("ls", "mhls"), calibrated=True,
                model=hub.DEFAULT, device=None, batch=1024, weights=None, exact=False, tta=None):
    """Find the period of a light curve.

    Parameters
    ----------
    time, mag, err
        As for :func:`fap`.
    fmin, fmax : float or Quantity, optional
        Frequency range in 1/day (or any frequency/period Quantity). ``fmin`` defaults to
        2 / baseline; ``fmax`` to 10 / day.
    oversample : float
        Grid samples per 1/baseline.
    method : {"hybrid", "full"}, optional
        ``"hybrid"`` (default for the new models): fast Lomb-Scargle and multi-harmonic
        Lomb-Scargle periodograms propose candidates (their best peaks and each one's 2x,
        1/2x, 3x and 1/3x) and the network chooses among them.
        ``"full"``: the network scores every grid frequency, then refines the best
        ``n_peaks`` peaks. Slower, and independent of the classical methods.
    n_candidates : int
        Hybrid: peaks taken from each proposing periodogram.
    proposers : tuple of {"ls", "mhls", "pdm"}
        Hybrid: periodograms that propose candidates. ``"ls"`` (fast generalised
        Lomb-Scargle) and ``"mhls"`` (fast 3-harmonic Lomb-Scargle) cost O(N + M log M);
        ``"pdm"`` (phase dispersion) costs O(N M) and is mainly for comparison.
    n_peaks, refine : int
        Full: peaks refined, and fine-grid points per side.
    prefer_shorter : bool, optional
        When the best period's half scores within 25% of it, report the half (and say so in
        ``note``). The default belongs to the model: on for models that cannot see 2P and P/2
        side by side, off for those that can.
    calibrated : bool
        Report ``best_fap`` as the false-alarm probability of the whole search: the fraction
        of pure-noise light curves of similar size (real survey cadences; white, red and
        outlier-ridden noise) whose best fold scores as well. Searching hundreds of periods
        makes some fold look good by chance, so the best fold's own FAP
        (``best_fap_fold``) is far too optimistic for detection. Default True.
    model, device, batch, weights, exact, tta
        As for :func:`fap`.

    Returns
    -------
    Periodogram
        ``best_period``, ``best_fap`` (of the search), ``best_fap_fold``, ``peaks``
        [(period, fold FAP), ...], the scored ``freq`` and ``fap`` arrays, a ``note``, and
        ``plot()``.
    """
    device = torch.device(device) if device else hub.default_device()
    net = _model(model, device, weights)
    time, mag, err, fmin, fmax = _days(time), _plain(mag), _plain(err), _freq(fmin), _freq(fmax)
    if prefer_shorter is None:
        prefer_shorter = getattr(net, "prefer_shorter", True)
    if method is None:          # the 2022 network only works as an exhaustive search
        method = "full" if getattr(net, "input_kind", "legacy") == "legacy" else "hybrid"
    if method == "hybrid":
        res = _hybrid(time, mag, err, fmin, fmax, oversample, n_candidates, net, device, batch, prefer_shorter,
                      proposers, tta)
        return _calibrate(res, model, np.isfinite(np.asarray(mag, float)).sum(), calibrated)
    t, m = np.asarray(time, float), np.asarray(mag, float)
    span = np.ptp(np.asarray(time, float)[np.isfinite(time)])
    df = 1.0 / (oversample * span)
    fmin = fmin or 2.0 / span
    freqs = np.arange(fmin, fmax, df)
    f_ap = _score(t, m, freqs, net, device, batch, exact, err, tta)

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
        ff = _score(t, m, fine, net, device, batch, exact, err, tta)
        k = int(np.argmin(ff))
        peaks.append((1.0 / fine[k], float(ff[k])))
    peaks.sort(key=lambda p: p[1])

    best, best_f = peaks[0]
    note = ""
    # A model that cannot tell P from 2P scores them alike; then the shorter of a tied pair is
    # reported (this helps the 2022 model; periods-v4 sees 2P and P/2 itself and needs no help).
    half = [p for p in peaks if abs(p[0] - best / 2) < 0.01 * best / 2]
    if prefer_shorter and half and half[0][1] <= 1.25 * best_f:
        note = f"twice-period alias: {best:.6g} d scores about the same as {half[0][0]:.6g} d; reporting the shorter"
        best, best_f = half[0]
    return _calibrate(Periodogram(freqs, f_ap, peaks, best, best_f, note), model,
                      np.isfinite(np.asarray(mag, float)).sum(), calibrated and method == "hybrid")


def _calibrate(res, model, n_points, calibrated):
    """Fold FAP of the best period -> FAP of the search (see calibrate_fap.py)."""
    res.best_fap_fold = res.best_fap
    table = hub.calibration(model) if calibrated else None
    if table is None:
        return res
    sig = -np.log10(max(res.best_fap_fold, 1e-12))
    lg = np.log10(max(n_points, 1))
    b = next((b for b in table["bins"] if lg < b["log10_n"][1]), table["bins"][-1])
    null = np.asarray(b["null_sig"])
    tail = b["tail"]
    if sig > tail["start"]:
        res.best_fap = float(tail["frac"] * np.exp(-tail["rate"] * (sig - tail["start"])))
    else:
        res.best_fap = float((np.sum(null >= sig) + 1) / (len(null) + 1))
    return res


def _top(score, n, sep=5):
    """Indices of the n highest values, at least `sep` grid steps apart."""
    picked = []
    for i in np.argsort(score)[::-1]:
        if all(abs(i - j) > sep for j in picked):
            picked.append(int(i))
        if len(picked) == n:
            break
    return picked


HARMONICS = (1.0, 2.0, 0.5, 3.0, 1 / 3)


def _grid_peaks(power, n, f0, df, sep=5):
    """Frequencies of the n strongest grid peaks (>= sep steps apart), each located to a
    fraction of a grid step by a parabola through it and its neighbours."""
    out = []
    for i in _top(power, n, sep):
        x = float(i)
        if 0 < i < len(power) - 1:
            a, b, c = power[i - 1], power[i], power[i + 1]
            den = a - 2 * b + c
            if den < 0:
                x += 0.5 * (a - c) / den
        out.append(f0 + df * x)
    return out


def _hybrid_candidates(time, mag, err, fmin, fmax, oversample, n_candidates, net, device, batch,
                       proposers=("ls", "mhls"), tta=None):
    """Fast classical periodograms propose, the network decides.

    Proposers run over the whole frequency grid in O(N + M log M) ("ls": generalised
    Lomb-Scargle, "mhls": 3-harmonic Lomb-Scargle; "pdm": brute-force phase dispersion,
    O(N M), optional). Their n_candidates strongest peaks and each one's 2x, 1/2x, 3x and
    1/3x are scored by the network, which sees the folds at P, 2P and P/2 together; the
    lowest FAP wins and only that period is refined exactly.
    """
    t, m = np.asarray(time, float), np.asarray(mag, float)
    e = estimate_errors(t, m) if err is None else np.asarray(err, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    span = np.ptp(t)
    df = 1.0 / (oversample * span)
    fmin = fmin or 2.0 / span
    nf = max(1, int(np.floor((fmax - fmin) / df)))
    cands, powers = [], {}
    for name in proposers:
        if name == "ls":
            power = gls_fast(t, m, e, fmin, df, nf, device)
        elif name == "mhls":
            power = mhls_fast(t, m, e, fmin, df, nf, 3, device)
        elif name == "pdm":
            power = -pdm_theta(t, m, fmin + df * np.arange(nf), device)
        else:
            raise ValueError(f"unknown proposer {name!r}")
        powers[name] = power
        cands += _grid_peaks(power, n_candidates, fmin, df)
    cands = np.array([c * r for c in cands for r in HARMONICS])
    cands = np.unique(np.round(cands[(cands >= fmin) & (cands <= fmax)], 12))
    f_ap = _score(t, m, cands, net, device, batch, False, e, tta)
    order = np.argsort(f_ap)
    peaks = []                                    # distinct periods (>= 1% apart), best first
    for i in order:
        if all(abs(cands[i] * q[0] - 1) > 0.01 for q in peaks):
            peaks.append((1.0 / cands[i], float(f_ap[i])))
        if len(peaks) == 10:
            break
    return {"t": t, "m": m, "e": e, "df": df, "fmin": fmin, "cands": cands, "f_ap": f_ap, "peaks": peaks,
            "powers": powers}


def _hybrid(time, mag, err, fmin, fmax, oversample, n_candidates, net, device, batch, prefer_shorter,
            proposers=("ls", "mhls"), tta=None):
    """See _hybrid_candidates; then the ranker (if the model has one) picks from the shortlist,
    and the winner is refined exactly."""
    c = _hybrid_candidates(time, mag, err, fmin, fmax, oversample, n_candidates, net, device, batch, proposers, tta)
    t, m, e, df, fmin, cands, f_ap, peaks, powers = (c[k] for k in ("t", "m", "e", "df", "fmin", "cands", "f_ap",
                                                                     "peaks", "powers"))
    ranker = getattr(net, "ranker", None)
    if ranker is not None and len(peaks) > 1:     # second stage: compare the shortlist together
        feats = shortlist_features(t, m, e, np.array([1 / p for p, _ in peaks]), np.array([a for _, a in peaks]),
                                   powers, fmin, df, net, device, tta)
        with torch.no_grad():
            sc = ranker(*(torch.tensor(feats[k], device=device)[None] for k in ("emb", "scalar", "rel")),
                        torch.ones(1, len(peaks), dtype=torch.bool, device=device))[0]
        k = int(torch.argmax(sc))
        peaks = [peaks[k]] + peaks[:k] + peaks[k + 1:]
    best, best_f = peaks[0]
    note = ""
    half = [p for p in peaks if abs(p[0] - best / 2) < 0.01 * best / 2]
    if prefer_shorter and half and half[0][1] <= 1.25 * best_f:
        note = f"twice-period alias: {best:.6g} d scores about the same as {half[0][0]:.6g} d; reporting the shorter"
        best, best_f = half[0]
    # exact refinement of the winner only: tightest fold within +-1 grid step
    fine = np.linspace(1 / best - df, 1 / best + df, 41)
    fine = fine[fine > 0]
    best = 1.0 / fine[int(np.argmin(pdm_theta(t, m, fine, device)))]
    peaks[0 if not note else next(i for i, p in enumerate(peaks) if p == half[0])] = (best, best_f)
    o = np.argsort(cands)
    return Periodogram(cands[o], f_ap[o], peaks, best, best_f, note)


def relation_matrix(freqs, df):
    """(K, K, len(RELATIONS)) one-hot: how candidate i relates to candidate j."""
    from .models import RELATIONS
    K = len(freqs); R = np.zeros((K, K, len(RELATIONS)), np.float32)
    ratios = {"same": 1.0, "x2": 2.0, "x1/2": 0.5, "x3": 3.0, "x1/3": 1 / 3, "x3/2": 1.5, "x2/3": 2 / 3}
    for i in range(K):
        for j in range(K):
            r = freqs[i] / freqs[j]
            for name, v in ratios.items():
                if abs(r / v - 1) < 0.01:
                    R[i, j, RELATIONS.index(name)] = 1
            d = abs(freqs[i] - freqs[j])
            if any(abs(d - k / 0.99726957) < max(3 * df, 0.003) for k in (1, 2)):
                R[i, j, RELATIONS.index("daily alias")] = 1
            if abs(d - 1 / 365.25) < max(1.5 * df, 2e-4) and i != j:
                R[i, j, RELATIONS.index("yearly alias")] = 1
    return R


def shortlist_features(t, m, e, freqs, faps, powers, fmin, df, net, device, tta=None):
    """Inputs of the candidate ranker for a shortlist (freqs, their fold FAPs): the network's
    fold summaries (TTA-averaged), per-candidate scalars and pairwise relations. Used both to
    build the ranker's training data and inside periodogram(), so the two cannot drift."""
    scorer = _score_points if getattr(net, "input_kind", "") == "points" else _score_binned
    _, emb = scorer(t, m, e, freqs, net, device, tta=tta, embed=True)
    span = np.ptp(t)
    def at(name):
        if name not in powers:
            return np.zeros(len(freqs))
        return np.interp((freqs - fmin) / df, np.arange(len(powers[name])), powers[name])
    fa = np.clip(np.asarray(faps, float), 1e-7, 1 - 1e-7)
    scalar = np.stack([np.log10(fa), np.log(fa / (1 - fa)), at("ls"), at("mhls"), np.log10(freqs),
                       np.log10(freqs * span), np.arange(len(freqs)) / 10.0,
                       np.full(len(freqs), np.log10(len(t)))], 1).astype(np.float32)
    return {"emb": emb.astype(np.float32), "scalar": scalar, "rel": relation_matrix(np.asarray(freqs), df),
            "freqs": np.asarray(freqs, float)}


SIDEREAL_DAY = 0.99726957     # days; daily sampling alias is |f +- 1/SIDEREAL_DAY|
YEAR = 365.25


@dataclass
class PeriodCheck:
    period: float                                   # the period you asked about
    best_period: float                              # the period the network prefers
    best_fap: float
    relation: str                                   # how best relates to period: "P", "2P", "P/2", "alias +1/d", ...
    candidates: list = field(default_factory=list)  # [(relation, period, fap)] best first

    def __repr__(self):
        rows = "\n".join(f"  {r:<12} {p:12.6f} d   FAP {f:.3g}" for r, p, f in self.candidates)
        return f"PeriodCheck({self.period:.6f} d -> {self.relation} = {self.best_period:.6f} d)\n{rows}"


def check_period(time, mag, period, err=None, model=hub.DEFAULT, device=None, weights=None, tta=None):
    """Check a known period: is it right, a harmonic, or a sampling alias?

    Scores ``period`` (from any method or catalogue) against 2P, P/2, 3P, P/3, 3P/2,
    2P/3 and its +-1/sidereal-day and +-1/year aliases, and reports the one the network
    prefers. Costs a dozen folds, so it is cheap enough for whole catalogues.

    Parameters
    ----------
    time, mag, err, model, device, weights
        As for :func:`fap`.
    period : float or Quantity
        The period to check, in days unless it carries units.

    Returns
    -------
    PeriodCheck
        ``best_period``, ``best_fap``, ``relation`` (e.g. ``"P"``, ``"2P"``,
        ``"alias +1/d"``) and every candidate as ``(relation, period, fap)``, best first.
    """
    device = torch.device(device) if device else hub.default_device()
    net = _model(model, device, weights)
    time, mag, err = _days(time), _plain(mag), _plain(err)
    period = float(_days(period))
    f0 = 1.0 / period
    cands = [("P", f0), ("2P", f0 / 2), ("P/2", 2 * f0), ("3P", f0 / 3), ("P/3", 3 * f0),
             ("3P/2", f0 * 2 / 3), ("2P/3", f0 * 1.5)]
    for sign in (1, -1):
        cands.append((f"alias {'+' if sign > 0 else '-'}1/d", abs(f0 + sign / SIDEREAL_DAY)))
        cands.append((f"alias {'+' if sign > 0 else '-'}1/yr", abs(f0 + sign / YEAR)))
    cands = [(name, f) for name, f in cands if f > 0]
    t, m = np.asarray(time, float), np.asarray(mag, float)
    faps = _score(t, m, np.array([f for _, f in cands]), net, device, 1024, False, err, tta)
    rows = sorted(((name, 1.0 / f, float(a)) for (name, f), a in zip(cands, faps)), key=lambda r: r[2])
    best = rows[0]
    return PeriodCheck(period, best[1], best[2], best[0], rows)
