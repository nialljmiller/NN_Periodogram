"""Phase-fold features.

The 2022 NN_FAP network reads 7 channels of 200 values built from a light
curve folded at a trial period.  `legacy_channels` is the original NumPy code,
kept verbatim (including its quirks) as the reference; `fold_channels` builds
the same channels for many trial periods at once in PyTorch.

Channels, in order:
    0  magnitude, sorted by phase
    1  10-nearest-neighbour (distance-weighted) magnitude on a uniform phase grid
    2  running interquartile range over a 2-bin phase window
    3  phase, sorted
    4  channel 1 smoothed with a 10-point box
    5  channel 1 smoothed with a 40-point box
    6  phase step between consecutive sorted points
"""

import numpy as np
import torch

N_POINTS = 200
KNN_K = 10


# --- reference: verbatim from NN_FAP.py (2022) -----------------------------

def _legacy_running_scatter(x, y, N):
    rn = []
    xs = np.linspace(min(x), max(x), num=N)
    for i in range(len(xs)):
        if i < 1:
            check = np.where(x < xs[i + 2])[0]
        if i == len(xs):
            check = np.where(x >= xs[i - 2])[0]
        else:
            check = np.where((x >= xs[i - 2]) & (x <= xs[i]))[0]
        if len(check) > 1:
            q75, q25 = np.percentile(y[check], [75, 25])
            rn.append(abs(q75 - q25))
        else:
            rn.append(0)
    return rn


def _legacy_smooth(y, box_pts):
    return np.convolve(y, np.ones(box_pts) / box_pts, mode="same")


def legacy_channels(mag, phase, N=N_POINTS):
    """Original gen_chan for exactly N finite points (no random trimming)."""
    from sklearn import neighbors
    asort = np.argsort(phase)
    mag, phase = mag[asort], phase[asort]
    knn = neighbors.KNeighborsRegressor(int(N / 20), weights="distance")
    knn_m = knn.fit(phase[:, None], mag).predict(np.linspace(0, 1, num=N)[:, None])
    rn = _legacy_running_scatter(phase, mag, N)
    delta_phase = np.diff(phase, prepend=0.0)
    return np.vstack((mag, knn_m, rn, phase, _legacy_smooth(knn_m, int(N / 20)),
                      _legacy_smooth(knn_m, int(N / 5)), delta_phase))


# --- batched PyTorch version -----------------------------------------------

def _box_same(y, width):
    """np.convolve(y, ones(width)/width, mode='same') along the last axis."""
    n = y.shape[-1]
    full = torch.nn.functional.conv1d(y.reshape(-1, 1, n), y.new_ones(1, 1, width) / width,
                                      padding=width - 1)
    start = (width - 1) // 2
    return full[..., start:start + n].reshape(y.shape)


def fold_phase(time, freqs):
    """(B, N) phases; do this in float64, since time * freq over long baselines needs it."""
    phase = torch.remainder(time[None, :] * freqs[:, None], 1.0)
    return torch.where(phase <= 0, phase + 1, phase)                  # legacy phaser maps 0 -> 1


def fold_channels(time, mag, freqs, N=N_POINTS):
    """Channels for every trial frequency (time, mag (N,), freqs (B,)) -> (B, 7, N)."""
    return channels_from_phase(fold_phase(time, freqs), mag, N)


def channels_from_phase(phase, mag, N=N_POINTS):
    """Channels from (B, N) phases and the (N,) normalised magnitudes; any float dtype/device."""
    time = phase                                                       # only used for device below
    phase, order = torch.sort(phase, dim=1)
    m = mag[order]                                                     # (B, N)

    # distance-weighted KNN on a uniform phase grid (plain distance, no wrap)
    grid = torch.linspace(0, 1, N, device=time.device, dtype=phase.dtype)
    d = (grid[None, :, None] - phase[:, None, :]).abs()                # (B, grid, pts)
    dk, ik = torch.topk(d, KNN_K, dim=2, largest=False)
    mk = torch.gather(m[:, None, :].expand(-1, N, -1), 2, ik)
    exact = dk == 0
    w = torch.where(exact.any(2, keepdim=True), exact.to(dk.dtype), 1.0 / dk.clamp_min(1e-300))
    knn = (w * mk).sum(2) / w.sum(2)

    # running IQR: window [xs[i-2], xs[i]] on a grid spanning the sampled phases;
    # the first two windows are empty in the original, so they are 0
    lo, hi = phase[:, :1], phase[:, -1:]
    xs = lo + (hi - lo) * torch.linspace(0, 1, N, device=time.device, dtype=phase.dtype)[None, :]
    xs[:, -1:] = hi                     # pin the endpoint exactly, as np.linspace does; float32 can fall short
    left = torch.roll(xs, 2, dims=1)
    inside = (phase[:, None, :] >= left[:, :, None]) & (phase[:, None, :] <= xs[:, :, None])
    vals = torch.where(inside, m[:, None, :], torch.nan)
    q = torch.nanquantile(vals, torch.tensor([0.75, 0.25], device=time.device, dtype=vals.dtype), dim=2)
    rn = torch.where(inside.sum(2) > 1, (q[0] - q[1]).abs(), torch.zeros_like(q[0]))
    rn[:, :2] = 0

    dphase = torch.diff(phase, dim=1, prepend=torch.zeros_like(phase[:, :1]))
    return torch.stack([m, knn, rn, phase, _box_same(knn, N // 20), _box_same(knn, N // 5), dphase], 1)


def prepare(time, mag, N=N_POINTS, seed=0):
    """Pick exactly N points (wrap-pad if fewer, fixed random subset if more) and normalise."""
    time, mag = np.asarray(time, float), np.asarray(mag, float)
    ok = np.isfinite(time) & np.isfinite(mag)
    time, mag = time[ok], mag[ok]
    if len(mag) < N:
        time, mag = np.pad(time, (0, N - len(time)), "wrap"), np.pad(mag, (0, N - len(mag)), "wrap")
    elif len(mag) > N:
        keep = np.sort(np.random.default_rng(seed).choice(len(mag), N, replace=False))
        time, mag = time[keep], mag[keep]
    mag = (mag - mag.min()) / (mag.max() - mag.min())
    return time, mag


# --- binned phase profile (NN_Periodogram models) ---------------------------

N_BINS = 64
RESOLUTIONS = (8, 16, 32, 64)
N_PROFILE = 4 * len(RESOLUTIONS)     # channels from binned_profile
N_GLOBAL = 3                         # values from global_features


def binned_profile(phase, z, ze, w, n_bins=N_BINS, resolutions=RESOLUTIONS):
    """Multi-resolution fold summary for light curves of any length.

    phase : (B, L) phases in [0, 1)
    z, ze : (B, L) robust-normalised magnitudes and their errors (same scale)
    w     : (B, L) 1 for real points, 0 for padding; L can be anything
    returns (B, 4 * len(resolutions), n_bins): for each resolution, per bin
    [mean z, scatter of z, mean error, log(1 + count)], repeated up to n_bins.
    Coarse bins keep sparse light curves informative; fine bins resolve shape.
    """
    out = []
    for nb in resolutions:
        idx = torch.clamp((phase * nb).long(), 0, nb - 1)
        n = torch.zeros(phase.shape[0], nb, device=phase.device, dtype=z.dtype).scatter_add_(1, idx, w)
        s1 = torch.zeros_like(n).scatter_add_(1, idx, w * z)
        s2 = torch.zeros_like(n).scatter_add_(1, idx, w * z * z)
        se = torch.zeros_like(n).scatter_add_(1, idx, w * ze)
        safe = n.clamp_min(1)
        mean = torch.where(n > 0, s1 / safe, torch.zeros_like(n))
        var = torch.where(n > 1, (s2 / safe - mean ** 2).clamp_min(0), torch.zeros_like(n))
        err = torch.where(n > 0, se / safe, torch.zeros_like(n))
        feats = torch.stack([mean, var.sqrt(), err, torch.log1p(n)], 1)          # (B, 4, nb)
        out.append(feats.repeat_interleave(n_bins // nb, dim=2))
    return torch.cat(out, 1)


def global_features(z, ze, w):
    """(B, 3): log10 number of points, log10 reduced chi^2 about the median, log10 median error."""
    n = w.sum(1)
    chi2 = (w * (z / ze.clamp_min(1e-3)) ** 2).sum(1) / n.clamp_min(1)
    med_err = torch.nanmedian(torch.where(w > 0, ze, torch.nan), dim=1).values
    return torch.stack([torch.log10(n.clamp_min(1)), torch.log10(chi2.clamp_min(1e-3)),
                        torch.log10(med_err.clamp_min(1e-3))], 1)


def robust_normalise(mag, err, w):
    """(m - median) / s and err / s per row, s = 1.4826 MAD, ignoring padding (w == 0)."""
    big = torch.where(w > 0, mag, torch.nan)
    med = torch.nanmedian(big, dim=1, keepdim=True).values
    s = (torch.nanmedian((big - med).abs(), dim=1, keepdim=True).values * 1.4826).clamp_min(1e-6)
    zero = torch.zeros_like(mag)
    # clipped so a single wild point cannot dominate a fold (or the network's normalisation)
    z = torch.clamp((mag - med) / s, -20, 20)
    ze = torch.clamp(err / s, 1e-3, 20)
    return torch.where(w > 0, z, zero), torch.where(w > 0, ze, zero)


def profile_views(t, freqs, z, ze, w, views=(1.0,), device=None):
    """Binned profiles of the same light curves folded at freqs * r for each r in views.

    t      : (B, L) or (1, L) float64 times (phases are computed in double precision on the
             device, or on the CPU for Apple MPS, which has no float64)
    freqs  : (B,) float64 trial frequencies
    z, ze, w : (B, L) float32 on `device`
    views=(1.0, 0.5, 2.0) stacks the folds at P, 2P and P/2 side by side, so the
    network can compare a candidate with its harmonics directly.
    returns (B, N_PROFILE * len(views), N_BINS)
    """
    device = device or z.device
    pdev = phase_device(device)
    t, freqs = t.to(pdev, torch.float64), freqs.to(pdev, torch.float64)
    out = []
    for r in views:
        ph = torch.remainder(t * (freqs[:, None] * r), 1.0).to(device=device, dtype=torch.float32)
        out.append(binned_profile(ph, z, ze, w))
    return torch.cat(out, 1)


def phase_device(device):
    """Where double-precision phases are computed: the device itself, except Apple MPS (no float64)."""
    device = torch.device(device)
    return torch.device("cpu") if device.type == "mps" else device


def point_inputs(t, freqs, z, ze, w, views=(1.0,), device=None):
    """Inputs of PointBinNet: per-view phases (B, V, L) and z, ze, w, scaled time (B, L).
    t is (B, L) or (1, L) float64 (phases in double precision, as in profile_views)."""
    device = device or z.device
    pdev = phase_device(device)
    tt, ff = t.to(pdev, torch.float64), freqs.to(pdev, torch.float64)
    ph = torch.stack([torch.remainder(tt * (ff[:, None] * r), 1.0) for r in views], 1).to(device, torch.float32)
    t32 = tt.to(device, torch.float32).expand(z.shape[0], -1)
    big = torch.where(w > 0, t32, torch.full_like(t32, float("inf"))).amin(1, keepdim=True)
    top = torch.where(w > 0, t32, torch.full_like(t32, -float("inf"))).amax(1, keepdim=True)
    tn = torch.where(w > 0, (t32 - big) / (top - big).clamp_min(1e-9), torch.zeros_like(t32))
    return ph, z, ze, w, tn
