"""Fast classical periodograms on the GPU: generalised Lomb-Scargle and phase dispersion.

Used by periodogram(method="hybrid") to propose candidate periods cheaply, and
by the training data verification. Phases are computed in float64 (time x
frequency over long baselines needs it), the rest in float32.
"""

import numpy as np
import torch

from .features import phase_device


def gls(t, y, dy, freqs, device, chunk=None):
    """Generalised Lomb-Scargle power (standard normalisation, floating mean)."""
    w = 1.0 / dy ** 2
    w = w / w.sum()
    y = y - (w * y).sum()
    pdev = phase_device(device)
    tt = torch.tensor(t - t.min(), dtype=torch.float64, device=pdev)
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    yy = float((w * y * y).sum())
    chunk = chunk or max(1, 2 ** 23 // len(t))          # ~2^23 phases per pass: bounded memory at any length
    out = []
    for s in range(0, len(freqs), chunk):
        f = torch.tensor(freqs[s:s + chunk], dtype=torch.float64, device=pdev)
        ph = torch.remainder(f[:, None] * tt[None, :], 1.0).float().to(device) * (2 * np.pi)
        c, sn = torch.cos(ph), torch.sin(ph)
        C, S = c @ wt, sn @ wt
        YC, YS = c @ (wt * yt), sn @ (wt * yt)
        CC = (c * c) @ wt - C * C
        SS = 1.0 - (c * c) @ wt - S * S
        CS = (c * sn) @ wt - C * S
        D = CC * SS - CS * CS
        out.append(((SS * YC ** 2 + CC * YS ** 2 - 2 * CS * YC * YS) / (yy * D)).cpu().numpy())
    return np.concatenate(out)


def pdm_theta(t, y, freqs, device, bins=(16, 32)):
    """Phase dispersion at each frequency: within-bin variance / total variance (lower = tighter fold)."""
    pdev = phase_device(device)
    tt = torch.tensor(t - t.min(), dtype=torch.float64, device=pdev)
    yt = torch.tensor(y - y.mean(), dtype=torch.float32, device=device)
    total = float((yt ** 2).mean())
    best = None
    step = max(1, 2 ** 23 // len(t))                   # bounded memory at any length
    for s in range(0, len(freqs), step):
        f = torch.tensor(freqs[s:s + step], dtype=torch.float64, device=pdev)
        ph = torch.remainder(f[:, None] * tt[None, :], 1.0).float().to(device)
        thetas = []
        for nb in bins:
            idx = torch.clamp((ph * nb).long(), 0, nb - 1)
            n = torch.zeros(len(f), nb, device=device).scatter_add_(1, idx, torch.ones_like(ph))
            s1 = torch.zeros_like(n).scatter_add_(1, idx, yt.expand_as(ph))
            s2 = torch.zeros_like(n).scatter_add_(1, idx, (yt ** 2).expand_as(ph))
            within = (s2 - s1 ** 2 / n.clamp_min(1)).sum(1) / len(t)
            thetas.append(within / total)
        th = torch.minimum(*thetas).cpu().numpy()
        best = th if best is None else np.concatenate([best, th])
    return best




# ---------------------------------------------------------------------------
# Fast Lomb-Scargle (Press & Rybicki 1989): O(N + M log M) instead of O(N M)
# ---------------------------------------------------------------------------

def _extirpolate(x, h, n, order=8):
    """Spread values h at real grid positions x (0 <= x < n) onto n integer nodes so that
    sums of any smooth n-periodic function over the nodes reproduce sums over the points
    (Lagrange weights over `order` neighbouring nodes, wrapping around the grid)."""
    real = x.dtype
    base = torch.floor(x).long() - (order // 2 - 1)
    nodes = base[:, None] + torch.arange(order, device=x.device)
    dx = x[:, None] - nodes.to(real)                     # (N, order)
    out = torch.zeros(n, dtype=h.dtype, device=h.device)
    for i in range(order):
        w = torch.ones_like(x)
        for j in range(order):
            if j != i:
                w = w * dx[:, j] / (i - j)
        out.index_add_(0, torch.remainder(nodes[:, i], n), h * w.to(h.dtype))
    return out


def _trig_sums(t, h, f0, df, nf, oversample=4):
    """sum_j h_j exp(2 pi i f t_j) for f = f0 + k df, k = 0..nf-1, via one FFT.
    t must start at 0 and be in the working precision (float64 on CPU)."""
    real = t.dtype
    cplx = torch.complex128 if real == torch.float64 else torch.complex64
    nfft = 1 << int(np.ceil(np.log2(oversample * nf)))
    hc = h.to(cplx) * torch.exp(2j * np.pi * torch.remainder(f0 * t, 1.0)).to(cplx)   # shift to f0
    x = torch.remainder(df * t * nfft, nfft)
    grid = _extirpolate(x, hc, nfft)
    return torch.fft.ifft(grid)[:nf] * nfft


def gls_fast(t, y, dy, f0, df, nf, device=None):
    """Generalised Lomb-Scargle power on the regular grid f0 + k df (k < nf), same
    normalisation as gls(), via Press & Rybicki extirpolation and FFTs.

    Runs in float64 on the CPU and float32 on GPUs; precise enough for finding peaks
    (use gls() to evaluate a few exact values)."""
    device = torch.device(device or "cpu")
    real = torch.float64 if device.type == "cpu" else torch.float32
    if device.type == "mps":                 # no float64 / limited complex support on MPS
        device, real = torch.device("cpu"), torch.float64
    t = np.asarray(t, float); y = np.asarray(y, float); dy = np.asarray(dy, float)
    w = 1.0 / dy ** 2
    w = w / w.sum()
    y = y - (w * y).sum()
    yy = float((w * y * y).sum())
    tt = torch.tensor(t - t.min(), dtype=torch.float64, device=device).to(real)
    wt = torch.tensor(w, dtype=real, device=device)
    yt = torch.tensor(y, dtype=real, device=device)
    A = _trig_sums(tt, wt, f0, df, nf)                    # sum w e^{iwt}
    B = _trig_sums(tt, wt * yt, f0, df, nf)               # sum w y e^{iwt}
    A2 = _trig_sums(tt, wt, 2 * f0, 2 * df, nf)           # sum w e^{2iwt}
    C, S = A.real, A.imag
    YC, YS = B.real, B.imag
    CC = 0.5 * (1 + A2.real) - C * C
    SS = 0.5 * (1 - A2.real) - S * S
    CS = 0.5 * A2.imag - C * S
    D = CC * SS - CS * CS
    p = (SS * YC ** 2 + CC * YS ** 2 - 2 * CS * YC * YS) / (yy * D)
    return p.cpu().numpy().astype(np.float64)


def mhls_fast(t, y, dy, f0, df, nf, nterms=2, device=None):
    """Multi-harmonic (Fourier series, `nterms` harmonics, floating mean) Lomb-Scargle power
    on the grid f0 + k df: the fraction of weighted variance explained at each frequency.
    Non-sinusoidal signals (eclipsing binaries, RR Lyrae) peak at their true period rather
    than a harmonic. O(N + M log M) via the same FFT trig sums as gls_fast (Palmer 2009);
    nterms=1 is the generalised Lomb-Scargle."""
    device = torch.device(device or "cpu")
    real = torch.float64 if device.type == "cpu" else torch.float32
    if device.type == "mps":
        device, real = torch.device("cpu"), torch.float64
    t = np.asarray(t, float); y = np.asarray(y, float); dy = np.asarray(dy, float)
    w = 1.0 / dy ** 2
    w = w / w.sum()
    y = y - (w * y).sum()
    yy = float((w * y * y).sum())
    tt = torch.tensor(t - t.min(), dtype=torch.float64, device=device).to(real)
    wt = torch.tensor(w, dtype=real, device=device)
    yt = torch.tensor(y, dtype=real, device=device)
    K = nterms
    A = [torch.ones(nf, dtype=tt.dtype, device=device).to(torch.complex128 if real == torch.float64 else torch.complex64)]
    A += [_trig_sums(tt, wt, m * f0, m * df, nf) for m in range(1, 2 * K + 1)]   # sum w e^{i m w t}
    B = [_trig_sums(tt, wt * yt, k * f0, k * df, nf) for k in range(1, K + 1)]   # sum w y e^{i k w t}
    Cm = lambda m: A[abs(m)].real                                               # noqa: E731
    Sm = lambda m: torch.sign(torch.tensor(float(m))) * A[abs(m)].imag if m else torch.zeros_like(A[0].real)  # noqa: E731
    n = 2 * K + 1
    M = torch.empty(nf, n, n, dtype=real, device=device)
    rhs = torch.zeros(nf, n, dtype=real, device=device)
    M[:, 0, 0] = 1.0
    for k in range(1, K + 1):
        ck, sk = 2 * k - 1, 2 * k
        M[:, 0, ck] = M[:, ck, 0] = Cm(k)
        M[:, 0, sk] = M[:, sk, 0] = Sm(k)
        rhs[:, ck], rhs[:, sk] = B[k - 1].real, B[k - 1].imag
        for l in range(1, K + 1):
            cl, sl = 2 * l - 1, 2 * l
            M[:, ck, cl] = 0.5 * (Cm(k - l) + Cm(k + l))
            M[:, sk, sl] = 0.5 * (Cm(k - l) - Cm(k + l))
            M[:, ck, sl] = M[:, sl, ck] = 0.5 * (Sm(k + l) - Sm(k - l))
    M = M + 1e-10 * torch.eye(n, dtype=real, device=device)
    sol = torch.linalg.solve(M, rhs[..., None])[..., 0]
    return ((rhs * sol).sum(1) / yy).cpu().numpy().astype(np.float64)
