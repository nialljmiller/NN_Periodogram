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


