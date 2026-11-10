"""Network definitions. LegacyNNFAP is an exact PyTorch port of the 2022 NN_FAP network.

Keras GRU with reset_after=True, gates packed [z, r, h] in kernel (in, 3u),
recurrent_kernel (u, 3u) and bias (2, 3u) = [input bias, recurrent bias]:

    z  = rec_act(x Wz + bz + h Uz + cz)
    r  = rec_act(x Wr + br + h Ur + cr)
    hh = tanh(x Wh + bh + r * (h Uh + ch))
    h  = z * h + (1 - z) * hh

The first layer uses sigmoid for rec_act, the rest Keras 2's hard_sigmoid,
clip(0.2 x + 0.5, 0, 1).  Input is (batch, 7, 200): the 7 NN_FAP channels
are the timesteps.
"""

import torch
import torch.nn as nn


def hard_sigmoid(x):
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class KerasGRU(nn.Module):
    def __init__(self, n_in, units, hard, return_sequences):
        super().__init__()
        self.units, self.hard, self.return_sequences = units, hard, return_sequences
        self.kernel = nn.Parameter(torch.empty(n_in, 3 * units))
        self.recurrent_kernel = nn.Parameter(torch.empty(units, 3 * units))
        self.bias = nn.Parameter(torch.empty(2, 3 * units))

    def forward(self, x):
        u = self.units
        act = hard_sigmoid if self.hard else torch.sigmoid
        xw = x @ self.kernel + self.bias[0]                  # (b, t, 3u): all timesteps at once
        h = x.new_zeros(x.shape[0], u)
        out = []
        for t in range(x.shape[1]):
            hu = h @ self.recurrent_kernel + self.bias[1]
            z = act(xw[:, t, :u] + hu[:, :u])
            r = act(xw[:, t, u:2 * u] + hu[:, u:2 * u])
            hh = torch.tanh(xw[:, t, 2 * u:] + r * hu[:, 2 * u:])
            h = z * h + (1 - z) * hh
            out.append(h)
        return torch.stack(out, 1) if self.return_sequences else h


class LegacyNNFAP(nn.Module):
    def __init__(self, n_in=200, units=1024, layers=14):
        super().__init__()
        self.grus = nn.ModuleList(
            KerasGRU(n_in if i == 0 else units, units, hard=i > 0, return_sequences=i < layers - 1)
            for i in range(layers))
        self.dense = nn.Linear(units, 1)

    def forward(self, x):
        for g in self.grus:
            x = g(x)
        return torch.sigmoid(self.dense(x)).squeeze(-1)


def from_keras_h5(path):
    """Build the model and load the 2022 Keras weights."""
    import h5py                     # only needed to convert the original Keras file
    m = LegacyNNFAP()
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["layer_names"]]
        grus = [n for n in names if n.startswith("gru")]
        assert len(grus) == len(m.grus), names
        for layer, name in zip(m.grus, grus):
            g = f[name][name]
            layer.kernel.data = torch.from_numpy(g["kernel:0"][:])
            layer.recurrent_kernel.data = torch.from_numpy(g["recurrent_kernel:0"][:])
            layer.bias.data = torch.from_numpy(g["bias:0"][:])
        dname = [n for n in names if n.startswith("dense")][0]
        m.dense.weight.data = torch.from_numpy(f[dname][dname]["kernel:0"][:]).T.contiguous()
        m.dense.bias.data = torch.from_numpy(f[dname][dname]["bias:0"][:])
    return m.eval()


class CircConv1d(nn.Conv1d):
    """Circularly padded 1-D convolution, same parameters as nn.Conv1d(padding_mode="circular").

    On the CPU it runs as one matrix multiply over gathered taps, which BLAS does about
    10x faster than PyTorch's direct CPU convolution for these small sequences; GPUs keep
    the native (cuDNN / MPS) convolution."""

    def __init__(self, c_in, c_out, k, dilation=1):
        super().__init__(c_in, c_out, k, padding=(k // 2) * dilation, dilation=dilation, padding_mode="circular")

    def forward(self, x):
        if x.device.type != "cpu" or torch.is_grad_enabled():
            return super().forward(x)
        B, C, L = x.shape
        K, d = self.kernel_size[0], self.dilation[0]
        idx = (torch.arange(L)[None, :] + (torch.arange(K)[:, None] - K // 2) * d) % L      # (K, L)
        cols = x[:, :, idx]                                                                  # (B, C, K, L)
        cols = cols.permute(0, 3, 2, 1).reshape(B * L, K * C)
        w = self.weight.permute(0, 2, 1).reshape(self.out_channels, K * C)
        out = torch.addmm(self.bias, cols, w.T) if self.bias is not None else cols @ w.T
        return out.reshape(B, L, -1).permute(0, 2, 1)


class _Res(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.c1 = CircConv1d(c, c, 5, dilation)
        self.c2 = CircConv1d(c, c, 5, dilation)
        # GroupNorm, not BatchNorm: no running statistics to be thrown off by extreme light curves
        self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)

    def forward(self, x):
        y = torch.relu(self.n1(self.c1(x)))
        return torch.relu(x + self.n2(self.c2(y)))


class PeriodsNet(nn.Module):
    """Circular 1-D CNN over the multi-resolution phase profile, conditioned on global
    light-curve features (number of points, chi^2, errors). Returns P(fold is a real signal)."""
    input_kind = "binned"

    def __init__(self, n_in=16, n_global=3, width=96, dilations=(1, 2, 4, 8, 1, 2, 4, 8), views=(1.0,)):
        super().__init__()
        # Folds the model looks at for each candidate frequency f: f * r for r in views.
        # (1.0, 0.5, 2.0) = the candidate period P with 2P and P/2 side by side.
        self.views = tuple(views)
        n_in = 16 * len(self.views)
        self.stem = nn.Sequential(CircConv1d(n_in, width, 5),
                                  nn.GroupNorm(8, width), nn.ReLU())
        self.blocks = nn.Sequential(*[_Res(width, d) for d in dilations])
        self.glob = nn.Sequential(nn.Linear(n_global, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * width + 32, 128), nn.ReLU(), nn.Linear(128, 1))

    def embed(self, x, g):
        """The fold summary the head reads: (B, 2 * width + 32)."""
        h = self.blocks(self.stem(x))
        return torch.cat([h.mean(-1), h.amax(-1), self.glob(g)], 1)

    def logits(self, x, g):
        return self.head(self.embed(x, g)).squeeze(-1)

    def forward(self, x, g):
        return torch.sigmoid(self.logits(x, g))


# Relations between two candidate frequencies f_i, f_j that the ranker is told about.
RELATIONS = ("same", "x2", "x1/2", "x3", "x1/3", "x3/2", "x2/3", "daily alias", "yearly alias")


class CandidateRanker(nn.Module):
    """Second stage of the search: looks at the shortlisted candidates together and picks one.

    Each candidate is described by the network's fold summary (embed) and a few scalars
    (fold FAP, periodogram powers, frequency, cycles, rank); attention between candidates is
    biased by how they are related (2x, 1/2x, 3x, aliases, ...), so the choice between a
    period and its harmonics is made by comparison rather than candidate by candidate."""

    def __init__(self, n_emb=224, n_scalar=8, d=128, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_emb + n_scalar, d), nn.ReLU(), nn.Linear(d, d))
        self.rel = nn.ModuleList([nn.Linear(len(RELATIONS), heads, bias=False) for _ in range(layers)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d)) for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.out = nn.Linear(d, 1)
        self.heads = heads

    def forward(self, emb, scalar, rel, mask):
        """emb (B,K,E), scalar (B,K,S), rel (B,K,K,R), mask (B,K) True for real candidates.
        Returns (B,K) scores; padded slots get -inf."""
        h = self.inp(torch.cat([emb, scalar], -1))
        B, K, _ = h.shape
        pad = ~mask
        for rel_lin, attn, ff, n1, n2 in zip(self.rel, self.attn, self.ff, self.n1, self.n2):
            bias = rel_lin(rel).permute(0, 3, 1, 2).reshape(B * self.heads, K, K)
            bias = bias.masked_fill(pad[:, None, None, :].expand(B, self.heads, K, K).reshape(B * self.heads, K, K),
                                    float("-inf"))
            a, _ = attn(h, h, h, attn_mask=bias, need_weights=False)
            h = n1(h + a)
            h = n2(h + ff(h))
        return self.out(h).squeeze(-1).masked_fill(pad, float("-inf"))


class Ensemble(nn.Module):
    """Average of several PeriodsNets' logits (same views). More accurate than any member."""
    input_kind = "binned"

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        views = {tuple(m.views) for m in members}
        assert len(views) == 1, "ensemble members must share their views"
        self.views = views.pop()
        self.prefer_shorter = all(getattr(m, "prefer_shorter", False) for m in members)

    def logits(self, x, g):
        return torch.stack([m.logits(x, g) for m in self.members]).mean(0)

    def embed(self, x, g):
        return torch.stack([m.embed(x, g) for m in self.members]).mean(0)

    def forward(self, x, g):
        return torch.sigmoid(self.logits(x, g))
