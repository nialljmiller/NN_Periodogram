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


class _Res(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 5, padding=2 * dilation, dilation=dilation, padding_mode="circular")
        self.c2 = nn.Conv1d(c, c, 5, padding=2 * dilation, dilation=dilation, padding_mode="circular")
        # GroupNorm, not BatchNorm: no running statistics to be thrown off by extreme light curves
        self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)

    def forward(self, x):
        y = torch.relu(self.n1(self.c1(x)))
        return torch.relu(x + self.n2(self.c2(y)))


class PeriodsNet(nn.Module):
    """Circular 1-D CNN over the multi-resolution phase profile, conditioned on global
    light-curve features (number of points, chi^2, errors). Returns P(fold is a real signal)."""
    input_kind = "binned"

    def __init__(self, n_in=16, n_global=3, width=96, dilations=(1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_in, width, 5, padding=2, padding_mode="circular"),
                                  nn.GroupNorm(8, width), nn.ReLU())
        self.blocks = nn.Sequential(*[_Res(width, d) for d in dilations])
        self.glob = nn.Sequential(nn.Linear(n_global, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * width + 32, 128), nn.ReLU(), nn.Linear(128, 1))

    def logits(self, x, g):
        h = self.blocks(self.stem(x))
        return self.head(torch.cat([h.mean(-1), h.amax(-1), self.glob(g)], 1)).squeeze(-1)

    def forward(self, x, g):
        return torch.sigmoid(self.logits(x, g))
