import numpy as np
import pytest
import torch

from nn_periodogram.features import fold_channels, legacy_channels, prepare


def _lc(n=260, period=0.5078, seed=1):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 3400, n))
    m = 14.5 + 0.2 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.03, n)
    return prepare(t, m)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_batched_channels_match_legacy(seed):
    t, m = _lc(seed=seed)
    freqs = np.array([1 / 0.5078, 1.234, 0.61, 3.7])
    got = fold_channels(torch.tensor(t), torch.tensor(m), torch.tensor(freqs)).numpy()
    for k, f in enumerate(freqs):
        ph = np.mod(t * f, 1.0)
        ph[ph <= 0] += 1
        ref = legacy_channels(m, ph)
        np.testing.assert_allclose(got[k], ref, rtol=1e-6, atol=1e-9, err_msg=f"freq {f}")


def test_prepare_pads_and_trims():
    t, m = prepare(np.arange(50.0), np.arange(50.0))
    assert len(t) == 200 and m.min() == 0 and m.max() == 1
    t, m = prepare(np.arange(500.0), np.arange(500.0))
    assert len(t) == 200 and np.all(np.diff(t) > 0)
