"""Every search mode runs end to end on each available device, with matching answers."""
import numpy as np
import pytest
import torch

import nn_periodogram as nnp
import importlib

P = importlib.import_module("nn_periodogram.periodogram")
from nn_periodogram.classical import gls, pdm_theta
from nn_periodogram.models import PeriodsNet

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + \
          (["mps"] if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else [])


def _lc(seed=0, n=300, period=0.731):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1500, n))
    return t, 14 + 0.3 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.05, n), np.full(n, 0.05), period


@pytest.mark.parametrize("device", DEVICES)
def test_classical_find_the_period(device):
    t, m, e, period = _lc()
    f = np.linspace(0.05, 5, 30000)
    assert abs(1 / f[np.argmax(gls(t, m, e, f, torch.device(device)))] / period - 1) < 0.01
    assert abs(1 / f[np.argmin(pdm_theta(t, m, f, torch.device(device)))] / period - 1) < 0.01


@pytest.mark.parametrize("device", DEVICES)
def test_all_modes_run(device):
    torch.manual_seed(0)
    net = PeriodsNet(views=(1.0, 0.5, 2.0)).eval()
    P._MODELS[("test-net", str(torch.device(device)), None)] = net.to(device)
    t, m, e, period = _lc()
    for method in ("hybrid", "full"):
        res = nnp.periodogram(t, m, err=e, fmax=3.0, model="test-net", device=device, method=method)
        assert np.isfinite(res.best_period) and len(res.peaks) > 0
    chk = nnp.check_period(t, m, period, err=e, model="test-net", device=device)
    assert len(chk.candidates) >= 7


def test_cpu_and_gpu_agree():
    if len(DEVICES) == 1:
        pytest.skip("no GPU here")
    t, m, e, _ = _lc(1)
    f = np.linspace(0.05, 5, 5000)
    a, b = gls(t, m, e, f, torch.device("cpu")), gls(t, m, e, f, torch.device(DEVICES[1]))
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_astropy_inputs_match_plain():
    u = pytest.importorskip("astropy.units")
    Time = pytest.importorskip("astropy.time").Time
    t, m, e, period = _lc(2)
    P._MODELS[("test-net", "cpu", None)] = PeriodsNet(views=(1.0, 0.5, 2.0)).eval()
    plain = nnp.fap(t, m, [period, 2 * period], err=e, model="test-net", device="cpu")
    with_units = nnp.fap(t * u.day, m * u.mag,
                         [period, 2 * period] * u.day, err=e * u.mag, model="test-net", device="cpu")
    np.testing.assert_allclose(plain, with_units, rtol=1e-6)
    as_time = nnp.fap(Time(t, format="mjd"), m, (period * u.day).to(u.hour), err=e, model="test-net", device="cpu")
    np.testing.assert_allclose(plain[0], as_time, rtol=1e-4)
