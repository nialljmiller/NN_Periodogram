"""The command-line tool reads common file layouts and finds the period."""
import importlib

import numpy as np
import pytest
import torch

from nn_periodogram import cli
from nn_periodogram.models import PeriodsNet

P = importlib.import_module("nn_periodogram.periodogram")


def _lc(n=300, period=0.731, seed=4):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 900, n))
    return t, 14 + 0.3 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.03, n), np.full(n, 0.03)


@pytest.mark.parametrize("layout", ["csv_header", "text_noheader", "csv_reordered"])
def test_read_lightcurve(tmp_path, layout):
    t, m, e = _lc()
    if layout == "csv_header":
        f = tmp_path / "a.csv"; np.savetxt(f, np.c_[t, m, e], delimiter=",", header="mjd,mag,magerr", comments="")
    elif layout == "text_noheader":
        f = tmp_path / "a.dat"; np.savetxt(f, np.c_[t, m, e])
    else:
        f = tmp_path / "a.csv"; np.savetxt(f, np.c_[m, t, e], delimiter=",", header="brightness,time,error", comments="")
    tt, mm, ee = cli.read_lightcurve(f)
    np.testing.assert_allclose(tt, t); np.testing.assert_allclose(mm, m); np.testing.assert_allclose(ee, e)


def test_cli_runs(tmp_path, capsys):
    torch.manual_seed(0)
    P._MODELS[("test-net", "cpu", None)] = PeriodsNet(views=(1.0, 0.5, 2.0)).eval()
    t, m, e = _lc()
    f = tmp_path / "a.csv"; np.savetxt(f, np.c_[t, m, e], delimiter=",", header="time,mag,err", comments="")
    out = tmp_path / "res.csv"
    assert cli.main([str(f), "--device", "cpu", "--model", "test-net", "--out", str(out)]) == 0
    assert "P =" in capsys.readouterr().out and out.read_text().count("\n") == 2
