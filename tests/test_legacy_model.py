"""The PyTorch port reproduces the 2022 Keras network (reference outputs computed with tf_keras)."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from nn_periodogram import hub

DATA = Path(__file__).parent / "data" / "tf_reference_cases.npz"
WEIGHTS = os.environ.get("NN_PERIODOGRAM_LEGACY_WEIGHTS") or hub.cache_dir() / hub.MODELS["legacy-2022"]["file"]


@pytest.mark.skipif(not Path(WEIGHTS).exists(), reason="legacy weights not downloaded")
def test_port_matches_tensorflow():
    ref = np.load(DATA)
    model = hub.load("legacy-2022", path=WEIGHTS, device="cpu")
    with torch.no_grad():
        y = model(torch.from_numpy(ref["x"])).numpy()
    np.testing.assert_allclose(y, ref["y"], atol=1e-5)
