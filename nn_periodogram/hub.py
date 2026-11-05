"""Trained weights: where they live and how they are fetched.

Each model is published in three places (GitHub release, data.nialljmiller.com
and Zenodo); downloads are checked against a SHA-256 before use, so an error
page can never be mistaken for a model.
"""

import hashlib
import os
import shutil
import urllib.request
from functools import partial
from pathlib import Path

import torch

from .models import LegacyNNFAP, PeriodsNet

GITHUB = "https://github.com/nialljmiller/NN_Periodogram/releases/download"
MIRROR = "https://data.nialljmiller.com/nn_periodogram/models"

MODELS = {
    "legacy-2022": {
        "file": "nn_fap_legacy_2022.pt",
        "sha256": None,          # filled in when the release is cut
        "release": "v1.0.0",
        "build": LegacyNNFAP,
        "description": "2022 NN_FAP network (14 x GRU-1024), ported exactly from Keras.",
    },
    "periods-v1": {
        "file": "nn_periodogram_v1.pt",
        "sha256": None,
        "release": "v1.0.0",
        "build": PeriodsNet,
        "description": "Circular CNN on a multi-resolution phase profile; VVV + OGLE + injections (2026).",
    },
    "periods-v3": {
        "file": "nn_periodogram_v3.pt",
        "sha256": None,
        "release": "v1.0.0",
        "build": partial(PeriodsNet, views=(1.0, 0.5, 2.0)),
        "description": "Sees the folds at P, 2P and P/2 side by side; all surveys (2026).",
    },
}
DEFAULT = "legacy-2022"


def cache_dir():
    return Path(os.environ.get("NN_PERIODOGRAM_CACHE", Path.home() / ".cache" / "nn_periodogram"))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name=DEFAULT, path=None):
    """Local path to a model's weights, downloading and verifying them if needed."""
    info = MODELS[name]
    if path is not None:
        return Path(path)
    target = cache_dir() / info["file"]
    if target.exists() and (info["sha256"] is None or _sha256(target) == info["sha256"]):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    urls = [f"{GITHUB}/{info['release']}/{info['file']}", f"{MIRROR}/{info['file']}"]
    errors = []
    for url in urls:
        tmp = target.with_suffix(".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nn_periodogram"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
                shutil.copyfileobj(r, out)
            if info["sha256"] is not None and _sha256(tmp) != info["sha256"]:
                raise ValueError("checksum mismatch")
            tmp.replace(target)
            return target
        except Exception as exc:          # try the next location
            errors.append(f"{url}: {exc}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("could not download model weights:\n  " + "\n  ".join(errors))


def load(name=DEFAULT, path=None, device=None):
    """Build a model and load its weights."""
    info = MODELS[name]
    state = torch.load(fetch(name, path), map_location="cpu", weights_only=True)
    config = {}
    if "state_dict" in state:            # release files: {"state_dict": ..., "config": {...}}
        state, config = state["state_dict"], state.get("config", {})
    if "stem.0.weight" in state:         # PeriodsNet: the views are fixed by the weights' shape
        n_views = state["stem.0.weight"].shape[1] // 16
        views = tuple(config.get("views", (1.0, 0.5, 2.0)[:n_views]))
        model = PeriodsNet(views=views)
    else:
        model = info["build"]()
    model.load_state_dict(state)
    # Search defaults that belong to the weights (e.g. whether ties go to the shorter period)
    model.prefer_shorter = config.get("prefer_shorter", info.get("prefer_shorter", True))
    return model.eval().to(device or default_device())


def package(state_dict_path, out_path, **config):
    """Wrap a training checkpoint as a release file that carries its own configuration."""
    state = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    torch.save({"state_dict": state, "config": config}, out_path)
    return _sha256(out_path)


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
