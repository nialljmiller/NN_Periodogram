"""Shared shard format for every survey.

A shard is NAME.npz (time [days], mag, err, offsets) plus NAME_meta.json, one
row per star: {"sourceid", "survey", "band", "split", "candidate_freqs", ...}.
Flux surveys convert to magnitudes before packing. Periods in the metadata are
catalogue candidates only; verify_periods.py decides the label.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

MIN_POINTS = 10


def split_of(key):
    h = int(hashlib.sha1(str(key).encode()).hexdigest(), 16) % 100
    return "test" if h < 10 else "val" if h < 20 else "train"


def clean(t, m, e):
    t, m, e = (np.asarray(x, float) for x in (t, m, e))
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    o = np.argsort(t)
    return t[o], m[o], e[o]


def flux_to_mag(flux, flux_err):
    flux, flux_err = np.asarray(flux, float), np.asarray(flux_err, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -2.5 * np.log10(flux), 1.0857 * flux_err / flux


class ShardWriter:
    """Collects light curves and writes one shard; call add() then close()."""

    def __init__(self, out_dir, name):
        self.path = Path(out_dir); self.path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.T, self.M, self.E, self.off, self.meta = [], [], [], [0], []

    def add(self, key, t, m, e, survey, band, candidate_periods=(), **info):
        t, m, e = clean(t, m, e)
        if len(t) < MIN_POINTS:
            return False
        freqs = [1.0 / p for p in candidate_periods if p and np.isfinite(p) and p > 0]
        self.T.append(t); self.M.append(m.astype(np.float32)); self.E.append(e.astype(np.float32))
        self.off.append(self.off[-1] + len(t))
        self.meta.append({"sourceid": str(key), "survey": survey, "band": band, "split": split_of(f"{survey}:{key}"),
                          "candidate_freqs": freqs, **info})
        return True

    def __len__(self):
        return len(self.meta)

    def close(self):
        if not self.meta:
            return 0
        np.savez(self.path / f"{self.name}.npz", time=np.concatenate(self.T), mag=np.concatenate(self.M),
                 err=np.concatenate(self.E), offsets=np.array(self.off))
        json.dump(self.meta, open(self.path / f"{self.name}_meta.json", "w"))
        return len(self.meta)


def vizier():
    """astroquery Vizier on the Harvard mirror (Strasbourg often times out), no row limit."""
    from astroquery.vizier import Vizier
    return Vizier(row_limit=-1, vizier_server="vizier.cfa.harvard.edu", timeout=600)


def retry(fn, tries=6, wait=60):
    """Call fn(), retrying with a growing wait on any exception (archives are often busy)."""
    import time
    for k in range(tries):
        try:
            return fn()
        except Exception as exc:
            if k == tries - 1:
                raise
            print(f"  retry {k + 1}/{tries - 1} after: {str(exc)[:120]}", flush=True)
            time.sleep(wait * (k + 1))
