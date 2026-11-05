#!/usr/bin/env python3
"""Kepler and TESS periodic variables -> shards, via MAST (lightkurve).

  kepler_eb  : Kepler eclipsing binaries (Kirk et al. 2016; keplerebs.villanova.edu export)
  tess_eb    : TESS eclipsing binaries (Prsa et al. 2022; tessebs.villanova.edu export)

PDCSAP fluxes, quality-flag cleaned, every quarter/sector stitched (each
normalised to its median) and converted to magnitudes. Downloads are cached
under --cache (not the home directory).

    python kepler_tess.py --out SHARDS --catalogues CATALOGUES --cache /media/nill/Musak/nn_periodogram/raw/mast
"""

import argparse
import os
import re
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import ShardWriter, flux_to_mag

# name: (local catalogue CSV, id column, period column, mission, MAST target format)
SETS = {
    "kepler_eb": ("kepler_eb_villanova.csv", "#KIC", "period", "Kepler", "KIC {}"),   # keplerebs.villanova.edu
    "tess_eb": ("tess_eb_villanova.csv", "tic", "period", "TESS", "TIC {}"),          # tessebs.villanova.edu
    # "kepler_rot": McQuillan et al. 2014 (J/ApJS/211/24) once CDS is reachable again
}


def catalogue(path, idcol, percol, n, seed):
    import pandas as pd
    lines = open(path).read().splitlines()
    head = next(i for i, l in enumerate(lines) if not l.startswith("##"))   # Villanova files open with '##' notes
    t = pd.read_csv(path, skiprows=head)
    t = t[pd.to_numeric(t[percol], errors="coerce") > 0]
    return t.sample(min(n, len(t)), random_state=seed)[[idcol, percol]].values.tolist()


def fetch(target, mission, cache):
    import lightkurve as lk
    warnings.filterwarnings("ignore")
    try:
        sr = lk.search_lightcurve(target, mission=mission, author="Kepler" if mission == "Kepler" else "SPOC",
                                  exptime=1800 if mission == "Kepler" else 120)
        if len(sr) == 0:
            return None
        try:
            lcs = sr.download_all(flux_column="pdcsap_flux", quality_bitmask="default", download_dir=cache)
        except lk.utils.LightkurveError as exc:         # corrupt cached file: delete it and fetch again
            bad = re.search(r"product (\S+\.fits)", str(exc))
            if bad:
                Path(bad.group(1)).unlink(missing_ok=True)
            lcs = sr.download_all(flux_column="pdcsap_flux", quality_bitmask="default", download_dir=cache)
        t, f, e = [], [], []
        for lc in lcs:
            lc = lc.remove_nans()
            med = np.nanmedian(lc.flux.value)
            if not np.isfinite(med) or med <= 0:
                continue
            t.append(lc.time.value); f.append(lc.flux.value / med); e.append(lc.flux_err.value / med)
        if not t:
            return None
        m, me = flux_to_mag(np.concatenate(f), np.concatenate(e))
        return np.concatenate(t), m, me
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--catalogues", required=True)
    ap.add_argument("--n", type=int, default=5000); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--log", required=True, help="progress log (stdout gets closed by the download stack)")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    logf = open(a.log, "a", buffering=1)
    log = lambda msg: logf.write(msg + "\n")
    from astroquery.mast import Observations
    Observations.enable_cloud_dataset = False
    import logging
    logging.getLogger("astroquery").setLevel(logging.ERROR)
    for name in a.sets:
        if (Path(a.out) / f"{name}_meta.json").exists():
            continue
        path, idcol, percol, mission, fmt = SETS[name]
        rows = catalogue(Path(a.catalogues) / path, idcol, percol, a.n, a.seed)
        log(f"{name}: {len(rows)} catalogue stars")
        w = ShardWriter(a.out, name)
        with ThreadPoolExecutor(a.workers) as pool:
            res = pool.map(lambda r: (r, fetch(fmt.format(int(r[0])), mission, a.cache)), rows)
            for k, (row, lc) in enumerate(res):
                if lc is not None:
                    w.add(int(row[0]), *lc, mission, "Kp" if mission == "Kepler" else "T", [float(row[1])], cls=name)
                if k % 250 == 0:
                    log(f"  {k}/{len(rows)}  kept {len(w)}")
        log(f"{name}: {w.close()} stars")


if __name__ == "__main__":
    main()
