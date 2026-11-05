#!/usr/bin/env python3
"""ZTF light curves of known periodic variables -> shards.

Candidate periods come from the ASAS-SN variable-star catalogue (local CSV,
stars north of dec -28 that ZTF covers; up to --per-class per variable type),
or from Chen et al. 2020 (VizieR J/ApJS/249/18) with --source chen;
light curves from the IRSA ZTF light-curve API (3 arcsec cone, catflags
clean). Each band (g, r) is packed as its own light curve, keyed by ZTF oid.

    python ztf.py --out /media/nill/Musak/nn_periodogram/shards --per-class 5000 --workers 8
"""

import argparse
import io
from pathlib import Path
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import ShardWriter, retry, vizier

API = ("https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE%20{ra}%20{dec}%200.0008"
       "&BAD_CATFLAGS_MASK=32768&FORMAT=csv")


def catalogue(per_class, seed, source="asassn", catalogues="."):
    import pandas as pd
    if source == "asassn":
        t = pd.read_csv(Path(catalogues) / "asassn_catalog_full.csv",
                        usecols=["asassn_name", "raj2000", "dej2000", "period", "variable_type", "mean_vmag"])
        t = t.rename(columns={"asassn_name": "ID", "raj2000": "RAdeg", "dej2000": "DEdeg", "period": "Per",
                              "variable_type": "Type"})
        t = t[(t["DEdeg"] > -28) & t["mean_vmag"].between(13, 17)]   # ZTF saturates brighter than ~12.5
    else:
        def get():
            v = vizier(); v.columns = ["ID", "RAdeg", "DEdeg", "Per", "Type"]
            return v.get_catalogs("J/ApJS/249/18/table2")[0].to_pandas()
        t = retry(get)
    t = t[np.isfinite(t["Per"]) & (t["Per"] > 0)]
    t = t.sample(frac=1, random_state=seed)                   # shuffle, then keep up to per_class of each type
    return t[t.groupby("Type").cumcount() < per_class].reset_index(drop=True)


def fetch(row):
    for attempt in range(4):
        try:
            url = API.format(ra=row.RAdeg, dec=row.DEdeg)
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nn_periodogram"}),
                                        timeout=120) as r:
                text = r.read().decode()
            break
        except Exception:
            time.sleep(10 * (attempt + 1))
    else:
        return row, []
    import pandas as pd
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "catflags" not in df:
        return row, []
    df = df[df["catflags"] == 0]
    out = []
    for (oid, band), g in df.groupby(["oid", "filtercode"]):
        if len(g) >= 10:
            out.append((oid, band, g["mjd"].to_numpy(), g["mag"].to_numpy(), g["magerr"].to_numpy()))
    return row, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--per-class", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--source", choices=["asassn", "chen"], default="asassn"); ap.add_argument("--catalogues", default=".")
    a = ap.parse_args()
    cat = catalogue(a.per_class, a.seed, a.source, a.catalogues)
    print(f"ZTF catalogue sample: {len(cat)} stars", cat["Type"].value_counts().to_dict(), flush=True)
    w = ShardWriter(a.out, "ztf")
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        for k, (row, lcs) in enumerate(pool.map(fetch, cat.itertuples())):
            for oid, band, t, m, e in lcs:
                w.add(oid, t, m, e, "ZTF", band.replace("z", ""), [row.Per], cls=row.Type, catalogue_id=row.ID)
            if k % 2000 == 0:
                print(f"{k}/{len(cat)} stars, {len(w)} light curves, {time.time() - t0:.0f}s", flush=True)
    print(f"ztf: {w.close()} light curves", flush=True)


if __name__ == "__main__":
    main()
