#!/usr/bin/env python3
"""ASAS-SN light curves of VSX periodic variables -> shards (Sky Patrol v2).

Candidate periods and classes come from the VSX catalogue as cross-matched
inside Sky Patrol (catalogue 'aavsovsx'); light curves are pulled by
asas_sn_id. Each band (V, g) is packed separately. Only good-quality
detections are kept, and each camera's median is subtracted within a band
so camera zero-point offsets do not masquerade as variability.

    python asassn.py --out /media/nill/Musak/nn_periodogram/shards --per-class 4000 --threads 8
"""

import argparse
import time

import numpy as np

from common import ShardWriter, retry

P_MIN, P_MAX = 0.05, 300.0


def clean_band(df):
    df = df[(df["quality"] == "G") & (df["mag_err"] > 0) & (df["mag_err"] < 0.2) & np.isfinite(df["mag"])]
    if "limit" in df:
        df = df[df["mag"] < df["limit"]]
    if df.empty:
        return None
    m = df["mag"].to_numpy(float).copy()
    for cam in df["camera"].unique():          # per-camera zero points
        k = (df["camera"] == cam).to_numpy()
        m[k] -= np.median(m[k])
    return df["jd"].to_numpy(float) - 2400000.5, m, df["mag_err"].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--per-class", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    from pyasassn.client import SkyPatrolClient
    c = SkyPatrolClient()
    cat = retry(lambda: c.adql_query(
        f"SELECT asas_sn_id, period, variability_type FROM aavsovsx WHERE period > {P_MIN} AND period < {P_MAX}"))
    cat = cat.dropna(subset=["variability_type"])
    cat["cls"] = cat["variability_type"].str.split("|").str[0].str.rstrip(":")
    cat = cat.sample(frac=1, random_state=a.seed)
    cat = cat[cat.groupby("cls").cumcount() < a.per_class].reset_index(drop=True)
    print(f"VSX periodic in Sky Patrol: {len(cat)} selected;", cat["cls"].value_counts().head(15).to_dict(), flush=True)
    per = dict(zip(cat.asas_sn_id, cat.period)); cls = dict(zip(cat.asas_sn_id, cat.cls))
    w, ids, t0 = ShardWriter(a.out, "asassn_vsx"), cat.asas_sn_id.tolist(), time.time()
    for s in range(0, len(ids), a.batch):
        try:
            lcs = retry(lambda: c.query_list(ids[s:s + a.batch], catalog="aavsovsx", download=True, threads=a.threads))
        except Exception:
            continue
        for sid, g in lcs.data.groupby("asas_sn_id"):
            for band, gb in g.groupby("phot_filter"):
                r = clean_band(gb)
                if r is not None:
                    w.add(f"{sid}:{band}", *r, "ASAS-SN", band, [per[sid]], cls=cls[sid])
        print(f"{s + a.batch}/{len(ids)} stars, {len(w)} light curves, {time.time() - t0:.0f}s", flush=True)
    print(f"asassn: {w.close()} light curves", flush=True)


if __name__ == "__main__":
    main()
