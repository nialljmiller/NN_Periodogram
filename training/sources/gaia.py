#!/usr/bin/env python3
"""Gaia DR3 variables -> shards (G band epoch photometry).

Candidate periods come from the DR3 specialist tables (vari_rrlyrae pf,
vari_cepheid pf, vari_eclipsing_binary 1/frequency,
vari_long_period_variable 1/frequency). Epoch photometry is fetched through
DataLink in batches of 5000; err = 1.0857 / flux_over_error.

    python gaia.py --out /media/nill/Musak/nn_periodogram/shards --per-class 40000
"""

import argparse
from pathlib import Path

import numpy as np

from common import ShardWriter, retry

QUERIES = {
    "rrlyrae": "SELECT source_id, pf AS period FROM gaiadr3.vari_rrlyrae WHERE pf IS NOT NULL",
    "cepheid": "SELECT source_id, pf AS period FROM gaiadr3.vari_cepheid WHERE pf IS NOT NULL",
    "eclipsing": "SELECT source_id, 1.0/frequency AS period FROM gaiadr3.vari_eclipsing_binary WHERE frequency > 0",
    "lpv": "SELECT source_id, 1.0/frequency AS period FROM gaiadr3.vari_long_period_variable WHERE frequency > 0",
}


def epoch_g(ids):
    """{source_id: (time, mag, err)} for G-band epochs (DATAMODEL_STANDARD layout, one row per transit)."""
    from astroquery.gaia import Gaia
    res = Gaia.load_data(ids=list(ids), data_release="Gaia DR3", retrieval_type="EPOCH_PHOTOMETRY",
                         data_structure="DATAMODEL_STANDARD", format="votable", verbose=False)
    out = {}
    for key, tables in res.items():
        t = tables[0].to_table()
        ok = ~np.asarray(t["rejected_by_photometry"], bool) & ~np.asarray(t["variability_flag_g_reject"], bool)
        time_, mag = np.asarray(t["g_transit_time"], float)[ok], np.asarray(t["g_transit_mag"], float)[ok]
        foe = np.asarray(t["g_transit_flux_over_error"], float)[ok]
        if len(time_):
            out[int(key.rsplit("_", 1)[1].split(".")[0])] = (time_, mag, 1.0857 / foe)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--per-class", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    from astroquery.gaia import Gaia
    for cls, q in QUERIES.items():
        name = f"gaia_{cls}"
        w = ShardWriter(a.out, name)
        if (Path(a.out) / f"{name}_meta.json").exists():
            continue
        # the vari_* tables have no random_index, so fetch the (small) period list and sample here
        rows = retry(lambda: Gaia.launch_job_async(q, dump_to_file=False).get_results())
        per = {int(s): float(p) for s, p in zip(rows["source_id"], rows["period"])}
        ids = sorted(per)
        ids = [ids[i] for i in np.random.default_rng(a.seed).permutation(len(ids))[: a.per_class]]
        for s in range(0, len(ids), 5000):
            try:
                lcs = retry(lambda: epoch_g(ids[s:s + 5000]))
            except Exception:
                continue
            for sid, (t, m, e) in lcs.items():
                w.add(sid, t, m, e, "Gaia DR3", "G", [per[sid]], cls=cls)
            print(f"{cls}: {s + 5000}/{len(ids)} queried, {len(w)} kept", flush=True)
        print(f"{cls}: {w.close()} stars", flush=True)


if __name__ == "__main__":
    main()
