#!/usr/bin/env python3
"""Select stars from PRIMVS, verify their periods, and pack their Ks light curves into shards.

PRIMVS periods are not reliable on their own, and its best_fap is the 2022
NN_FAP network's output, so nothing here trusts a single catalogue number:

  asassn : PRIMVS stars matched to ASAS-SN; truth = ASAS-SN period. Held out for
           the benchmark and not filtered (our own check is recorded, not applied).
  real   : candidates where PRIMVS's LS and PDM periods agree (|df| < 1/(3T)),
           or that have a period_alt; periods near 1, 1/2, 1/3 and 2 days are
           dropped (sampling aliases). Each is verified with our own Lomb-Scargle
           on the quality-cut Ks light curve: one of our top three peaks must lie
           within 1/(3T) of the candidate and have Baluev FAP < 1e-3. The label
           is our refined peak frequency. Classes are capped so eclipsing
           binaries do not swamp the set. Verification and labels: verify_periods.py.
  quiet  : VIRAC light curves outside PRIMVS_P whose scatter matches their
           errors; hosts for injections and clean negatives.

Splits are by star (sha1 of sourceid). Rules, counts and seeds go to manifest.json.

    python build_dataset.py --out /media/data4/nn_periodogram_data_v2 --workers 7
"""

import argparse
import csv
import hashlib
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PRIMVS = Path("/media/data4/PRIMVS")
LC_ROOT = PRIMVS / "light_curves"
P_MIN, P_MAX = 0.05, 300.0          # days
F_MAX = 20.0                        # highest frequency searched when verifying, 1/day
MIN_POINTS = 40
CLASS_CAP = 60000
ALIASES = (1.0, 0.5, 1 / 3, 2.0)


def lc_path(sourceid):
    s = str(int(sourceid))
    return LC_ROOT / s[:3] / s[3:6] / f"{s}.csv"


def read_ks(path):
    """Ks detections passing quality cuts: (time, mag, err)."""
    t, m, e = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["filter"] != "Ks" or row["detected"] != "1" or not row["ks_mag"]:
                continue
            try:
                err, chi = float(row["ks_err"]), float(row["chi"] or 0)
            except ValueError:
                continue
            if err <= 0 or err > 0.2 or chi > 10:
                continue
            t.append(float(row["mjd"])); m.append(float(row["ks_mag"])); e.append(err)
    return np.array(t), np.array(m), np.array(e)


def split_of(sourceid):
    h = int(hashlib.sha1(str(int(sourceid)).encode()).hexdigest(), 16) % 100
    return "test" if h < 10 else "val" if h < 20 else "train"


def load_real(item):
    """Extract only; verification happens on the GPU in verify_periods.py."""
    sid, cands, info = item
    p = lc_path(sid)
    if not p.is_file():
        return None
    t, m, e = read_ks(p)
    if len(t) < MIN_POINTS:
        return None
    return sid, t, m, e, {**info, "candidate_freqs": [float(c) for c in cands]}


def load_asassn(item):
    sid, period = item
    p = lc_path(sid)
    if not p.is_file():
        return None
    t, m, e = read_ks(p)
    if len(t) < MIN_POINTS:
        return None
    return sid, t, m, e, {"period": period, "candidate_freqs": [1 / period]}


def scan_quiet(args):
    """Quiet light curves from one random leaf folder."""
    folder, exclude_path, seed = args
    exclude = load_exclude(exclude_path)
    rng = random.Random(seed)
    out = []
    try:
        files = list(Path(folder).iterdir())
    except OSError:
        return out
    for f in rng.sample(files, min(40, len(files))):
        try:
            sid = int(f.stem)
        except ValueError:
            continue
        if sid in exclude:
            continue
        t, m, e = read_ks(f)
        if len(t) < MIN_POINTS:
            continue
        if np.mean(((m - np.median(m)) / e) ** 2) < 2.0:
            out.append((sid, t, m, e, {}))
    return out


_EXCLUDE = {}


def load_exclude(path):
    if path not in _EXCLUDE:
        _EXCLUDE[path] = set(np.load(path).tolist())
    return _EXCLUDE[path]


def select(seed):
    import duckdb
    from astropy.io import fits
    web = f"read_parquet('{PRIMVS / 'PRIMVS_P_WEB.parquet'}')"
    with fits.open(PRIMVS / "PRIMVS_P_ASASSN.fits", memmap=True) as h:
        d = h[1].data
        asassn = {int(s): float(p) for s, p in zip(d["sourceid"], d["period"]) if np.isfinite(p) and P_MIN <= p <= P_MAX}
    rows = duckdb.sql(f"""
        SELECT sourceid, any_value(ls_p), any_value(pdm_p), any_value(period_alt), any_value(time_range),
               any_value(variable_type)
        FROM {web}
        WHERE ks_n_detections >= {MIN_POINTS} AND time_range > 0 AND true_amplitude > 3 * magerr_avg
        GROUP BY sourceid""").fetchall()
    all_ids = duckdb.sql(f"SELECT DISTINCT sourceid FROM {web}").fetchnumpy()["sourceid"]
    by_class = {}
    for sid, ls, pdm, alt, T, vtype in rows:
        if sid in asassn:
            continue
        cands = []
        ok = lambda p: p is not None and P_MIN <= p <= P_MAX and all(abs(p / a - 1) > 0.02 for a in ALIASES)
        if ok(ls) and ok(pdm) and abs(1 / ls - 1 / pdm) < 1 / (3 * T):
            cands.append(1 / ls)
        if ok(alt):
            cands.append(1 / alt)
        if cands:
            by_class.setdefault(vtype or "", []).append((int(sid), cands, {"type": vtype or "", "span": float(T)}))
    rng = random.Random(seed)
    real = []
    for v, items in by_class.items():
        rng.shuffle(items)
        real += items[:CLASS_CAP]
    rng.shuffle(real)
    counts = {v: min(len(i), CLASS_CAP) for v, i in by_class.items()}
    return asassn, real, all_ids, counts


def pack(out, name, results):
    T, M, E, off, meta = [], [], [], [0], []
    for r in results:
        if r is None:
            continue
        sid, t, m, e, info = r
        T.append(t); M.append(m); E.append(e); off.append(off[-1] + len(t))
        meta.append({"sourceid": int(sid), "split": split_of(sid), **info})
    np.savez(out / f"{name}.npz", time=np.concatenate(T), mag=np.concatenate(M).astype(np.float32),
             err=np.concatenate(E).astype(np.float32), offsets=np.array(off))
    json.dump(meta, open(out / f"{name}_meta.json", "w"))
    return len(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-quiet", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    asassn, real, all_ids, class_counts = select(a.seed)
    print(f"ASAS-SN: {len(asassn)}; real candidates: {len(real)} {class_counts}", flush=True)
    excl = out / "exclude_ids.npy"
    np.save(excl, np.unique(np.concatenate([all_ids, np.array(list(asassn), dtype=all_ids.dtype)])))

    counts = {}
    with Pool(a.workers) as pool:
        if not (out / "asassn_meta.json").exists():
            counts["asassn"] = pack(out, "asassn", pool.imap(load_asassn, asassn.items(), chunksize=64))
            print(f"packed asassn: {counts['asassn']}", flush=True)
        if not (out / "real_meta.json").exists():
            res = []
            for k, r in enumerate(pool.imap_unordered(load_real, real, chunksize=64)):
                res.append(r)
                if k % 50000 == 0:
                    print(f"  real {k}/{len(real)}", flush=True)
            counts["real"] = pack(out, "real", res)
            print(f"packed real candidates: {counts['real']} of {len(real)}", flush=True)
        if not (out / "quiet_meta.json").exists():
            rng = random.Random(a.seed)
            buckets = [p for p in LC_ROOT.iterdir() if p.is_dir()]
            leaves = [q for b in buckets for q in b.iterdir() if q.is_dir()]
            rng.shuffle(leaves)
            quiet, seen = [], set()
            jobs = ((str(f), str(excl), a.seed + i) for i, f in enumerate(leaves))
            for batch in pool.imap_unordered(scan_quiet, jobs, chunksize=4):
                for r in batch:
                    if r[0] not in seen:
                        seen.add(r[0]); quiet.append(r)
                if len(quiet) >= a.n_quiet:
                    pool.terminate(); break
            counts["quiet"] = pack(out, "quiet", quiet[:a.n_quiet])
            print(f"packed quiet: {counts['quiet']}", flush=True)

    json.dump({"seed": a.seed, "period_range_days": [P_MIN, P_MAX], "min_points": MIN_POINTS,
               "ks_cuts": "filter=Ks, detected=1, 0<ks_err<=0.2, chi<=10",
               "real_rule": ("candidates: PRIMVS ls_p and pdm_p agree within 1/(3T), or period_alt; "
                             "periods within 2% of 1, 1/2, 1/3, 2 d dropped; amplitude > 3 x mean error; "
                             f"class cap {CLASS_CAP}; verified by verify_periods.py (see its docstring)"),
               "test_rule": "ASAS-SN matched, truth = ASAS-SN period, not filtered",
               "quiet_rule": "not in PRIMVS_P or ASAS-SN set, mean ((m - median)/err)^2 < 2",
               "split_rule": "sha1(sourceid) % 100: <10 test, <20 val, else train",
               "class_candidates": class_counts, "counts": counts},
              open(out / "manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
