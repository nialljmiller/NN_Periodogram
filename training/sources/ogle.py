#!/usr/bin/env python3
"""OGLE-IV Collection of Variable Stars -> shards (I band).

For each region/class folder with a phot.tar.gz (searched two levels deep),
reads every period table (*.dat with OGLE- IDs; the period is the first
column after the I and V mean magnitudes) and the I-band light curves from
the tarball, streaming it so nothing large is unpacked. Stars per class are
capped so no class dominates. Raw tarballs are kept under --raw.

    python ogle.py --raw RAW/ogle --out SHARDS
"""

import argparse
import random
import re
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from common import ShardWriter

FTP = "ftp://ftp.astrouw.edu.pl/ogle/ogle4/OCVS"
REGIONS = ("blg", "lmc", "smc", "gd")
SKIP = ("OLD", "Misclass", "fcharts", "phot", "tmp")


def listing(url):
    out = subprocess.run(["curl", "-s", "--max-time", "60", url + "/"], capture_output=True, text=True).stdout
    return [line.split()[-1] for line in out.splitlines() if line.startswith("d")], \
           [(line.split()[-1], int(line.split()[4])) for line in out.splitlines() if line.startswith("-")]


def find_classes():
    """(remote folder, label) for every folder holding a phot.tar.gz, up to two levels deep."""
    found = []
    for r in REGIONS:
        dirs, _ = listing(f"{FTP}/{r}")
        for d in dirs:
            if any(s in d for s in SKIP):
                continue
            sub, files = listing(f"{FTP}/{r}/{d}")
            if any(f == "phot.tar.gz" for f, _ in files):
                found.append((f"{r}/{d}", f"{r}/{d}"))
            for s in sub:
                if any(x in s for x in SKIP):
                    continue
                _, f2 = listing(f"{FTP}/{r}/{d}/{s}")
                if any(f == "phot.tar.gz" for f, _ in f2):
                    found.append((f"{r}/{d}/{s}", f"{r}/{d}/{s}"))
    return found


def periods(folder, raw):
    """{OGLE ID: period} from every period table in the folder."""
    _, files = listing(f"{FTP}/{folder}")
    out = {}
    for name, size in files:
        if not name.endswith(".dat") or name in ("ident.dat", "remarks.dat") or size > 200_000_000:
            continue
        local = raw / folder / name
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["curl", "-s", "--max-time", "600", "-o", str(local), f"{FTP}/{folder}/{name}"])
        for line in open(local, errors="ignore"):
            tok = line.split()
            if len(tok) >= 4 and tok[0].startswith("OGLE-"):
                for v in tok[3:5]:              # usually column 4; fall back to 5 if 4 is a placeholder
                    try:
                        p = float(v)
                    except ValueError:
                        continue
                    if 0.01 < p < 5000:
                        out.setdefault(tok[0], p)
                        break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=40000, help="max stars per class folder")
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    raw = Path(a.raw)
    for folder, label in find_classes():
        name = "ogle_" + re.sub(r"[^a-z0-9]+", "_", folder.lower()).strip("_")
        if (Path(a.out) / f"{name}_meta.json").exists():
            continue
        per = periods(folder, raw)
        if not per:
            print(f"{folder}: no periods parsed, skipped", flush=True); continue
        tgz = raw / folder / "phot.tar.gz"
        if not tgz.exists():
            tgz.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["curl", "-s", "--max-time", "7200", "-o", str(tgz), f"{FTP}/{folder}/phot.tar.gz"])
        keep = set(random.Random(a.seed).sample(sorted(per), min(a.cap, len(per))))
        w = ShardWriter(a.out, name)
        with tarfile.open(tgz, "r|gz") as tar:          # stream: no unpacking to disk
            for member in tar:
                if not member.isfile() or "/I/" not in f"/{member.name}":
                    continue
                oid = Path(member.name).stem
                if oid not in keep:
                    continue
                d = np.loadtxt(tar.extractfile(member), ndmin=2)
                if d.shape[1] >= 3:
                    w.add(oid, d[:, 0], d[:, 1], d[:, 2], "OGLE-IV", "I", [per[oid]], cls=label)
        print(f"{folder}: {w.close()} stars ({len(per)} with periods)", flush=True)


if __name__ == "__main__":
    main()
