#!/usr/bin/env python3
"""Training labels from verified stars, fixing harmonic labels.

verify_periods.py decides WHICH stars are kept (our own LS or PDM finds the
candidate period, or its double or half). It must not decide the label: for an
eclipsing binary the matching Lomb-Scargle peak is at half the orbital period.

  catalogue shards (OGLE, Kepler, TESS, ASAS-SN/VSX): the label is the catalogue
      period itself, refined by phase dispersion within +-1/T of exactly 1x;
  VVV (candidates came from LS/PDM, no independent truth): only stars verified at
      the candidate itself (ratio 1) are kept, at their verified frequency.

Writes {shard}_labels.json: [{"index": i, "freq": f}], read by train.py.

    python relabel.py --data SHARDS
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn_periodogram.classical import pdm_theta  # noqa: E402
from train import mapped  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for vpath in sorted(Path(a.data).glob("*_verified.json")):
        name = vpath.name[:-len("_verified.json")]
        if a.only and name not in a.only:
            continue
        ver = [v for v in json.load(open(vpath)) if v.get("verified")]
        out, t0 = [], time.time()
        if name.startswith("vvv_real"):
            out = [{"index": v["index"], "freq": v["freq"]} for v in ver if float(v.get("ratio", 1)) == 1.0]
        else:
            meta = json.load(open(Path(a.data) / f"{name}_meta.json"))
            d = mapped(Path(a.data) / name); off = d["offsets"]
            for v in ver:
                i = v["index"]
                t = np.asarray(d["time"][off[i]:off[i + 1]], float); m = np.asarray(d["mag"][off[i]:off[i + 1]], float)
                fc = meta[i]["candidate_freqs"][0]; span = np.ptp(t)
                fine = np.linspace(fc - 1 / span, fc + 1 / span, 201)
                fine = fine[fine > 0]
                out.append({"index": i, "freq": float(fine[int(np.argmin(pdm_theta(t, m, fine, dev)))])})
        json.dump(out, open(Path(a.data) / f"{name}_labels.json", "w"))
        print(f"{name:<18} {len(out):7d} labels  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
