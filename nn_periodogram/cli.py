"""Command line: find (or check) periods in light-curve files.

    nn-periodogram star.csv                      # one file
    nn-periodogram lcs/*.csv --out periods.csv   # many files -> one table
    nn-periodogram star.fits --check 0.5077      # is this period right, or 2P, P/2, an alias?
    nn-periodogram star.dat --time-col 0 --mag-col 1 --err-col 2 --plot star.png

Columns are found by name (time/mjd/jd/hjd/bjd/t, mag/flux/y, err/magerr/fluxerr/sigma) or
given explicitly by name or index. Reads CSV, whitespace-separated text and FITS tables
(FITS needs astropy).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

TIME_NAMES = ("time", "mjd", "jd", "hjd", "bjd", "btjd", "bkjd", "t", "date")
MAG_NAMES = ("mag", "magnitude", "flux", "pdcsap_flux", "sap_flux", "y", "value")
ERR_NAMES = ("err", "error", "magerr", "mag_err", "e_mag", "fluxerr", "flux_err", "pdcsap_flux_err",
             "sap_flux_err", "sigma", "dy", "e")


def _pick(names, wanted, explicit, what):
    if explicit is not None:
        if str(explicit).isdigit():
            return int(explicit)
        low = [n.lower() for n in names]
        if explicit.lower() in low:
            return low.index(explicit.lower())
        raise SystemExit(f"no column {explicit!r} for {what}; columns: {', '.join(names)}")
    low = [n.lower() for n in names]
    for w in wanted:
        if w in low:
            return low.index(w)
    return None


def read_lightcurve(path, time_col=None, mag_col=None, err_col=None):
    """(time, mag, err or None) from a CSV / text / FITS file."""
    path = Path(path)
    if path.suffix.lower() in (".fits", ".fit", ".fz"):
        from astropy.table import Table
        tab = Table.read(path)
        names = tab.colnames
        cols = [np.asarray(tab[n], float) if tab[n].dtype.kind in "fiu" else None for n in names]
    else:
        text = path.read_text().splitlines()
        rows = [ln for ln in text if ln.strip() and not ln.lstrip().startswith("#")]
        delim = "," if "," in rows[0] else None
        first = rows[0].split(delim)
        try:
            [float(x) for x in first]
            names = [str(i) for i in range(len(first))]
            body = rows
        except ValueError:
            names = [x.strip().strip('"') for x in first]
            body = rows[1:]
        data = [r.split(delim) if delim is None else next(csv.reader([r])) for r in body]
        cols = []
        for j in range(len(names)):
            try:
                cols.append(np.array([float(r[j]) if j < len(r) and r[j].strip() else np.nan for r in data]))
            except ValueError:
                cols.append(None)
    numeric = [n for n, c in zip(names, cols) if c is not None]
    it = _pick(names, TIME_NAMES, time_col, "time")
    im = _pick(names, MAG_NAMES, mag_col, "magnitude/flux")
    ie = _pick(names, ERR_NAMES, err_col, "errors")
    if it is None or im is None:                       # fall back to the first unused numeric columns
        idx = [j for j, c in enumerate(cols) if c is not None]
        if len(idx) < 2:
            raise SystemExit(f"{path}: need at least two numeric columns (found {numeric})")
        free = [j for j in idx if j not in (it, im, ie)]
        if it is None:
            it = free.pop(0)
        if im is None:
            im = free.pop(0)
        if ie is None and err_col is None and free:
            ie = free.pop(0)
    t, m = cols[it], cols[im]
    e = cols[ie] if ie is not None else None
    ok = np.isfinite(t) & np.isfinite(m) & (np.isfinite(e) & (e > 0) if e is not None else True)
    return t[ok], m[ok], (e[ok] if e is not None else None)


def main(argv=None):
    import nn_periodogram as nnp
    ap = argparse.ArgumentParser(prog="nn-periodogram", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("files", nargs="+", help="light-curve files (CSV, text or FITS)")
    ap.add_argument("--time-col"); ap.add_argument("--mag-col"); ap.add_argument("--err-col")
    ap.add_argument("--check", type=float, help="check this period (days) instead of searching")
    ap.add_argument("--fmax", type=float, default=10.0, help="highest frequency searched, 1/day (default 10)")
    ap.add_argument("--fmin", type=float, help="lowest frequency, 1/day (default 2/baseline)")
    ap.add_argument("--method", choices=["hybrid", "full"], default="hybrid")
    ap.add_argument("--device", help="cpu, cuda or mps (default: best available)")
    ap.add_argument("--model", default=None, help="model name (default: the package default)")
    ap.add_argument("--weights", help="local weights file(s), comma-separated for an ensemble")
    ap.add_argument("--peaks", type=int, default=3, help="runners-up to print")
    ap.add_argument("--out", help="write a CSV table of results")
    ap.add_argument("--plot", help="save a plot (single file only): fold at the best period")
    a = ap.parse_args(argv)
    kw = dict(device=a.device, weights=a.weights)
    if a.model:
        kw["model"] = a.model
    results = []
    for f in a.files:
        try:
            t, m, e = read_lightcurve(f, a.time_col, a.mag_col, a.err_col)
            if a.check:
                c = nnp.check_period(t, m, a.check, err=e, **kw)
                row = {"file": f, "n": len(t), "period": c.best_period, "fap": c.best_fap, "note": c.relation}
                print(f"{f}: {c.period:.6g} d -> {c.relation} = {c.best_period:.8g} d (fold FAP {c.best_fap:.3g})")
            else:
                p = nnp.periodogram(t, m, err=e, fmin=a.fmin, fmax=a.fmax, method=a.method, **kw)
                row = {"file": f, "n": len(t), "period": p.best_period, "fap": p.best_fap, "note": p.note}
                rest = ", ".join(f"{q:.6g} d" for q, _ in p.peaks[1:a.peaks + 1])
                print(f"{f}: P = {p.best_period:.8g} d   FAP {p.best_fap:.3g}   ({len(t)} points)"
                      + (f"   next: {rest}" if rest else ""))
            results.append(row)
        except SystemExit as exc:
            print(exc, file=sys.stderr); results.append({"file": f, "note": str(exc)})
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "n", "period", "fap", "note"])
            w.writeheader(); w.writerows(results)
    if a.plot and len(a.files) == 1 and results and "period" in results[0]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        P = results[0]["period"]
        ph = np.mod(t - t.min(), P) / P
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(np.concatenate([ph, ph + 1]), np.concatenate([m, m]), fmt=".", ms=3, alpha=0.6,
                    yerr=None if e is None else np.concatenate([e, e]), elinewidth=0.5)
        ax.invert_yaxis(); ax.set_xlabel("phase"); ax.set_ylabel("magnitude")
        ax.set_title(f"{Path(a.files[0]).name}: P = {P:.8g} d, FAP {results[0]['fap']:.2g}")
        fig.tight_layout(); fig.savefig(a.plot, dpi=130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
