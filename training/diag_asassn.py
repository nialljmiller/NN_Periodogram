"""Why do classical methods miss ASAS-SN periods? Inspect test-split failures."""
import json, numpy as np, torch, sys
sys.path.insert(0, ".")
from benchmark_classical import search
from benchmark import verdict
D = "/media/nill/Musak/nn_periodogram/shards"
d = dict(np.load(f"{D}/asassn_vsx.npz")); meta = json.load(open(f"{D}/asassn_vsx_meta.json")); off = d["offsets"]
idx = [i for i, r in enumerate(meta) if r["split"] == "test"][:400]
dev = torch.device("cuda"); n = 0
for i in idx:
    t, m, e = d["time"][off[i]:off[i+1]], d["mag"][off[i]:off[i+1]].astype(float), d["err"][off[i]:off[i+1]].astype(float)
    pt = 1 / meta[i]["candidate_freqs"][0]
    if pt > np.ptp(t) / 3 or 1 / pt > 10: continue
    p = search(t, m, e, "pdm", 10.0, dev); v = verdict(p, pt)
    if v == "wrong" and n < 25:
        n += 1
        print(f"{meta[i]['sourceid']:>16} {meta[i]['band']} {meta[i]['cls']:<6} n={len(t):5d} span={np.ptp(t):6.0f}d std={np.std(m):.3f} med_err={np.median(e):.3f}  P_true={pt:10.4f}  P_found={p:10.4f}  ratio={p/pt:8.3f}")
