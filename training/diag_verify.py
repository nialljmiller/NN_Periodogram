"""Verification yield on 400 random VVV candidates, with the LS and PDM checks."""
import json, time, numpy as np, torch
import verify_periods as v
D = "/media/nill/Musak/nn_periodogram/shards"
d = dict(np.load(f"{D}/vvv_real.npz")); meta = json.load(open(f"{D}/vvv_real_meta.json")); off = d["offsets"]
dev = torch.device("cuda"); rng = np.random.default_rng(0)
cnt = {"ls": 0, "pdm": 0, "none": 0}; by_type = {}; t0 = time.time()
for i in rng.choice(len(meta), 400, replace=False):
    s, e = off[i], off[i + 1]; t, m, er = d["time"][s:e], d["mag"][s:e].astype(float), d["err"][s:e].astype(float)
    r = v.verify_one(t, m, er, meta[i]["candidate_freqs"], dev)
    if r is not None and r["fap"] < 1e-3: k = "ls"
    else:
        p = v.pdm_check(t, m, meta[i]["candidate_freqs"], dev); k = "pdm" if p and p["p"] < 1 / 2000 else "none"
    cnt[k] += 1; ty = meta[i].get("type", ""); by_type.setdefault(ty, [0, 0]); by_type[ty][0] += k != "none"; by_type[ty][1] += 1
print(cnt, f"{(time.time() - t0) / 400 * 1000:.0f} ms/star")
print({k: f"{a}/{b}" for k, (a, b) in sorted(by_type.items(), key=lambda x: -x[1][1])})
