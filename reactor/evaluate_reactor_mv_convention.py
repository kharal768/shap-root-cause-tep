"""Reactor results when the loop's manipulated variable is also counted as documented for the two sensor
faults (commanded coolant flow for the temperature bias, outlet flow for the level drift), as for the TEP.

Input: ../out/cstr_seed<k>.json. Output: ../out/cstr_tepconvention.json
(per ranking: Hit-at-1, Hit-at-3, sensor-fault hit rate, largest split-level p at the top three).
Paper: Section 4.7, Supplementary Section S12."""
import json, glob, numpy as np
from math import comb
meta = json.load(open("../data/meta.json")); N = meta["names"]; M = len(N)
GT = {int(k): [N.index(c) for c in v] for k, v in meta["gt"].items()}
GT[10] = [N.index("T"), N.index("Fc_cmd")]; GT[11] = [N.index("h"), N.index("Fout")]
runs = [json.load(open(p)) for p in sorted(glob.glob("../out/cstr_seed*.json"))]
chance = lambda g, k: 1 - comb(M - g, k) / comb(M, k)
def ptail(h, ps):
    d = np.array([1.0])
    for p in ps: d = np.convolve(d, [1 - p, p])
    return d[h:].sum()
res = {}
for m in runs[0]["rank"]:
    H1, H3, P, sens = [], [], [], []
    for r in runs:
        ev = [(int(f), v) for f, v in r["rank"][m].items() if v is not None]
        b = {f: min(v[i] for i in GT[f]) for f, v in ev}
        H1.append(np.mean([x <= 1 for x in b.values()])); h3 = [x <= 3 for x in b.values()]; H3.append(np.mean(h3))
        P.append(ptail(int(sum(h3)), [chance(len(GT[f]), 3) for f in b])); sens += [b[f] <= 3 for f in (10, 11) if f in b]
    res[m] = [float(np.mean(H1)), float(np.mean(H3)), float(np.mean(sens)), float(max(P))]
json.dump(res, open("../out/cstr_tepconvention.json", "w"), indent=1)
