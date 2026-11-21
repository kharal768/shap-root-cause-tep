"""MRR against channel-level AUROC, average precision and recall at k, and agreement of method orderings.

Usage: python metric_robustness.py <root>. Output: ROOT/out/metric_check.json. Paper: Supplementary Section S9."""
import sys, io, contextlib, json, os
import numpy as np
from scipy.stats import spearmanr
ddir = sys.argv[1]
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ["aggregate_all.py", ddir]
    exec(open("aggregate_all.py").read())

M = 52
out = {}
for m in sorted({m for s in src for m in src[s]}):
    rr, au, ap, h1 = [], [], [], []
    for s in src:
        if m not in src[s]: continue
        e = full(src[s][m])
        a_rr, a_au, a_ap = [], [], []
        for f, v in e.items():
            rk = np.array(v["rank"]); pos = sorted(rk[[N.index(c) for c in GT[f]]]); g = len(pos)
            a_rr.append(1 / pos[0])
            # AUROC: fraction of (pos, neg) pairs with pos ranked higher (smaller rank)
            a_au.append(np.mean([(M - p) - sum(q > p for q in pos) for p in pos]) / (M - g))
            a_ap.append(np.mean([(i + 1) / p for i, p in enumerate(pos)]))
        rr.append(np.mean(a_rr)); au.append(np.mean(a_au)); ap.append(np.mean(a_ap))
    out[m] = {"mrr": float(np.mean(rr)), "auroc": float(np.mean(au)), "ap": float(np.mean(ap))}
main = [m for m in out if not any(k in m for k in ("(test)", "(same", "80%", "95%", "lag1", "lag3", "a.05", "BIC"))]
r_au = spearmanr([out[m]["mrr"] for m in main], [out[m]["auroc"] for m in main]).correlation
r_ap = spearmanr([out[m]["mrr"] for m in main], [out[m]["ap"] for m in main]).correlation
for m in main: print(f"{m:26s} MRR {out[m]['mrr']:.3f}  AUROC {out[m]['auroc']:.3f}  AP {out[m]['ap']:.3f}")
print("n rankings", len(main), "Spearman MRR-AUROC %.2f  MRR-AP %.2f" % (r_au, r_ap))
json.dump({"per_method": out, "spearman_mrr_auroc": r_au, "spearman_mrr_ap": r_ap, "methods": main},
          open(os.path.join(ddir, "out/metric_check.json"), "w"), indent=1)
