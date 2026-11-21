"""Sensitivity to the cut-off k = 1..10 and the minimum-sample threshold, from stored rankings.

Usage: python sensitivity_evaluation.py <root>. Output: ROOT/out/sens_eval.json. Paper: Fig. 3, Section 4.6."""
import sys, io, contextlib, json, os
import numpy as np
ddir = sys.argv[1]
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ["aggregate_all.py", ddir]
    exec(open("aggregate_all.py").read())
MAIN = [m for m in ("MLP SHAP", "Lagged MLP SHAP", "CNN SHAP", "GRU SHAP", "XGB Tree SHAP", "XGB perm SHAP",
                    "Mean shift", "Mean/var", "KS", "PCA-RBC") if m in R]
SHAPM = [m for m in MAIN if "SHAP" in m]; BASE = [m for m in MAIN if "SHAP" not in m]
out = {"k": {}, "threshold": {}}
for k in range(1, 11):
    row = {}
    for m in MAIN:
        hs, sig, chs = [], 0, []
        for s, br in R[m].items():
            fs = list(br); h = sum(br[f] <= k for f in fs)
            hs.append(h / len(fs)); chs.append(np.mean([ch(len(GT[f]), k) for f in fs]))
            sig += pb([ch(len(GT[f]), k) for f in fs], h) < 0.05
        row[m] = {"hit": float(np.mean(hs)), "chance": float(np.mean(chs)), "sig": f"{sig}/{len(hs)}"}
    row["best_shap"] = max(SHAPM, key=lambda m: row[m]["hit"]); row["best_base"] = max(BASE, key=lambda m: row[m]["hit"])
    out["k"][k] = row
for t in (5, 10, 20, 30):
    row = {}
    for m in SHAPM:
        h1, h5, nf = [], [], []
        for s in src:
            if m not in src[s]: continue
            e = full(src[s][m]); fs = [f for f in e if e[f].get("n", 30) >= t]
            if not fs: continue
            br = {f: min(e[f]["rank"][N.index(c)] for c in GT[f]) for f in fs}
            h1.append(np.mean([br[f] <= 1 for f in fs])); h5.append(np.mean([br[f] <= 5 for f in fs])); nf.append(len(fs))
        row[m] = {"hit1": float(np.mean(h1)), "hit5": float(np.mean(h5)), "n_faults": float(np.mean(nf))}
    out["threshold"][t] = row
json.dump(out, open(os.path.join(ddir, "out/sens_eval.json"), "w"), indent=1)
print("k  " + " ".join(f"{m[:10]:>10s}" for m in MAIN) + "  chance  best SHAP / best base")
for k, r in out["k"].items():
    print(f"{k:2d} " + " ".join(f"{r[m]['hit']*100:10.1f}" for m in MAIN) + f"  {r[MAIN[0]]['chance']*100:5.1f}  {r['best_shap']} {r[r['best_shap']]['hit']*100:.1f} / {r['best_base']} {r[r['best_base']]['hit']*100:.1f}  sig " + ",".join(r[m]['sig'] for m in SHAPM))
for t, r in out["threshold"].items():
    print(t, {m: (round(v["hit1"] * 100, 1), round(v["hit5"] * 100, 1), round(v["n_faults"], 1)) for m, v in r.items()})
