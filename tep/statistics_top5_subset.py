"""Paired Hit-at-5 comparisons with the best top-five classifier-free ranking, on all faults and on the
direct or strongly coupled faults (Table 2).

Usage: python statistics_top5_subset.py <root>. Output: ROOT/out/stats_v44.json. Paper: Sections 4.3 and 4.5."""
import sys, io, contextlib, json, itertools, numpy as np
ddir = sys.argv[1]
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ["aggregate_all.py", ddir]; exec(open("aggregate_all.py").read())
SUB = {1, 4, 5, 6, 7, 11, 12, 14}
def fv(m):
    out = {}
    for s in src:
        if m in src[s]:
            for f, b in ranks(src[s][m]).items(): out.setdefault(f, []).append(b <= 5)
    return {f: np.mean(v) for f, v in out.items()}
def paired(a, b, fs_keep=None):
    A, B = fv(a), fv(b); fs = sorted(set(A) & set(B) & (fs_keep or set(A)))
    d = np.array([A[f] - B[f] for f in fs]); bs = np.random.default_rng(12345).choice(d, (10000, len(d))).mean(1)
    nz = d[np.abs(d) > 1e-12]
    p = float(np.mean(np.abs((np.array(list(itertools.product([-1, 1], repeat=len(nz)))) * nz).mean(1)) >= abs(nz.mean()) - 1e-12)) if len(nz) else 1.0
    return dict(diff=100*d.mean(), lo=100*np.percentile(bs, 2.5), hi=100*np.percentile(bs, 97.5), p=p, nz=len(nz), nf=len(fs))
free = [m for m in ["Mean shift", "Mean/var", "KS", "PCA-RBC", "PCA-RBC PA", "Granger"] if any(m in src[s] for s in src)]
rate = lambda m, fs=None: 100*np.mean([v for f, v in fv(m).items() if fs is None or f in fs])
best = max(free, key=rate); bestS = max(free, key=lambda m: rate(m, SUB))
shap = [m for m in ["MLP SHAP", "Lagged MLP SHAP", "XGB perm SHAP", "XGB Tree SHAP", "CNN SHAP", "GRU SHAP", "MLP cond. SHAP"] if any(m in src[s] for s in src)]
res = {"best_all": best, "best_sub": bestS, "free_rates_all": {m: rate(m) for m in free}, "free_rates_sub": {m: rate(m, SUB) for m in free}, "all": {}, "sub": {}}
for m in shap:
    res["all"][m] = paired(m, best); res["sub"][m] = paired(m, bestS, SUB); res["sub"][m]["rate"] = rate(m, SUB)
print(json.dumps(res, indent=1, default=float))
json.dump(res, open(f"{ddir}/out/stats_v44.json", "w"), indent=1, default=float)
