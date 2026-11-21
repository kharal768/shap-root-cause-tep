"""Power of the fault-level paired Hit-at-5 test: per-fault differences between each SHAP ranking and the best
top-five classifier-free ranking are centred, shifted by delta, resampled over faults and tested with the exact
two-sided sign-flip test (alpha 0.05).

Usage: python power_top5.py <root>. Output: ROOT/out/power_top5.json. Paper: Limitations, Supplementary Section S12."""
import sys, io, contextlib, json, itertools, numpy as np
ddir = sys.argv[1]
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ["aggregate_all.py", ddir]; exec(open("aggregate_all.py").read())
def fv(m):
    out = {}
    for s in src:
        if m in src[s]:
            for f, b in ranks(src[s][m]).items(): out.setdefault(f, []).append(b <= 5)
    return {f: np.mean(v) for f, v in out.items()}
def signflip_p(d):
    nz = d[np.abs(d) > 1e-12]
    if len(nz) == 0: return 1.0
    S = np.array(list(itertools.product([-1, 1], repeat=len(nz))))
    return float(np.mean(np.abs((S * nz).mean(1)) >= abs(nz.mean()) - 1e-12))
free = [m for m in ["Mean shift", "Mean/var", "KS", "PCA-RBC", "PCA-RBC PA", "Granger"] if any(m in src[s] for s in src)]
rate = lambda m: np.mean(list(fv(m).values()))
best = max(free, key=rate); B = fv(best)
shap = [m for m in ["MLP SHAP", "Lagged MLP SHAP", "XGB Tree SHAP", "XGB perm SHAP", "CNN SHAP", "GRU SHAP", "MLP cond. SHAP"] if any(m in src[s] for s in src)]
rng = np.random.default_rng(2026); deltas = np.arange(0, 0.41, 0.025); res = {"best": best, "power": {}}
for m in shap:
    A = fv(m); fs = sorted(set(A) & set(B)); d0 = np.array([A[f] - B[f] for f in fs]); d0 = d0 - d0.mean()
    pw = []
    for dl in deltas:
        hits = sum(signflip_p(np.clip(rng.choice(d0, len(d0)) + dl, -1, 1)) < 0.05 for _ in range(400))
        pw.append(hits / 400)
    res["power"][m] = {"n_faults": len(fs), "delta": deltas.tolist(), "power": pw,
                       "mdd80": next((float(x) for x, p in zip(deltas, pw) if p >= 0.8), None)}
    print(m, len(fs), "80% power at", res["power"][m]["mdd80"])
json.dump(res, open(f"{ddir}/out/power_top5.json", "w"), indent=1)
