"""Fault-level comparisons for conditional SHAP, PCA with parallel analysis and the sequence-model
deviation ranking.

Usage: python statistics_extra_rankings.py <root>. Output: ROOT/out/stats_v41.json. Paper: Sections 4.3-4.4."""
import sys, io, contextlib, json, itertools, numpy as np
ddir = sys.argv[1]
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ["aggregate_all.py", ddir]; exec(open("aggregate_all.py").read())
def fv(m, k):
    out = {}
    for s in src:
        if m not in src[s]: continue
        for f, b in ranks(src[s][m]).items(): out.setdefault(f, []).append({"hit1": b <= 1, "hit5": b <= 5, "mrr": 1 / b}[k])
    return {f: np.mean(v) for f, v in out.items()}, out
def split_rate(m, k):
    v = [np.mean([{"hit1": b <= 1, "hit5": b <= 5, "mrr": 1 / b}[k] for b in ranks(src[s][m]).values()]) for s in src if m in src[s]]
    return np.mean(v), len(v)
def paired(a, b, k):
    A, _ = fv(a, k); B, _ = fv(b, k); fs = sorted(set(A) & set(B)); d = np.array([A[f] - B[f] for f in fs])
    bs = np.random.default_rng(12345).choice(d, (10000, len(d))).mean(1); nz = d[np.abs(d) > 1e-12]
    p = float(np.mean(np.abs((np.array(list(itertools.product([-1, 1], repeat=len(nz)))) * nz).mean(1)) >= abs(nz.mean()) - 1e-12)) if len(nz) else 1.0
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p, len(nz)
best = "Mean shift" if ddir.endswith("work") else "Mean/var"
res = {"rates": {}, "pairs": {}}
for m in ["MLP SHAP", "MLP cond. SHAP", "PCA-RBC", "PCA-RBC PA", "CNN SHAP", "GRU SHAP", "CNN abl: deviation", "GRU abl: deviation", "CNN abl: dev_all", "GRU abl: dev_all", best]:
    res["rates"][m] = {k: split_rate(m, k)[0] for k in ("hit1", "hit5", "mrr")}
    print(f"{m:22s}", {k: round(v * (100 if k != 'mrr' else 1), 1 if k != 'mrr' else 3) for k, v in res["rates"][m].items()})
for a, b in [("MLP cond. SHAP", best), ("MLP cond. SHAP", "MLP SHAP"), ("CNN abl: deviation", "CNN SHAP"), ("GRU abl: deviation", "GRU SHAP"), ("PCA-RBC PA", best)]:
    res["pairs"][f"{a} vs {b}"] = {k: paired(a, b, k) for k in ("hit1", "hit5", "mrr")}
    print(f"{a} vs {b}:", {k: (round(v[0] * (100 if k != 'mrr' else 1), 1 if k != 'mrr' else 3), [round(x * (100 if k != 'mrr' else 1), 1 if k != 'mrr' else 3) for x in v[1]], round(v[2], 3), v[3]) for k, v in res["pairs"][f"{a} vs {b}"].items()})
# agreement of hit outcomes between sequence SHAP and its deviation ranking
for a in ("CNN", "GRU"):
    ag = n = 0
    for s in src:
        if f"{a} abl: deviation" not in src[s]: continue
        A = ranks(src[s][f"{a} SHAP"]); B = ranks(src[s][f"{a} abl: deviation"])
        for f in set(A) & set(B): ag += (A[f] <= 5) == (B[f] <= 5); n += 1
    res[f"{a}_agree"] = [ag, n]; print(a, "hit agreement", ag, n)
json.dump(res, open(f"{ddir}/out/stats_v41.json", "w"), indent=1)
