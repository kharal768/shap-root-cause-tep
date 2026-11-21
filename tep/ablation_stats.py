"""Each ablation variant against the full pipeline on matched faults: mean difference and fault-level
bootstrap interval.

Usage: python ablation_stats.py <root>. Output: ROOT/out/ablstats.json. Paper: Table 5."""
import sys, io, contextlib, json, os
import numpy as np
ddir = sys.argv[1]
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ["aggregate_all.py", ddir]
    exec(open("aggregate_all.py").read())
rng = np.random.default_rng(777); B = 10000
FI = {f: i for i, f in enumerate(F)}
COUNTS = rng.multinomial(len(F), [1 / len(F)] * len(F), size=B).astype(float)


def mat(m, j):
    seeds = sorted(s for s in src if m in src[s]); V = np.full((len(seeds), len(F)), np.nan)
    for si, s in enumerate(seeds):
        e = full(src[s][m])
        for f in e:
            r = min(e[f]["rank"][N.index(c)] for c in GT[f]); V[si, FI[f]] = (r <= 1, r <= 5, 1 / r)[j]
    return seeds, V


def seedmean(V, C):
    W = ~np.isnan(V); Vz = np.nan_to_num(V)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = (C @ Vz.T) / (C @ W.T.astype(float))
    return np.nanmean(per, axis=1)


def compare(a, b):
    out = {}
    for j, k in enumerate(("hit1", "hit5", "mrr")):
        sa, Va = mat(a, j); sb, Vb = mat(b, j); common = sorted(set(sa) & set(sb))
        Va = Va[[sa.index(s) for s in common]]; Vb = Vb[[sb.index(s) for s in common]]
        D = Va - Vb
        point = float(seedmean(D, np.ones((1, len(F))))[0]); bs = seedmean(D, COUNTS)
        per_seed = np.nanmean(D, axis=1)
        out[k] = {"mean": point, "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                  "better": int((per_seed > 1e-12).sum()), "worse": int((per_seed < -1e-12).sum()), "splits": len(common)}
    return out


res = {"levels": {}, "pairs": {}}
for m in R:
    if " abl: " in m or m in ("MLP SHAP", "MLP-random SHAP", "Lagged MLP SHAP", "GRU SHAP", "CNN SHAP", "XGB Tree SHAP", "XGB perm SHAP", "Mean shift (same samples)"):
        rows = [metrics(b) for b in R[m].values()]
        res["levels"][m] = {k: float(np.mean([r[k] for r in rows])) for k in ("hit1", "hit5", "mrr")}
pairs = [("MLP abl: " + v, "MLP abl: full") for v in ("bg_normal", "out_normal", "out_other", "all_samples", "signed", "deviation")]
pairs += [("XGB abl: " + v, "XGB abl: full") for v in ("out_normal", "out_other", "all_samples", "signed", "deviation")]
pairs += [("MLP-random SHAP", "MLP SHAP"), ("Lagged MLP SHAP", "MLP SHAP"), ("GRU SHAP", "MLP SHAP"), ("CNN SHAP", "MLP SHAP"),
          ("XGB perm SHAP", "XGB Tree SHAP"), ("MLP abl: deviation", "MLP-random SHAP")]
for a, b in pairs:
    if a in R and b in R: res["pairs"][f"{a} vs {b}"] = compare(a, b)
json.dump(res, open(os.path.join(ddir, "out/ablstats.json"), "w"), indent=1)
for m, v in sorted(res["levels"].items()):
    print(f"{m:28s} H1 {v['hit1']*100:5.1f} H5 {v['hit5']*100:5.1f} MRR {v['mrr']:.3f}")
for k, v in res["pairs"].items():
    print(f"{k:45s} " + "  ".join(f"{m} {v[m]['mean']*(100 if m!='mrr' else 1):+.2f} [{v[m]['ci'][0]*(100 if m!='mrr' else 1):+.2f},{v[m]['ci'][1]*(100 if m!='mrr' else 1):+.2f}] {v[m]['better']}/{v[m]['worse']}" for m in ("hit1", "hit5", "mrr")))
