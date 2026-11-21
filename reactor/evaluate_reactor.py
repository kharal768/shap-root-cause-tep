"""Hit rates, MRR, exact chance tests and fault-level paired comparisons for the reactor runs.

Input: ../out/cstr_seed<k>.json. Output: ../out/cstr_summary.json. Paper: Section 4.7, Supplementary Table S10."""
import json, glob, itertools, numpy as np
meta = json.load(open("../data/meta.json")); N = meta["names"]; GT = {int(k): [N.index(c) for c in v] for k, v in meta["gt"].items()}
DIRECT = set(meta["direct"]); M = len(N)
runs = [json.load(open(p)) for p in sorted(glob.glob("../out/cstr_seed*.json"))]
def chance(g, k):  # P(at least one of g documented channels in top k of M) under a uniform ranking
    from math import comb
    return 1 - comb(M - g, k) / comb(M, k)
def pois_binom_p(hits, ps):  # exact upper tail
    dist = np.array([1.0])
    for p in ps: dist = np.convolve(dist, [1 - p, p])
    return dist[hits:].sum()
methods = list(runs[0]["rank"].keys()); res = {}
for m in methods:
    H = {1: [], 3: []}; RR = []; P = {1: [], 3: []}; perf = {}
    for r in runs:
        ev = [(int(f), v) for f, v in r["rank"][m].items() if v is not None]
        for k in (1, 3):
            h = [min(v[i] for i in GT[f]) <= k for f, v in ev]; H[k].append(np.mean(h))
            P[k].append(pois_binom_p(int(sum(h)), [chance(len(GT[f]), k) for f, _ in ev]))
        RR.append(np.mean([1 / min(v[i] for i in GT[f]) for f, v in ev]))
        for f, v in ev: perf.setdefault(f, []).append(min(v[i] for i in GT[f]))
    res[m] = {"hit1": np.mean(H[1]), "hit3": np.mean(H[3]), "mrr": np.mean(RR), "sd3": np.std(H[3], ddof=1),
              "maxp3": max(P[3]), "maxp1": max(P[1]), "perfault3": {f: float(np.mean(np.array(x) <= 3)) for f, x in sorted(perf.items())},
              "direct3": float(np.mean([np.mean(np.array(x) <= 3) for f, x in perf.items() if f in DIRECT])),
              "proxy3": float(np.mean([np.mean(np.array(x) <= 3) for f, x in perf.items() if f not in DIRECT])), "n": len(perf)}
c1 = np.mean([chance(len(GT[f]), 1) for f in GT]); c3 = np.mean([chance(len(GT[f]), 3) for f in GT])
print(f"runs {len(runs)} chance hit1 {c1*100:.1f} hit3 {c3*100:.1f}  acc mlp {np.mean([r['acc_mlp'] for r in runs])*100:.1f} xgb {np.mean([r['acc_xgb'] for r in runs])*100:.1f}")
for m, v in res.items():
    print(f"{m:16s} H1 {v['hit1']*100:5.1f} H3 {v['hit3']*100:5.1f} (sd {v['sd3']*100:4.1f}) MRR {v['mrr']:.3f} maxp1 {v['maxp1']:.4f} maxp3 {v['maxp3']:.4f} direct {v['direct3']*100:5.1f} proxy {v['proxy3']*100:5.1f} nf {v['n']} ", {f: round(x, 1) for f, x in v['perfault3'].items()})
# fault-level paired: each SHAP ranking vs best classifier-free on hit3 and mrr
cf = ["Mean shift", "Mean/var", "KS", "PCA-RBC"]
def fault_vals(m, metric):
    out = {}
    for r in runs:
        for f, v in r["rank"][m].items():
            if v is None: continue
            b = min(v[i] for i in GT[int(f)]); out.setdefault(int(f), []).append((b <= 3) if metric == "hit3" else (b <= 1) if metric == "hit1" else 1 / b)
    return {f: np.mean(x) for f, x in out.items()}
best = max(cf, key=lambda m: res[m]["hit3"]); print("best classifier-free:", best)
pair = {}
for m in ["MLP SHAP", "MLP cond. SHAP", "XGB Tree SHAP", "Deviation", "MLP-random SHAP"]:
    pair[m] = {}
    for metric in ("hit1", "hit3", "mrr"):
        a, b = fault_vals(m, metric), fault_vals(best, metric); fs = sorted(set(a) & set(b)); dlt = np.array([a[f] - b[f] for f in fs])
        bs = np.random.default_rng(1).choice(dlt, (10000, len(dlt))).mean(1); nz = dlt[np.abs(dlt) > 1e-12]
        signs = np.array(list(itertools.product([-1, 1], repeat=len(nz)))) if len(nz) else np.ones((1, 0))
        p = float(np.mean(np.abs((signs * nz).mean(1)) >= abs(nz.mean()) - 1e-12)) if len(nz) else 1.0
        pair[m][metric] = {"mean": float(dlt.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": p, "n_nonzero": int(len(nz))}
    print(f"{m:16s} vs {best}: " + "  ".join(f"{k} {v['mean']*(100 if k!='mrr' else 1):+.2f} [{v['ci'][0]*(100 if k!='mrr' else 1):+.2f},{v['ci'][1]*(100 if k!='mrr' else 1):+.2f}] p={v['p']:.3f} nz={v['n_nonzero']}" for k, v in pair[m].items()))
json.dump({"summary": res, "pairs": pair, "best": best, "chance": [c1, c3]}, open("../out/cstr_summary.json", "w"), indent=1, default=float)
