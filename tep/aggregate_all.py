"""Aggregates every ranking and the paired comparisons on matched faults, with the cross-fault control.

Usage: python aggregate_all.py <root>. Output: ROOT/out/agg2.json. Paper: Tables 3 and 4, Supplementary Table S8."""
import json, glob, os, sys, numpy as np
from math import comb
from scipy.stats import wilcoxon
from fault_metadata import ALL_CHANNEL_NAMES as N, GROUND_TRUTH_CHANNELS as GT
F = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]
ddir = sys.argv[1]; O = os.path.join(ddir, "out")


def ld(p):
    return json.load(open(p)) if os.path.exists(p) else None


def ch(g, k): return 1 - comb(52 - g, k) / comb(52, k)


def pb(ps, x):
    d = np.zeros(len(ps) + 1); d[0] = 1
    for p in ps: d[1:] = d[1:] * (1 - p) + d[:-1] * p; d[0] *= 1 - p
    return float(d[x:].sum())


def chance_mrr(g, M=52):
    return sum(comb(M - j, g - 1) / comb(M, g) / j for j in range(1, M - g + 2))


src = {}
for s in range(10):
    r = {}
    sw = ld(f"{O}/sweep_seed{s}.json"); tp = ld(f"{O}/temporal_seed{s}.json")
    if sw is None: continue
    r["MLP SHAP"] = sw["mlp"]["shap"]; r["XGB Tree SHAP"] = sw["xgb"]["shap"]
    if "shap_perm" in sw["xgb"]: r["XGB perm SHAP"] = sw["xgb"]["shap_perm"]
    r["Mean shift"] = sw["baseline"]
    if tp: r["Lagged MLP SHAP"] = tp["mlp_lag"]["shap"]; r["Mean/var"] = tp["baseline_var"]
    for a in ("cnn", "gru"):
        x = ld(f"{O}/{a}_seed{s}.json")
        if x: r[f"{a.upper()} SHAP"] = x[a]["shap"]
    x = ld(f"{O}/pca_seed{s}.json")
    if x: r["PCA-RBC"] = {k: v for k, v in x.items() if k.isdigit()}
    x = ld(f"{O}/base2_classfree_seed{s}.json")
    if x: r["KS"] = x["ks"]; r["Granger"] = x["granger"]
    x = ld(f"{O}/base2_mlprand_seed{s}.json")
    if x: r["MLP-random SHAP"] = x["mlprand"]
    for a in ("cnn", "gru"):
        x = ld(f"{O}/base2_ig_{a}_seed{s}.json")
        if x: r[f"{a.upper()} IG"] = x[f"ig_{a}"]
    x = ld(f"{O}/fair_seed{s}.json")
    if x:
        for k, nm in (("ms_test", "Mean shift (test)"), ("mv_test", "Mean/var (test)"), ("ks_test", "KS (test)"),
                      ("pca_test", "PCA-RBC (test)"), ("pca80", "PCA-RBC 80%"), ("pca95", "PCA-RBC 95%"),
                      ("granger_l1", "Granger lag1"), ("granger_l3", "Granger lag3"), ("granger_a05", "Granger a.05")):
            r[nm] = x[k]
    x = ld(f"{O}/hpo_seed{s}.json")
    if x: r["MLP SHAP tuned"] = x["mlp"]["shap"]
    for a, nm in (("xgb", "XGB Tree SHAP tuned"), ("cnn", "CNN SHAP tuned"), ("gru", "GRU SHAP tuned")):
        x = ld(f"{O}/tune_{a}_seed{s}.json")
        if x: r[nm] = x["shap"]
    for mdl, lab in (("mlp", "MLP"), ("xgb", "XGB")):
        x = ld(f"{O}/ablate_{mdl}_seed{s}.json")
        if x:
            for k, v in x.items():
                if isinstance(v, dict): r[f"{lab} abl: {k}"] = v
    x = ld(f"{O}/matched_seed{s}.json")
    if x:
        for k, nm in (("ms_same", "Mean shift (same samples)"), ("mv_same", "Mean/var (same samples)"),
                      ("ks_same", "KS (same samples)"), ("pca_same", "PCA-RBC (same samples)"), ("granger_bic", "Granger BIC")):
            r[nm] = x[k]
    x = ld(f"{O}/condshap_seed{s}.json")
    if x: r["MLP cond. SHAP"] = x["cond"]
    x = ld(f"{O}/pcapa_seed{s}.json")
    if x: r["PCA-RBC PA"] = {k: v for k, v in x.items() if k.isdigit()}
    for a in ("cnn", "gru"):
        x = ld(f"{O}/seqdev_{a}_seed{s}.json")
        if x: r[f"{a.upper()} abl: deviation"] = x["deviation"]; r[f"{a.upper()} abl: dev_all"] = x["dev_all"]
    src[s] = r


def ranks(e):
    out = {}
    for f in F:
        v = e.get(str(f), e.get(f))
        if v is not None: out[f] = min(v["rank"][N.index(c)] for c in GT[f])
    return out


R = {m: {s: ranks(src[s][m]) for s in src if m in src[s]} for m in sorted({m for s in src for m in src[s]})}


def metrics(br, fs=None):
    fs = [f for f in (fs or br) if f in br]
    h = {k: sum(br[f] <= k for f in fs) for k in (1, 3, 5)}
    return {"n": len(fs), **{f"hit{k}": h[k] / len(fs) for k in h},
            **{f"p{k}": pb([ch(len(GT[f]), k) for f in fs], h[k]) for k in h},
            "mrr": float(np.mean([1 / br[f] for f in fs])),
            "chance_mrr": float(np.mean([chance_mrr(len(GT[f])) for f in fs]))}


def full(e):
    return {f: (e.get(str(f)) or e.get(f)) for f in F if (e.get(str(f)) or e.get(f)) is not None}


S4 = {"XMEAS(4)", "XMV(4)"}
spec = {}
for m in sorted({m for s in src for m in src[s]}):
    fp = tot = 0; ag = []
    for s in src:
        if m not in src[s]: continue
        e = full(src[s][m])
        for f, v in e.items():
            if f in (1, 2, 7, 8, 10): continue
            top = set(np.argsort(v["rank"])[:5]); tot += 1
            fp += any(N.index(c) in top for c in S4)
        # fault-agnostic control: one constant ranking per seed = mean rank over the method's faults
        mr = np.mean([v["rank"] for v in e.values()], axis=0)
        cr = np.empty(52, int); cr[np.argsort(mr)] = np.arange(1, 53)
        br = {f: min(cr[N.index(c)] for c in GT[f]) for f in e}
        ag.append(np.mean([br[f] <= 5 for f in br]))
    # cross-fault null: hit of fault f's documented channels in the ranking made for another fault f'
    # whose documented set is disjoint from G_f (the ranking carries no information about f).
    cross = {1: [], 5: []}; matched = {1: [], 5: []}
    for s in src:
        if m not in src[s]: continue
        e = full(src[s][m])
        for k in (1, 5):
            hm = [min(e[f]["rank"][N.index(c)] for c in GT[f]) <= k for f in e]
            hc = [min(e[g]["rank"][N.index(c)] for c in GT[f]) <= k for f in e for g in e
                  if g != f and not (GT[f] & GT[g])]
            matched[k].append(np.mean(hm)); cross[k].append(np.mean(hc))
    spec[m] = {"s4_in_top5_nonS4": f"{fp}/{tot}", "agnostic_hit5": float(np.mean(ag)),
               "matched_hit1": float(np.mean(matched[1])), "cross_hit1": float(np.mean(cross[1])),
               "matched_hit5": float(np.mean(matched[5])), "cross_hit5": float(np.mean(cross[5]))}

summary = {}
for m, bs in R.items():
    rows = [metrics(b) for b in bs.values()]
    summary[m] = {"seeds": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in ("hit1", "hit3", "hit5", "mrr")},
                  **{f"min_{k}": float(min(r[k] for r in rows)) for k in ("hit1", "hit5")},
                  **{f"max_{k}": float(max(r[k] for r in rows)) for k in ("hit1", "hit5")},
                  "max_p": float(max(max(r["p1"], r["p3"], r["p5"]) for r in rows)),
                  "chance_mrr": float(np.mean([r["chance_mrr"] for r in rows]))}
    summary[m]["perfault_hit5"] = {f: f"{sum(b[f] <= 5 for b in bs.values() if f in b)}/{sum(f in b for b in bs.values())}" for f in F}


def paired(a, b):
    """a vs b on the faults evaluated by both in the same seed; mean difference over seeds and Wilcoxon."""
    seeds = sorted(set(R[a]) & set(R[b])); out = {"seeds": len(seeds)}
    for k in ("hit1", "hit5", "mrr"):
        d = []
        for s in seeds:
            fs = [f for f in R[a][s] if f in R[b][s]]
            d.append(metrics(R[a][s], fs)[k] - metrics(R[b][s], fs)[k])
        d = np.array(d)
        try: p = float(wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
        except ValueError: p = 1.0
        out[k] = {"mean": float(d.mean()), "min": float(d.min()), "max": float(d.max()), "wins": int((d > 0).sum()),
                  "losses": int((d < 0).sum()), "p": p}
    return out


shap_m = [m for m in R if "SHAP" in m and "random" not in m and "tuned" not in m]
base_m = [m for m in ("Mean shift", "Mean/var", "KS", "PCA-RBC", "Granger", "Mean shift (test)", "Mean/var (test)",
                      "KS (test)", "PCA-RBC (test)") if m in R]
pairs = {f"{a} vs {b}": paired(a, b) for a in shap_m for b in base_m}
for a, b in (("MLP SHAP", "MLP-random SHAP"), ("CNN SHAP", "CNN IG"), ("GRU SHAP", "GRU IG"),
             ("MLP SHAP tuned", "MLP SHAP"), ("XGB Tree SHAP tuned", "XGB Tree SHAP"),
             ("CNN SHAP tuned", "CNN SHAP"), ("GRU SHAP tuned", "GRU SHAP"),
             ("MLP SHAP", "Mean shift (same samples)"), ("MLP SHAP", "Mean/var (same samples)"),
             ("MLP SHAP", "KS (same samples)"), ("MLP SHAP", "PCA-RBC (same samples)"),
             ("Mean shift (same samples)", "Mean shift"), ("Granger BIC", "Granger")):
    if a in R and b in R: pairs[f"{a} vs {b}"] = paired(a, b)
json.dump({"summary": summary, "pairs": pairs, "spec": spec}, open(f"{O}/agg2.json", "w"), indent=1)
for m, v in summary.items():
    print(f"{m:22s} n={v['seeds']:2d} H1 {v['hit1']*100:5.1f} H3 {v['hit3']*100:5.1f} H5 {v['hit5']*100:5.1f} "
          f"({v['min_hit5']*100:.1f}-{v['max_hit5']*100:.1f}) MRR {v['mrr']:.3f} maxp {v['max_p']:.4f}  {v['perfault_hit5']}")
print()
for k, v in pairs.items():
    print(f"{k:45s} " + "  ".join(f"{m}: {v[m]['mean']*100 if m!='mrr' else v[m]['mean']:+.3f} W{v[m]['wins']}/L{v[m]['losses']} p={v[m]['p']:.3f}" for m in ("hit1", "hit5", "mrr")))
print()
for m, v in spec.items(): print(f"{m:22s} Stream-4 channel in top 5 for other faults: {v['s4_in_top5_nonS4']:7s} fault-agnostic Hit@5 {v['agnostic_hit5']*100:5.1f} | H5 matched {v['matched_hit5']*100:5.1f} cross {v['cross_hit5']*100:5.1f} | H1 matched {v['matched_hit1']*100:5.1f} cross {v['cross_hit1']*100:5.1f}")
sig = {}
for m, bs in R.items():
    rows = [metrics(b) for b in bs.values()]
    sig[m] = (sum(r[f"p{k}"] < 0.05 for r in rows for k in (1, 3, 5)), 3 * len(rows))
print("SIG", {m: f"{a}/{b}" for m, (a, b) in sig.items()})
