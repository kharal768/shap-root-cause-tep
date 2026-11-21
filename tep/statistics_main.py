"""Statistical tests of the main results: split s.d., fault-level and group bootstrap intervals, effect
size against chance, fault-label permutation test, paired intervals and exact sign-flip tests (Holm).

Usage: python statistics_main.py <root> "<baseline>" [output name]. Output: ROOT/out/stats2.json
(revised dataset: "Mean shift"; original dataset: "Mean/var", and "Mean shift" with output
stats2_meanshift.json). Paper: Table 4, Section 4.2, Supplementary Table S2."""
import sys, io, contextlib, json, os, itertools
import numpy as np
ddir, strongest = sys.argv[1], sys.argv[2]
OUTNAME = sys.argv[3] if len(sys.argv) > 3 else "stats2.json"
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ["aggregate_all.py", ddir]
    exec(open("aggregate_all.py").read())

rng = np.random.default_rng(12345)
B = 10000
SETS = {}
for f in F: SETS.setdefault(frozenset(GT[f]), []).append(f)
CL = list(SETS.values())


def best_rank(e, f, g=None):
    v = e[g if g is not None else f]["rank"]
    return min(v[N.index(c)] for c in GT[f])


def per_fault(m):
    """fault -> list over seeds of (hit1, hit5, rr)"""
    out = {}
    for s in src:
        if m not in src[s]: continue
        e = full(src[s][m])
        for f in e:
            r = best_rank(e, f); out.setdefault(f, []).append((r <= 1, r <= 5, 1 / r))
    return {f: np.mean(v, axis=0) for f, v in out.items()}


chance5 = {f: ch(len(GT[f]), 5) for f in F}


def boot_ci(vals_by_fault, clusters=None, stat=lambda a: a.mean()):
    fs = sorted(vals_by_fault)
    if clusters is None:
        arr = np.array([vals_by_fault[f] for f in fs])
        idx = rng.integers(0, len(fs), (B, len(fs)))
        bs = np.array([stat(arr[i]) for i in idx])
    else:
        cl = [[f for f in c if f in vals_by_fault] for c in clusters]; cl = [c for c in cl if c]
        bs = []
        for _ in range(B):
            pick = [cl[j] for j in rng.integers(0, len(cl), len(cl))]
            bs.append(stat(np.array([vals_by_fault[f] for c in pick for f in c])))
        bs = np.array(bs)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def perm_test(m, k):
    obs = 0; null = np.zeros(B)
    for s in src:
        if m not in src[s]: continue
        e = full(src[s][m]); fs = sorted(e)
        hitmat = np.array([[best_rank(e, f, g) <= k for g in fs] for f in fs])  # [true fault, ranking used]
        obs += np.trace(hitmat)
        for b in range(B):
            p = rng.permutation(len(fs)); null[b] += hitmat[np.arange(len(fs)), p].sum()
    return int(obs), float(np.mean(null)), float((1 + np.sum(null >= obs)) / (1 + B))


def signflip_p(d):
    d = np.asarray(d); d = d[np.abs(d) > 1e-12]
    if len(d) == 0: return 1.0
    obs = abs(d.mean()); n = len(d)
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    return float(np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-12))


FI = {f: i for i, f in enumerate(F)}
COUNTS = rng.multinomial(len(F), [1 / len(F)] * len(F), size=B).astype(float)          # fault bootstrap
CLC = np.zeros((B, len(F)))                                                             # cluster bootstrap
for b in range(B):
    for j in rng.integers(0, len(CL), len(CL)):
        for f in CL[j]: CLC[b, FI[f]] += 1


def mat(m, j):
    """seeds x faults matrix of the per-fault value (hit1, hit5, rr) or nan if not evaluated"""
    seeds = sorted(s for s in src if m in src[s])
    V = np.full((len(seeds), len(F)), np.nan)
    for si, s in enumerate(seeds):
        e = full(src[s][m])
        for f in e:
            r = best_rank(e, f); V[si, FI[f]] = (r <= 1, r <= 5, 1 / r)[j]
    return seeds, V


def seedmean(V, C):
    """C: (B, F) weights -> (B,) mean over seeds of the weighted mean over evaluated faults"""
    W = ~np.isnan(V); Vz = np.nan_to_num(V)
    num = C @ Vz.T; den = C @ W.T.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = num / den
    return np.nanmean(per, axis=1)


def ci(x): return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


res = {"rankings": {}, "paired": {}}
for m in sorted(R):
    if m not in summary: continue
    entry = {}
    for j, k in enumerate(("hit1", "hit5", "mrr")):
        seeds, V = mat(m, j)
        point = float(seedmean(V, np.ones((1, len(F))))[0])
        sv = [np.nanmean(V[i]) for i in range(len(seeds))]
        entry[k] = {"mean": point, "sd_seeds": float(np.std(sv, ddof=1)), "ci_fault": ci(seedmean(V, COUNTS)),
                    "ci_cluster": ci(seedmean(V, CLC))}
    seeds, V5 = mat(m, 1)
    Cm = np.where(np.isnan(V5), np.nan, np.array([chance5[f] for f in F])[None, :])
    ratio_b = seedmean(V5, COUNTS) / seedmean(Cm, COUNTS)
    entry["ratio5"] = {"value": float(seedmean(V5, np.ones((1, len(F))))[0] / seedmean(Cm, np.ones((1, len(F))))[0]),
                       "ci": ci(ratio_b)}
    for k in (1, 5):
        o, nm, pp = perm_test(m, k); entry[f"perm{k}"] = {"obs": o, "null_mean": nm, "p": pp}
    res["rankings"][m] = entry

shap_m = [m for m in ("MLP SHAP", "Lagged MLP SHAP", "CNN SHAP", "GRU SHAP", "XGB Tree SHAP", "XGB perm SHAP") if m in R]
for j, k in enumerate(("hit1", "hit5", "mrr")):
    ps = []
    for a in shap_m:
        sa, Va = mat(a, j); sb, Vb = mat(strongest, j)
        common = sorted(set(sa) & set(sb))
        Va = Va[[sa.index(s) for s in common]]; Vb = Vb[[sb.index(s) for s in common]]
        D = Va - Vb                                    # nan where either missing
        point = float(seedmean(D, np.ones((1, len(F))))[0])
        dfault = np.nanmean(D, axis=0); dfault = dfault[~np.isnan(dfault)]
        pp = signflip_p(dfault)
        res["paired"].setdefault(a, {})[k] = {"mean": point, "ci": ci(seedmean(D, COUNTS)),
                                             "ci_cluster": ci(seedmean(D, CLC)), "p": pp,
                                             "n_nonzero": int(np.sum(np.abs(dfault) > 1e-12))}
        ps.append((pp, a))
    order = sorted(ps); n = len(order); running = 0
    for i, (pp, a) in enumerate(order):
        adj = min(1.0, max(running, (n - i) * pp)); running = adj
        res["paired"][a][k]["p_holm"] = adj

json.dump(res, open(os.path.join(ddir, "out", OUTNAME), "w"), indent=1)
for m, e in res["rankings"].items():
    print(f"{m:26s} H5 {e['hit5']['mean']*100:5.1f}±{e['hit5']['sd_seeds']*100:4.1f} CI[{e['hit5']['ci_fault'][0]*100:5.1f},{e['hit5']['ci_fault'][1]*100:5.1f}] cl[{e['hit5']['ci_cluster'][0]*100:5.1f},{e['hit5']['ci_cluster'][1]*100:5.1f}] "
          f"H1 {e['hit1']['mean']*100:5.1f}±{e['hit1']['sd_seeds']*100:4.1f} CI[{e['hit1']['ci_fault'][0]*100:5.1f},{e['hit1']['ci_fault'][1]*100:5.1f}] "
          f"MRR {e['mrr']['mean']:.3f}±{e['mrr']['sd_seeds']:.3f} CI[{e['mrr']['ci_fault'][0]:.3f},{e['mrr']['ci_fault'][1]:.3f}] "
          f"x{e['ratio5']['value']:.2f}[{e['ratio5']['ci'][0]:.2f},{e['ratio5']['ci'][1]:.2f}] perm5 p={e['perm5']['p']:.4f} ({e['perm5']['obs']} vs {e['perm5']['null_mean']:.1f}) perm1 p={e['perm1']['p']:.4f}")
print()
for a, e in res["paired"].items():
    print(f"{a:18s} vs {strongest}: " + "  ".join(f"{k} {e[k]['mean']*(100 if k!='mrr' else 1):+.3f} [{e[k]['ci'][0]*(100 if k!='mrr' else 1):+.3f},{e[k]['ci'][1]*(100 if k!='mrr' else 1):+.3f}] cl[{e[k]['ci_cluster'][0]*(100 if k!='mrr' else 1):+.2f},{e[k]['ci_cluster'][1]*(100 if k!='mrr' else 1):+.2f}] p={e[k]['p']:.4f} holm={e[k]['p_holm']:.4f} nz={e[k]['n_nonzero']}" for k in ("hit1", "hit5", "mrr")))
