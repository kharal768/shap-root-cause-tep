"""Aggregates the robustness runs: accuracy, evaluated faults, hit rates, chance tests and paired intervals.

Usage: python robustness_aggregate.py <root>. Output: ROOT/out/robagg.json. Paper: Supplementary Section S7."""
import json, os, sys, glob
import numpy as np
from math import comb
from fault_metadata import ALL_CHANNEL_NAMES as N, GROUND_TRUTH_CHANNELS as GT
F = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]; FI = {f: i for i, f in enumerate(F)}
ddir = sys.argv[1]; O = os.path.join(ddir, "out")
rng = np.random.default_rng(99); B = 10000
COUNTS = rng.multinomial(len(F), [1 / len(F)] * len(F), size=B).astype(float)


def ch(g, k): return 1 - comb(52 - g, k) / comb(52, k)


def pb(ps, x):
    d = np.zeros(len(ps) + 1); d[0] = 1
    for p in ps: d[1:] = d[1:] * (1 - p) + d[:-1] * p; d[0] *= 1 - p
    return float(d[x:].sum())


def mat(entries):
    V = np.full((len(entries), 3, len(F)), np.nan)
    for si, e in enumerate(entries):
        for f in F:
            v = e.get(str(f))
            if v is None: continue
            r = min(v["rank"][N.index(c)] for c in GT[f]); V[si, :, FI[f]] = (r <= 1, r <= 5, 1 / r)
    return V


def seedmean(V, C):
    W = ~np.isnan(V); Vz = np.nan_to_num(V)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = (C @ Vz.T) / (C @ W.T.astype(float))
    return np.nanmean(per, axis=1)


rows = {}
for kind, pat in (("test", "robtest_seed*.json"), ("train", "robtrain_seed*.json")):
    files = sorted(glob.glob(os.path.join(O, pat)))
    if not files: continue
    runs = [json.load(open(f)) for f in files]
    conds = [k for k in runs[0] if k != "seed"]
    for c in conds:
        for m, key in ((("mlp", "shap"), ("mlp", "ms"), ("xgb", "shap")) if kind == "test" else ((None, "shap"), (None, "ms"))):
            ents, accs = [], []
            for r in runs:
                rr = r[c][m] if m else r[c]
                if key not in rr: continue
                ents.append(rr[key]); accs.append(rr["acc"])
            if not ents: continue
            V = mat(ents)
            sig = 0; nf = []
            for si in range(len(ents)):
                ev = [f for f in F if not np.isnan(V[si, 0, FI[f]])]; nf.append(len(ev))
                for k, j in ((1, 0), (5, 1)):
                    h = int(np.nansum(V[si, j])); sig += pb([ch(len(GT[f]), k) for f in ev], h) < 0.05
            name = f"{kind}:{c}:{(m or 'mlp')}-{key}"
            rows[name] = {"V": V, "acc": float(np.mean(accs)), "n_faults": float(np.mean(nf)), "sig": f"{sig}/{2 * len(ents)}",
                          **{k: float(np.nanmean(np.nanmean(V[:, j], axis=1))) for j, k in enumerate(("hit1", "hit5", "mrr"))}}
# reference for the training-condition mean shift: the main-analysis mean shift from all training samples
sw = [json.load(open(os.path.join(O, f"sweep_seed{s}.json")))["baseline"] for s in range(10) if os.path.exists(os.path.join(O, f"sweep_seed{s}.json"))]
Vr = mat(sw)
rows["ref:clean:mlp-mstrain"] = {"V": Vr, "acc": float("nan"), "n_faults": 12.0, "sig": "-",
                                 **{k: float(np.nanmean(np.nanmean(Vr[:, j], axis=1))) for j, k in enumerate(("hit1", "hit5", "mrr"))}}
out = {}
for name, r in rows.items():
    kind, c, mk = name.split(":")
    if kind == "test": ref = rows.get(f"test:clean:{mk}")
    elif kind == "train": ref = rows.get("test:clean:mlp-shap") if mk.endswith("shap") else rows.get("ref:clean:mlp-mstrain")
    else: ref = None
    e = {k: v for k, v in r.items() if k != "V"}
    if ref is not None and ref is not r:
        for j, k in enumerate(("hit1", "hit5", "mrr")):
            D = r["V"][:, j] - ref["V"][:len(r["V"]), j]
            pt = float(seedmean(D, np.ones((1, len(F))))[0]); bs = seedmean(D, COUNTS)
            e[f"d_{k}"] = [pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    out[name] = e
json.dump(out, open(os.path.join(O, "robagg.json"), "w"), indent=1)
for n, e in out.items():
    d = " ".join(f"d{k[0:4]} {e[f'd_{k}'][0]*(100 if k!='mrr' else 1):+.1f}[{e[f'd_{k}'][1]*(100 if k!='mrr' else 1):+.1f},{e[f'd_{k}'][2]*(100 if k!='mrr' else 1):+.1f}]" for k in ("hit1", "hit5") if f"d_{k}" in e)
    print(f"{n:30s} acc {e['acc']*100:5.1f} nf {e['n_faults']:4.1f} H1 {e['hit1']*100:5.1f} H5 {e['hit5']*100:5.1f} MRR {e['mrr']:.3f} sig {e['sig']:6s} {d}")
