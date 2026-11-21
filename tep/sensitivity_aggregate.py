"""Aggregates the one-at-a-time sensitivity sweeps.

Usage: python sensitivity_aggregate.py <root>. Output: ROOT/out/sensagg.json. Paper: Supplementary Table S6."""
import json, os, sys, glob
import numpy as np
from math import comb
from fault_metadata import ALL_CHANNEL_NAMES as N, GROUND_TRUTH_CHANNELS as GT
F = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]
ddir = sys.argv[1]; O = os.path.join(ddir, "out")


def ch(g, k): return 1 - comb(52 - g, k) / comb(52, k)


def pb(ps, x):
    d = np.zeros(len(ps) + 1); d[0] = 1
    for p in ps: d[1:] = d[1:] * (1 - p) + d[:-1] * p; d[0] *= 1 - p
    return float(d[x:].sum())


def score(entries):
    h1, h5, rr, nf, sig = [], [], [], [], 0
    for e in entries:
        br = {f: min(e[str(f)]["rank"][N.index(c)] for c in GT[f]) for f in F if e.get(str(f))}
        if not br: continue
        fs = list(br)
        h1.append(np.mean([br[f] <= 1 for f in fs])); h5.append(np.mean([br[f] <= 5 for f in fs]))
        rr.append(np.mean([1 / br[f] for f in fs])); nf.append(len(fs))
        for k, h in ((1, sum(br[f] <= 1 for f in fs)), (5, sum(br[f] <= 5 for f in fs))):
            sig += pb([ch(len(GT[f]), k) for f in fs], h) < 0.05
    return {"hit1": float(np.mean(h1)), "hit5": float(np.mean(h5)), "mrr": float(np.mean(rr)),
            "n_faults": float(np.mean(nf)), "sig": f"{sig}/{2 * len(h1)}", "splits": len(h1)}


out = {}
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(O, "sens_seed*.json")))]
for par in ("n_explained", "bg_size", "bg_normal", "noise", "missing", "train_frac"):
    if not runs: break
    levels = list(runs[0][par])
    out[par] = {"shap": {lv: score([r[par][lv] for r in runs]) for lv in levels}}
    if par + "_ms" in runs[0]:
        out[par]["mean_shift"] = {lv: score([r[par + "_ms"][lv] for r in runs]) for lv in levels}
    key = {"noise": "noise", "missing": "missing", "train_frac": "train"}.get(par)
    if key:
        for lv in levels:
            out[par]["shap"][lv]["acc"] = float(np.mean([r["acc"][f"{key}{lv}"] for r in runs]))
base = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(O, "sensbase_seed*.json")))]
if base:
    out["pca"] = {"pca": {lv: score([r["pca"][lv] for r in base]) for lv in base[0]["pca"]}}
    out["granger_alpha"] = {"granger": {lv: score([r["granger"][lv] for r in base]) for lv in base[0]["granger"]}}
ev = json.load(open(os.path.join(O, "sens_eval.json")))
out["threshold"] = {"MLP SHAP": {t: v["MLP SHAP"] for t, v in ev["threshold"].items()}}
for par, d in out.items():
    for meth, lv in d.items():
        h5 = [v["hit5"] for v in lv.values()]; h1 = [v["hit1"] for v in lv.values()]
        print(f"{par:13s} {meth:10s} swing H5 {(max(h5) - min(h5)) * 100:5.1f}  H1 {(max(h1) - min(h1)) * 100:5.1f} | " +
              "  ".join(f"{k}: {v['hit1'] * 100:.1f}/{v['hit5'] * 100:.1f}" + (f" acc{v['acc'] * 100:.0f}" if "acc" in v else "") +
                        (f" sig{v['sig']}" if "sig" in v else "") for k, v in lv.items()))
json.dump(out, open(os.path.join(O, "sensagg.json"), "w"), indent=1)
