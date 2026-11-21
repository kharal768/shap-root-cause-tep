"""Agreement and Cohen's kappa of top-five hit outcomes between each SHAP ranking and its deviation ranking
and the classifier-free statistics.

Usage: python hit_agreement_kappa.py <root>. Output: ROOT/out/agree_kappa.json. Paper: Section 4.4."""
import sys, io, contextlib, json, numpy as np
ddir = sys.argv[1]
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ["aggregate_all.py", ddir]; exec(open("aggregate_all.py").read())
def outcomes(a, b):
    x = []; y = []
    for s in src:
        if a not in src[s] or b not in src[s]: continue
        A = ranks(src[s][a]); B = ranks(src[s][b])
        for f in set(A) & set(B): x.append(A[f] <= 5); y.append(B[f] <= 5)
    x = np.array(x); y = np.array(y); po = np.mean(x == y); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan, len(x)
res = {}
pairs = [("MLP SHAP", "MLP abl: deviation"), ("XGB Tree SHAP", "XGB abl: deviation"), ("CNN SHAP", "CNN abl: deviation"), ("GRU SHAP", "GRU abl: deviation")]
for shp, dev in pairs:
    res[shp] = {}
    for other in (dev, "Mean shift", "KS", "PCA-RBC", "Mean/var"):
        po, k, n = outcomes(shp, other); res[shp][other] = [po, k, n]
    print(shp, {o: (round(v[0] * 100, 1), round(v[1], 2)) for o, v in res[shp].items()})
json.dump(res, open(f"{ddir}/out/agree_kappa.json", "w"), indent=1)
