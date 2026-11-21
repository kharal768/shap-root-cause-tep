"""Fit and ranking time of every ranking on one split, single CPU thread.

Usage: OMP_NUM_THREADS=1 python computational_cost.py <root>. Output: ROOT/out/timing_seed0.json.
Paper: Supplementary Table S9, Discussion."""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import torch
torch.set_num_threads(1)
import shap, xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS as F
import run_mlp_xgboost, baselines2 as B, fair as FA
import run_lagged_mlp as T
import run_sequence_models as S
torch.set_num_threads(1)

ddir = sys.argv[1]; seed = 0
d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True)
X, y = d["X"], d["y"]; Xf = X.astype(float)
Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
tr, te = B.split_idx(y, seed)
res = {}


def rec(name, fit, rank, n_samples=None):
    res[name] = {"fit_s": fit, "rank_s": rank, "n_explained": n_samples}
    print(f"{name:32s} fit {fit:8.1f}s  rank {rank:8.2f}s  samples {n_samples}", flush=True)


def nexp(out): return int(sum(v["n"] for v in out.values() if v))


# classifier-free
N = Xtr[ytr == 0].astype(float)

t = time.time()
mu0, sd0 = N.mean(0), N.std(0) + 1e-12
for f in F: B.ranking(np.abs(Xtr[ytr == f].mean(0) - mu0) / sd0)
rec("Mean shift", 0.0, time.time() - t)
t = time.time()
for f in F:
    A = Xtr[ytr == f]; ms = np.abs(A.mean(0) - mu0) / sd0
    B.ranking(np.maximum(ms, np.abs(np.log((A.std(0) + 1e-12) / sd0))))
rec("Mean/variance", 0.0, time.time() - t)
from scipy import stats
t = time.time()
for f in F:
    A = Xtr[ytr == f].astype(float)
    B.ranking(np.array([stats.ks_2samp(A[:, j], N[:, j]).statistic for j in range(52)]))
rec("Kolmogorov-Smirnov", 0.0, time.time() - t)
t = time.time(); pca = FA.pca_model(N, 0.90); tf = time.time() - t
t = time.time()
for f in F: FA.rbc_rank(Xtr[ytr == f].astype(float), pca)
rec("PCA contribution", tf, time.time() - t)
t = time.time()
for f in F: B.granger_rank(Xf, y, tr, f)
rec("Granger causality", 0.0, time.time() - t)

# MLP + permutation SHAP
sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
rng = np.random.default_rng(seed); bg = Xtr_s[rng.choice(len(Xtr_s), 30, replace=False)]
t = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=300, random_state=seed, early_stopping=True,
                    validation_fraction=0.1, n_iter_no_change=15).fit(Xtr_s, ytr)
tf = time.time() - t; p = mlp.predict(Xte_s)
t = time.time(); o = sweep.explain(mlp.predict_proba, bg, Xte_s, yte, p, np.random.default_rng(seed))
rec("MLP, permutation SHAP", tf, time.time() - t, nexp(o))

# XGBoost
t = time.time()
xg = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                       random_state=seed, n_jobs=1, tree_method="hist").fit(Xtr_s, ytr)
tf = time.time() - t; p = xg.predict(Xte_s)
t = time.time(); o = sweep.explain(None, None, Xte_s, yte, p, np.random.default_rng(seed), tree=shap.TreeExplainer(xg))
rec("XGBoost, TreeExplainer", tf, time.time() - t, nexp(o))
t = time.time(); o = sweep.explain(xg.predict_proba, bg, Xte_s, yte, p, np.random.default_rng(seed))
rec("XGBoost, permutation SHAP", tf, time.time() - t, nexp(o))

# lagged MLP
t = time.time()
XL = T.lagged(X)
XLtr, _, XLte, _, _ = episode_train_test_split(XL, y, 0.3, seed=seed)
scl = StandardScaler().fit(XLtr); A_, B_ = scl.transform(XLtr), scl.transform(XLte)
ml = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=300, random_state=seed, early_stopping=True,
                   validation_fraction=0.1, n_iter_no_change=15).fit(A_, ytr)
tf = time.time() - t; p = ml.predict(B_)
bgl = A_[np.random.default_rng(seed).choice(len(A_), 30, replace=False)]
t = time.time(); n = 0
for f in F:
    idx = np.where((yte == f) & (p == f))[0]
    if len(idx) < 5: continue
    s = np.random.default_rng(seed).choice(idx, size=min(30, len(idx)), replace=False); n += len(s)
    shap.Explainer(ml.predict_proba, bgl, algorithm="permutation", seed=0)(B_[s])
rec("Lagged MLP, permutation SHAP", tf, time.time() - t, n)

# sequence models
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
Xs = ((X - mu) / sd).astype(np.float32); yte2 = y[te]
for arch in ("cnn", "gru"):
    S.ARCH = arch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = S.GRU() if arch == "gru" else S.CNN()
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4); lossf = torch.nn.CrossEntropyLoss()
    t = time.time()
    for ep in range(S.EPOCHS):
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), 512):
            b = tr[perm[i:i + 512]]
            xb = torch.as_tensor(S.windows(Xs, b)); yb = torch.as_tensor(y[b], dtype=torch.long)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    model.eval(); tf = time.time() - t
    p = S.proba(model, S.windows(Xs, te)).argmax(1)
    bgw = S.windows(Xs, rng.choice(tr, 30, replace=False))
    sel = {}
    for f in F:
        ok = np.where((yte2 == f) & (p == f))[0]
        if len(ok) >= 5: sel[f] = np.random.default_rng(seed).choice(ok, size=min(30, len(ok)), replace=False)
    n = int(sum(len(v) for v in sel.values()))
    t = time.time()
    for f, s in sel.items():
        xs = S.windows(Xs, te[s])
        for k in range(len(s)): S.channel_shap(model, xs[k], bgw, f, rng)
    rec(f"{arch.upper()}, channel permutation SHAP", tf, time.time() - t, n)
    steps = 32; alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    t = time.time()
    for f, s in sel.items():
        xs = torch.as_tensor(S.windows(Xs, te[s])); m = len(s)
        path = (alphas[:, None, None, None] * xs[None]).reshape(steps * m, *xs.shape[1:]).requires_grad_(True)
        g, = torch.autograd.grad(torch.softmax(model(path), 1)[:, f].sum(), path)
    rec(f"{arch.upper()}, Integrated Gradients", tf, time.time() - t, n)

json.dump(res, open(os.path.join(ddir, "out/timing_seed0.json"), "w"), indent=1)
