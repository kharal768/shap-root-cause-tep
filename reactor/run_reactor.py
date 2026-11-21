"""Evaluation on the reactor: MLP with permutation and conditional SHAP, XGBoost with TreeExplainer, an
untrained MLP, the classifier-free rankings and the deviation ranking, over ten episode-level splits.

Usage: python run_reactor.py <seeds>. Input: ../data/reactor.npz. Output: ../out/cstr_seed<k>.json.
Paper: Section 4.7, Supplementary Table S10."""
import json, sys, warnings, numpy as np, shap, xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from data_split import episode_train_test_split
from conditional_shap import cond_phi
warnings.filterwarnings("ignore")
d = np.load("../data/reactor.npz", allow_pickle=True); X, y = d["X"], d["y"]
meta = json.load(open("../data/meta.json")); N = meta["names"]; FAULTS = list(range(1, 12))

def ranking(s):
    o = np.argsort(s)[::-1]; r = np.empty(len(o), int); r[o] = np.arange(1, len(o) + 1); return r.tolist()
def as3(v): return v if v.ndim == 3 else np.moveaxis(v, 0, -1)

for seed in map(int, sys.argv[1:]):
    Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    rng = np.random.default_rng(seed); bg = A[rng.choice(len(A), 30, replace=False)]
    mlp = MLPClassifier((128, 64), alpha=1e-4, max_iter=300, random_state=seed, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15).fit(A, ytr)
    rnd = MLPClassifier((128, 64), max_iter=1, random_state=seed + 100); rnd.partial_fit(A[:1], ytr[:1], classes=np.arange(12))
    xg = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=2, tree_method="hist").fit(A, ytr)
    tex = shap.TreeExplainer(xg); pm, px = mlp.predict(B), xg.predict(B)
    mu, C = A.mean(0), np.cov(A, rowvar=False) + 1e-3 * np.eye(A.shape[1]); crng = np.random.default_rng(5000 + seed)
    N0 = A[ytr == 0]; mu0, sd0 = N0.mean(0), N0.std(0) + 1e-9
    # PCA-RBC on normal training data, 90% variance
    lam, V = np.linalg.eigh(np.cov(((N0 - mu0) / sd0).T)); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
    a = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.90) + 1); P, L = V[:, :a], lam[:a]
    Ct = np.eye(len(N)) - P @ P.T; Dt = P @ np.diag(1 / L) @ P.T; Z0 = (N0 - mu0) / sd0
    t2 = np.einsum("ij,jk,ik->i", Z0, Dt, Z0); spe = np.einsum("ij,jk,ik->i", Z0, Ct, Z0)
    Phi = Dt / np.percentile(t2, 99) + Ct / np.percentile(spe, 99)
    R = {k: {} for k in ["MLP SHAP", "MLP cond. SHAP", "XGB Tree SHAP", "MLP-random SHAP", "Mean shift", "Mean/var", "KS", "PCA-RBC", "Deviation"]}
    srng = np.random.default_rng(seed)
    for f in FAULTS:
        Ff = A[ytr == f]
        ms = np.abs(Ff.mean(0) - mu0) / sd0; R["Mean shift"][f] = ranking(ms)
        R["Mean/var"][f] = ranking(np.maximum(ms, np.abs(np.log((Ff.std(0) + 1e-9) / sd0))))
        R["KS"][f] = ranking([ks_2samp(Ff[:, j], N0[:, j]).statistic for j in range(len(N))])
        Zf = (Ff - mu0) / sd0; R["PCA-RBC"][f] = ranking(((Zf @ Phi) ** 2 / np.diag(Phi)).mean(0))
        for name, pred, kind in [("MLP SHAP", pm, "mlp"), ("XGB Tree SHAP", px, "xgb")]:
            idx = np.where((yte == f) & (pred == f))[0]
            if len(idx) < 5:
                R[name][f] = None
                if kind == "mlp": R["MLP cond. SHAP"][f] = R["MLP-random SHAP"][f] = R["Deviation"][f] = None
                continue
            s = srng.choice(idx, size=min(30, len(idx)), replace=False); xs = B[s]
            if kind == "mlp":
                v = as3(shap.Explainer(mlp.predict_proba, bg, algorithm="permutation", seed=0)(xs).values)[:, :, f]
                R[name][f] = ranking(np.abs(v).mean(0))
                vr = as3(shap.Explainer(rnd.predict_proba, bg, algorithm="permutation", seed=0)(xs).values)[:, :, f]
                R["MLP-random SHAP"][f] = ranking(np.abs(vr).mean(0))
                vc = np.mean([np.abs(cond_phi(mlp.predict_proba, x, mu, C, f, crng)) for x in xs], 0)
                R["MLP cond. SHAP"][f] = ranking(vc)
                R["Deviation"][f] = ranking(np.abs(xs - bg.mean(0)).mean(0))
            else:
                R[name][f] = ranking(np.abs(as3(tex.shap_values(xs))[:, :, f]).mean(0))
    out = {"seed": seed, "acc_mlp": float((pm == yte).mean()), "acc_xgb": float((px == yte).mean()), "pca_a": a, "rank": R}
    json.dump(out, open(f"../out/cstr_seed{seed}.json", "w"))
    print(seed, round(out["acc_mlp"], 4), round(out["acc_xgb"], 4), flush=True)
