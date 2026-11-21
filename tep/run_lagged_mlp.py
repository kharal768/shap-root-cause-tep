"""Lagged MLP (value plus causal rolling mean and s.d. over 10 samples) with permutation SHAP, and the
mean/variance baseline.

Input: ROOT/data/tep_clean.npz. Output: ROOT/out/temporal_seed<k>.json. Paper: Table 3."""
import os
import json, sys, warnings
import numpy as np, pandas as pd, shap
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS
ROOT = os.environ.get("TEP_ROOT", "../runs/revised")  # folder with data/tep_clean.npz and out/
warnings.filterwarnings("ignore")
W = 10


def ranking(score):
    order = np.argsort(score)[::-1]
    r = np.empty(len(order), int); r[order] = np.arange(1, len(order) + 1)
    return r.tolist()


def lagged(X):
    df = pd.DataFrame(X)
    m = df.rolling(W, min_periods=1).mean().values
    s = df.rolling(W, min_periods=2).std().fillna(0).values
    return np.hstack([X, m, s])


def run(seed, X, XL, y):
    Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, 0.3, seed=seed)
    XLtr, _, XLte, _, _ = episode_train_test_split(XL, y, 0.3, seed=seed)
    res = {"seed": seed}
    mu0, sd0 = Xtr[ytr == 0].mean(0), Xtr[ytr == 0].std(0) + 1e-12
    res["baseline_var"] = {}
    for f in VALIDATABLE_FAULTS:
        Xf = Xtr[ytr == f]
        score = np.maximum(np.abs(Xf.mean(0) - mu0) / sd0, np.abs(np.log((Xf.std(0) + 1e-12) / sd0)))
        res["baseline_var"][f] = {"rank": ranking(score)}
    sc = StandardScaler().fit(XLtr); A, B = sc.transform(XLtr), sc.transform(XLte)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=300, random_state=seed,
                        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15).fit(A, ytr)
    p = mlp.predict(B)
    rep = classification_report(yte, p, output_dict=True, zero_division=0)
    rng = np.random.default_rng(seed); bg = A[rng.choice(len(A), 30, replace=False)]
    sh = {}
    for f in VALIDATABLE_FAULTS:
        idx = np.where((yte == f) & (p == f))[0]
        if len(idx) < 5: sh[f] = None; continue
        s = np.random.default_rng(seed).choice(idx, size=min(30, len(idx)), replace=False)
        v = np.abs(shap.Explainer(mlp.predict_proba, bg, algorithm="permutation", seed=0)(B[s]).values[:, :, f]).mean(0)
        sh[f] = {"n": int(len(s)), "rank": ranking(v[:52] + v[52:104] + v[104:])}
    res["mlp_lag"] = {"acc": accuracy_score(yte, p), "macro_f1": f1_score(yte, p, average="macro"),
                      "f1": {c: rep[str(c)]["f1-score"] for c in range(21) if str(c) in rep}, "shap": sh}
    json.dump(res, open(ROOT + f"/out/temporal_seed{seed}.json", "w"))
    print(f"seed {seed}: lagged MLP acc {res['mlp_lag']['acc']:.4f}", flush=True)


if __name__ == "__main__":
    d = np.load(ROOT + "/data/tep_clean.npz", allow_pickle=True)
    X, y = d["X"], d["y"]; XL = lagged(X)
    for s in [int(a) for a in sys.argv[1:]]:
        run(s, X, XL, y)
