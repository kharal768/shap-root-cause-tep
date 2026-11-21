"""MLP (permutation SHAP), XGBoost (TreeExplainer and permutation SHAP) and the mean-shift baseline
over ten episode-level splits.

Input: ROOT/data/tep_clean.npz. Output: ROOT/out/sweep_seed<k>.json (channel rankings per fault).
Paper: Table 3. Use --no-xgb-perm on the original dataset."""
import os
import json, sys, time, warnings
import numpy as np
import shap, xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_split import episode_train_test_split
from fault_metadata import ALL_CHANNEL_NAMES, VALIDATABLE_FAULTS
ROOT = os.environ.get("TEP_ROOT", "../runs/revised")  # folder with data/tep_clean.npz and out/
SKIP_XGB_PERM = "--no-xgb-perm" in sys.argv  # the original dataset was run without this explainer
warnings.filterwarnings("ignore")

N_BG, N_EXPLAIN = 30, 30


def ranking(mean_abs):
    order = np.argsort(mean_abs)[::-1]
    rank = np.empty(len(order), int); rank[order] = np.arange(1, len(order) + 1)
    return rank.tolist()  # rank[channel_idx]


def explain(fn, bg, Xte_s, yte, pred, rng, tree=None):
    out = {}
    for f in VALIDATABLE_FAULTS:
        idx = np.where((yte == f) & (pred == f))[0]
        if len(idx) < 5:
            out[f] = None; continue
        s = rng.choice(idx, size=min(N_EXPLAIN, len(idx)), replace=False)
        if tree is not None:
            sv = tree.shap_values(Xte_s[s]); v = sv[:, :, f] if sv.ndim == 3 else sv[f]
        else:
            e = shap.Explainer(fn, bg, algorithm="permutation", seed=0)
            v = e(Xte_s[s]).values[:, :, f]
        out[f] = {"n": int(len(s)), "rank": ranking(np.abs(v).mean(0))}
    return out


def run(seed):
    d = np.load(ROOT + "/data/tep_clean.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    Xtr, ytr, Xte, yte, rec = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    rng = np.random.default_rng(seed)
    bg = Xtr_s[rng.choice(len(Xtr_s), N_BG, replace=False)]
    res = {"seed": seed, "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "test_counts": {int(c): int((yte == c).sum()) for c in range(21)}}
    # baseline: |mean shift| of fault vs normal in training data, in normal-SD units
    mu0, sd0 = Xtr[ytr == 0].mean(0), Xtr[ytr == 0].std(0) + 1e-12
    res["baseline"] = {f: {"rank": ranking(np.abs(Xtr[ytr == f].mean(0) - mu0) / sd0)} for f in VALIDATABLE_FAULTS}
    # MLP
    t = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4,
                        max_iter=300, random_state=seed, early_stopping=True,
                        validation_fraction=0.1, n_iter_no_change=15).fit(Xtr_s, ytr)
    p = mlp.predict(Xte_s)
    rep = classification_report(yte, p, output_dict=True, zero_division=0)
    res["mlp"] = {"acc": accuracy_score(yte, p), "macro_f1": f1_score(yte, p, average="macro"),
                  "f1": {c: rep[str(c)]["f1-score"] for c in range(21) if str(c) in rep},
                  "shap": explain(mlp.predict_proba, bg, Xte_s, yte, p, np.random.default_rng(seed))}
    res["mlp"]["time"] = time.time() - t
    # XGBoost
    t = time.time()
    xg = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9,
                           colsample_bytree=0.9, random_state=seed, n_jobs=4, tree_method="hist").fit(Xtr_s, ytr)
    p = xg.predict(Xte_s)
    rep = classification_report(yte, p, output_dict=True, zero_division=0)
    res["xgb"] = {"acc": accuracy_score(yte, p), "macro_f1": f1_score(yte, p, average="macro"),
                  "f1": {c: rep[str(c)]["f1-score"] for c in range(21) if str(c) in rep},
                  "shap": explain(None, None, Xte_s, yte, p, np.random.default_rng(seed), tree=shap.TreeExplainer(xg))}
    if not SKIP_XGB_PERM:
        res["xgb"]["shap_perm"] = explain(xg.predict_proba, bg, Xte_s, yte, p, np.random.default_rng(seed))
    res["xgb"]["time"] = time.time() - t
    json.dump(res, open(ROOT + f"/out/sweep_seed{seed}.json", "w"))
    print(f"seed {seed}: mlp acc {res['mlp']['acc']:.4f} xgb acc {res['xgb']['acc']:.4f} "
          f"({res['mlp']['time']:.0f}s, {res['xgb']['time']:.0f}s)", flush=True)


if __name__ == "__main__":
    for s in [int(a) for a in sys.argv[1:] if not a.startswith("--")]:
        run(s)
