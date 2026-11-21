"""Validation-tuned MLP (episode-level hold-out, macro-F1) and its SHAP rankings.

Usage: python tuning_mlp.py <root> <seeds>. Output: ROOT/out/hpo_seed<k>.json. Paper: Supplementary Section S5."""
import json, os, sys, time, warnings
import numpy as np, shap, xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from data_split import episode_train_test_split
from run_mlp_xgboost import explain
warnings.filterwarnings("ignore")

MLP_GRID = [(h, a) for h in [(64,), (128, 64), (256, 128)] for a in (1e-4, 1e-3)]
XGB_GRID = [(d, lr) for d in (4, 6, 8) for lr in (0.05, 0.1)]


def mk_mlp(h, a, seed):
    return MLPClassifier(hidden_layer_sizes=h, alpha=a, max_iter=300, random_state=seed, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=15)


def mk_xgb(dp, lr, seed):
    return xgb.XGBClassifier(n_estimators=200, max_depth=dp, learning_rate=lr, subsample=0.9, colsample_bytree=0.9,
                             random_state=seed, n_jobs=2, tree_method="hist")


if __name__ == "__main__":
    ddir = sys.argv[1]
    d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True)
    X, y = d["X"], d["y"]
    for seed in [int(a) for a in sys.argv[2:]]:
        t = time.time()
        Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
        Xfit, yfit, Xval, yval, _ = episode_train_test_split(Xtr, ytr, test_frac=0.2, seed=100 + seed)
        sv = StandardScaler().fit(Xfit); A, V = sv.transform(Xfit), sv.transform(Xval)
        res = {"seed": seed, "mlp_val": {}, "xgb_val": {}}
        for h, a in MLP_GRID:
            res["mlp_val"][f"{h}|{a}"] = f1_score(yval, mk_mlp(h, a, seed).fit(A, yfit).predict(V), average="macro")
        bm = max(res["mlp_val"], key=res["mlp_val"].get)
        res["mlp_best"] = bm
        sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        rng = np.random.default_rng(seed)
        bg = Xtr_s[rng.choice(len(Xtr_s), 30, replace=False)]
        h, a = MLP_GRID[[f"{g[0]}|{g[1]}" for g in MLP_GRID].index(bm)]
        mlp = mk_mlp(h, a, seed).fit(Xtr_s, ytr); p = mlp.predict(Xte_s)
        res["mlp"] = {"acc": accuracy_score(yte, p), "macro_f1": f1_score(yte, p, average="macro"),
                      "shap": explain(mlp.predict_proba, bg, Xte_s, yte, p, np.random.default_rng(seed))}
        json.dump(res, open(os.path.join(ddir, f"out/hpo_seed{seed}.json"), "w"), default=float)
        print(ddir, seed, bm, round(res["mlp"]["acc"], 4),
              f"{time.time() - t:.0f}s", flush=True)
