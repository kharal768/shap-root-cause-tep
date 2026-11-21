"""Robustness to test-time changes: measurement noise, missing values and the other simulator's data.

Usage: python robustness_test.py <root_A> <root_B> <seeds>. Output: ROOT_A/out/robtest_seed<k>.json.
Paper: Table 6, Supplementary Section S7."""
import json, os, sys, time, warnings
import numpy as np, shap, xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS as F
warnings.filterwarnings("ignore")


def ranking(score):
    order = np.argsort(score)[::-1]; r = np.empty(len(order), int); r[order] = np.arange(1, len(order) + 1)
    return r.tolist()


def as3(v):
    if isinstance(v, list): return np.stack(v, axis=-1)
    return v if v.ndim == 3 else np.moveaxis(v, 0, -1)


if __name__ == "__main__":
    dA, dB = sys.argv[1], sys.argv[2]
    A = np.load(os.path.join(dA, "data/tep_clean.npz"), allow_pickle=True)
    Bd = np.load(os.path.join(dB, "data/tep_clean.npz"), allow_pickle=True)
    X, y = A["X"], A["y"]; XB, yB = Bd["X"], Bd["y"]
    for seed in [int(a) for a in sys.argv[3:]]:
        t0 = time.time()
        Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
        _, _, XteB, yteB, _ = episode_train_test_split(XB, yB, test_frac=0.3, seed=seed)
        sc = StandardScaler().fit(Xtr); Xtr_s = sc.transform(Xtr)
        mu0, sd0 = Xtr[ytr == 0].mean(0), Xtr[ytr == 0].std(0)
        rng = np.random.default_rng(seed)
        bg = Xtr_s[rng.choice(len(Xtr_s), 30, replace=False)]
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4,
                            max_iter=300, random_state=seed, early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=15).fit(Xtr_s, ytr)
        xg = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                               random_state=seed, n_jobs=1, tree_method="hist").fit(Xtr_s, ytr)
        tex = shap.TreeExplainer(xg)
        prng = np.random.default_rng(5000 + seed)
        conds = {"clean": (Xte, yte)}
        for s in (0.1, 0.5, 1.0):
            conds[f"noise{s}"] = (Xte + prng.normal(0, 1, Xte.shape) * (s * sd0), yte)
        for p in (0.1, 0.3):
            Xm = Xte.astype(float).copy(); mask = prng.random(Xm.shape) < p
            Xm[mask] = np.broadcast_to(sc.mean_, Xm.shape)[mask]
            conds[f"missing{p}"] = (Xm, yte)
        conds["cross"] = (XteB, yteB)
        out = {"seed": seed}
        for name, (Xc, yc) in conds.items():
            Xc_s = sc.transform(Xc)
            res = {}
            for mname, clf in (("mlp", mlp), ("xgb", xg)):
                pred = clf.predict(Xc_s)
                r = {"acc": float((pred == yc).mean()), "macro_f1": float(f1_score(yc, pred, average="macro")), "shap": {}}
                if mname == "mlp": r["ms"] = {}
                srng = np.random.default_rng(seed)
                for f in F:
                    idx = np.where((yc == f) & (pred == f))[0]
                    if len(idx) < 5:
                        r["shap"][f] = None
                        if mname == "mlp": r["ms"][f] = None
                        continue
                    s_ = srng.choice(idx, size=min(30, len(idx)), replace=False)
                    if mname == "mlp":
                        v = as3(shap.Explainer(mlp.predict_proba, bg, algorithm="permutation", seed=0)(Xc_s[s_]).values)
                        r["ms"][f] = {"rank": ranking(np.abs(Xc[s_].mean(0) - mu0) / (sd0 + 1e-12))}
                    else:
                        v = as3(tex.shap_values(Xc_s[s_]))
                    r["shap"][f] = {"n": int(len(s_)), "rank": ranking(np.abs(v[:, :, f]).mean(0))}
                res[mname] = r
            out[name] = res
        json.dump(out, open(os.path.join(dA, f"out/robtest_seed{seed}.json"), "w"))
        print(dA, seed, {k: (round(v["mlp"]["acc"], 3), round(v["xgb"]["acc"], 3)) for k, v in out.items() if k != "seed"},
              f"{time.time() - t0:.0f}s", flush=True)
