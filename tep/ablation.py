"""Component ablation of the SHAP pipeline (MLP with permutation SHAP; XGBoost with TreeExplainer): each
variant changes one component (background, explained output, explained samples, aggregation) or removes
model and explainer (deviation ranking).

Usage: python ablation.py <root> mlp|xgb <seeds>. Output: ROOT/out/ablate_<model>_seed<k>.json. Paper: Table 5."""
import json, os, sys, time, warnings
import numpy as np, shap
from sklearn.preprocessing import StandardScaler
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS as F
warnings.filterwarnings("ignore")


def ranking(score):
    order = np.argsort(score)[::-1]; r = np.empty(len(order), int); r[order] = np.arange(1, len(order) + 1)
    return r.tolist()


def as3(v):
    if isinstance(v, list): return np.stack(v, axis=-1)
    return v if v.ndim == 3 else np.moveaxis(v, 0, -1)


def other(f):
    return f % 20 + 1 if f % 20 + 1 != f else (f + 1) % 20 + 1


if __name__ == "__main__":
    ddir, model = sys.argv[1], sys.argv[2]
    d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True)
    X, y = d["X"], d["y"]
    for seed in [int(a) for a in sys.argv[3:]]:
        t0 = time.time()
        Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
        sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        rng = np.random.default_rng(seed)
        bg = Xtr_s[rng.choice(len(Xtr_s), 30, replace=False)]
        nrm = np.where(ytr == 0)[0]
        bgn = Xtr_s[np.random.default_rng(1000 + seed).choice(nrm, 30, replace=False)]
        if model == "mlp":
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4,
                                max_iter=300, random_state=seed, early_stopping=True,
                                validation_fraction=0.1, n_iter_no_change=15).fit(Xtr_s, ytr)
            def shapvals(Xs, background):
                return as3(shap.Explainer(clf.predict_proba, background, algorithm="permutation", seed=0)(Xs).values)
            sv_full = lambda Xs: shapvals(Xs, bg)
        else:
            import xgboost as xgb
            clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9,
                                    colsample_bytree=0.9, random_state=seed, n_jobs=1, tree_method="hist").fit(Xtr_s, ytr)
            tex = shap.TreeExplainer(clf)
            def shapvals(Xs, background):
                return as3(shap.TreeExplainer(clf, data=background, feature_perturbation="interventional",
                                          model_output="raw").shap_values(Xs))
            sv_full = lambda Xs: as3(tex.shap_values(Xs))
        pred = clf.predict(Xte_s)
        V = {k: {} for k in ("full", "bg_normal", "out_normal", "out_other", "all_samples", "signed", "deviation")}
        if model == "xgb": del V["bg_normal"]
        srng = np.random.default_rng(seed); arng = np.random.default_rng(2000 + seed)
        for f in F:
            idx = np.where((yte == f) & (pred == f))[0]
            alli = np.where(yte == f)[0]
            sa = arng.choice(alli, size=min(30, len(alli)), replace=False)
            va = sv_full(Xte_s[sa])
            V["all_samples"][f] = {"n": int(len(sa)), "rank": ranking(np.abs(va[:, :, f]).mean(0))}
            if len(idx) < 5:
                for k in V:
                    if k != "all_samples": V[k][f] = None
                continue
            s = srng.choice(idx, size=min(30, len(idx)), replace=False)
            v = sv_full(Xte_s[s])
            V["full"][f] = {"n": int(len(s)), "rank": ranking(np.abs(v[:, :, f]).mean(0))}
            V["out_normal"][f] = {"rank": ranking(np.abs(v[:, :, 0]).mean(0))}
            V["out_other"][f] = {"rank": ranking(np.abs(v[:, :, other(f)]).mean(0)), "class": other(f)}
            V["signed"][f] = {"rank": ranking(np.abs(v[:, :, f].mean(0)))}
            if model == "mlp":
                vn = shapvals(Xte_s[s], bgn)
                V["bg_normal"][f] = {"rank": ranking(np.abs(vn[:, :, f]).mean(0))}
            V["deviation"][f] = {"rank": ranking(np.abs(Xte_s[s] - bg.mean(0)).mean(0))}
        V["acc"] = float((pred == yte).mean()); V["seed"] = seed
        json.dump(V, open(os.path.join(ddir, f"out/ablate_{model}_seed{seed}.json"), "w"))
        print(ddir, model, seed, round(V["acc"], 4), f"{time.time() - t0:.0f}s", flush=True)
