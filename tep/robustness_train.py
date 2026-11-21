"""Robustness to training-set changes: smaller sets, stronger imbalance, balanced classes (MLP).

Usage: python robustness_train.py <root> <seeds>. Output: ROOT/out/robtrain_seed<k>.json.
Paper: Table 6, Supplementary Section S7."""
import json, os, sys, time, warnings
import numpy as np, shap
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
    ddir = sys.argv[1]
    d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True)
    X, y = d["X"], d["y"]
    for seed in [int(a) for a in sys.argv[2:]]:
        t0 = time.time()
        Xtr0, ytr0, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
        g = np.random.default_rng(7000 + seed)
        sub = {}
        for frac in (0.1, 0.3):
            keep = np.concatenate([g.choice(np.where(ytr0 == c)[0], max(1, int(round(frac * (ytr0 == c).sum()))), replace=False)
                                   for c in np.unique(ytr0)])
            sub[f"size{int(frac * 100)}"] = np.sort(keep)
        keep = np.concatenate([np.where(ytr0 == 0)[0]] + [g.choice(np.where(ytr0 == c)[0], max(1, int(round(0.1 * (ytr0 == c).sum()))), replace=False)
                                                         for c in np.unique(ytr0) if c != 0])
        sub["imbalance"] = np.sort(keep)
        mx = max((ytr0 == c).sum() for c in np.unique(ytr0) if c != 0)
        keep = np.concatenate([g.choice(np.where(ytr0 == 0)[0], mx, replace=False)] + [np.where(ytr0 == c)[0] for c in np.unique(ytr0) if c != 0])
        sub["balanced"] = np.sort(keep)
        out = {"seed": seed}
        for name, kp in sub.items():
            Xtr, ytr = Xtr0[kp], ytr0[kp]
            sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            rng = np.random.default_rng(seed); bg = Xtr_s[rng.choice(len(Xtr_s), 30, replace=False)]
            mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4,
                                max_iter=300, random_state=seed, early_stopping=True,
                                validation_fraction=0.1, n_iter_no_change=15).fit(Xtr_s, ytr)
            pred = mlp.predict(Xte_s)
            mu0, sd0 = Xtr[ytr == 0].mean(0), Xtr[ytr == 0].std(0) + 1e-12
            r = {"n_train": int(len(ytr)), "normal_frac": float((ytr == 0).mean()), "acc": float((pred == yte).mean()),
                 "macro_f1": float(f1_score(yte, pred, average="macro")), "shap": {}, "ms": {}}
            srng = np.random.default_rng(seed)
            for f in F:
                r["ms"][f] = {"rank": ranking(np.abs(Xtr[ytr == f].mean(0) - mu0) / sd0)}
                idx = np.where((yte == f) & (pred == f))[0]
                if len(idx) < 5: r["shap"][f] = None; continue
                s_ = srng.choice(idx, size=min(30, len(idx)), replace=False)
                v = as3(shap.Explainer(mlp.predict_proba, bg, algorithm="permutation", seed=0)(Xte_s[s_]).values)
                r["shap"][f] = {"n": int(len(s_)), "rank": ranking(np.abs(v[:, :, f]).mean(0))}
            out[name] = r
        json.dump(out, open(os.path.join(ddir, f"out/robtrain_seed{seed}.json"), "w"))
        print(ddir, seed, {k: (v["n_train"], round(v["acc"], 3)) for k, v in out.items() if k != "seed"}, f"{time.time() - t0:.0f}s", flush=True)
