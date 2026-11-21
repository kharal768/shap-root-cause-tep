"""Sensitivity to the label onset: the first 0, 1, 3 or 6 hours of every fault episode are excluded.

Usage: python sensitivity_onset.py <root> <seeds>. Output: ROOT/out/onset_seed<k>.json. Paper: Supplementary Section S8."""
import json, os, sys, time, warnings
import numpy as np, shap
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from data_split import episode_train_test_split, find_episodes
from fault_metadata import VALIDATABLE_FAULTS as F
import baselines_extra as B
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
    pos = np.zeros(len(y), int)
    for s, e, _ in find_episodes(y): pos[s:e] = np.arange(e - s)
    for seed in [int(a) for a in sys.argv[2:]]:
        t0 = time.time()
        Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
        tr, te = B.split_idx(y, seed); ptr, pte = pos[tr], pos[te]
        sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        rng = np.random.default_rng(seed); bg = Xtr_s[rng.choice(len(Xtr_s), 30, replace=False)]
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4, max_iter=300,
                            random_state=seed, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15).fit(Xtr_s, ytr)
        pred = mlp.predict(Xte_s)
        mu0, sd0 = Xtr[ytr == 0].mean(0), Xtr[ytr == 0].std(0) + 1e-12
        res = {"seed": seed, "shap": {}, "ms": {}}
        for m in (0, 20, 60, 120):
            srng = np.random.default_rng(seed); sh, ms = {}, {}
            for f in F:
                A = Xtr[(ytr == f) & (ptr >= m)]
                ms[f] = {"rank": ranking(np.abs(A.mean(0) - mu0) / sd0)} if len(A) else None
                idx = np.where((yte == f) & (pred == f) & (pte >= m))[0]
                if len(idx) < 5: sh[f] = None; continue
                s = srng.choice(idx, size=min(30, len(idx)), replace=False)
                v = as3(shap.Explainer(mlp.predict_proba, bg, algorithm="permutation", seed=0)(Xte_s[s]).values)
                sh[f] = {"n": int(len(s)), "rank": ranking(np.abs(v[:, :, f]).mean(0))}
            res["shap"][str(m)], res["ms"][str(m)] = sh, ms
        json.dump(res, open(os.path.join(ddir, f"out/onset_seed{seed}.json"), "w"))
        print(ddir, seed, f"{time.time() - t0:.0f}s", flush=True)
