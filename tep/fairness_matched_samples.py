"""Classifier-free statistics on exactly the samples the MLP's SHAP ranking explains; Granger lag by BIC.

Usage: python fairness_matched_samples.py <root> <seeds>. Output: ROOT/out/matched_seed<k>.json.
Paper: Supplementary Section S5."""
import json, os, sys, warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS as F
import baselines_extra as B
import fairness_test_set as FA
warnings.filterwarnings("ignore")


def bic_lag(X, y, tr, f, pmax=4):
    trs = np.zeros(len(y), bool); trs[tr] = True
    mask = trs & (y == f)
    keep = np.where(X[y == 0].std(0) > 0)[0]
    mu, sd = X[trs & (y == 0)][:, keep].mean(0), X[trs & (y == 0)][:, keep].std(0)
    Z = (X[:, keep] - mu) / sd
    b = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    segs = [(s, e) for s, e in zip(b[::2], b[1::2]) if e - s > pmax + 1]
    m = len(keep); out = {}
    for p in range(1, pmax + 1):
        Ys, Xs = [], []
        for s, e in segs:
            seg = Z[s:e]
            Ys.append(seg[pmax:])  # common estimation sample for all orders
            Xs.append(np.hstack([seg[pmax - l:len(seg) - l] for l in range(1, p + 1)]))
        Yt, Xl = np.vstack(Ys), np.vstack(Xs); Xl = np.hstack([Xl, np.ones((len(Xl), 1))])
        Bc, *_ = np.linalg.lstsq(Xl, Yt, rcond=None)
        E = Yt - Xl @ Bc; n = len(Yt)
        sign, logdet = np.linalg.slogdet(E.T @ E / n)
        out[p] = logdet + np.log(n) / n * (p * m * m + m)
    return min(out, key=out.get)


if __name__ == "__main__":
    ddir = sys.argv[1]
    d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True)
    X, y = d["X"], d["y"]; Xf = X.astype(float)
    for seed in [int(a) for a in sys.argv[2:]]:
        Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
        sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4,
                            max_iter=300, random_state=seed, early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=15).fit(Xtr_s, ytr)
        pred = mlp.predict(Xte_s)
        N = Xtr[ytr == 0].astype(float); pca = FA.pca_model(N, 0.90)
        rng = np.random.default_rng(seed)
        res = {"seed": seed, "acc_check": float((pred == yte).mean()),
               "ms_same": {}, "mv_same": {}, "ks_same": {}, "pca_same": {}, "granger_bic": {}, "bic_lag": {}}
        for f in F:
            idx = np.where((yte == f) & (pred == f))[0]
            if len(idx) < 5:
                for k in ("ms_same", "mv_same", "ks_same", "pca_same"): res[k][f] = None
            else:
                s = rng.choice(idx, size=min(30, len(idx)), replace=False)
                A = Xte[s].astype(float)
                ms, mv, ks = FA.univariate(A, N)
                res["ms_same"][f] = {"rank": ms, "n": int(len(s))}; res["mv_same"][f] = {"rank": mv}
                res["ks_same"][f] = {"rank": ks}; res["pca_same"][f] = {"rank": FA.rbc_rank(A, pca)}
            tr, _ = B.split_idx(y, seed)
            p = bic_lag(Xf, y, tr, f); res["bic_lag"][f] = int(p)
            B.LAG, B.ALPHA = p, 0.01
            res["granger_bic"][f] = {"rank": B.granger_rank(Xf, y, tr, f)}
            B.LAG = 2
        json.dump(res, open(os.path.join(ddir, f"out/matched_seed{seed}.json"), "w"))
        print(ddir, seed, res["acc_check"], res["bic_lag"], flush=True)
