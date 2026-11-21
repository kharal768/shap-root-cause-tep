"""Classifier-free rankings recomputed on the test samples, and PCA and Granger setting variants.

Usage: python fairness_test_set.py <root> <seeds>. Output: ROOT/out/fair_seed<k>.json.
Paper: Section 3.5, Supplementary Section S5."""
import json, os, sys, numpy as np
from scipy import stats
import baselines_extra as B
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS as F


def rank(s): return B.ranking(np.asarray(s, float))


def univariate(A, N):
    mu0, sd0 = N.mean(0), N.std(0) + 1e-12
    ms = np.abs(A.mean(0) - mu0) / sd0
    mv = np.maximum(ms, np.abs(np.log((A.std(0) + 1e-12) / sd0)))
    ks = np.array([stats.ks_2samp(A[:, j], N[:, j]).statistic if (A[:, j].std() > 0 or N[:, j].std() > 0) else 0.0
                   for j in range(A.shape[1])])
    return rank(ms), rank(mv), rank(ks)


def pca_model(N, var):
    mu, sd = N.mean(0), N.std(0); sd[sd == 0] = 1
    Z = (N - mu) / sd
    lam, V = np.linalg.eigh(np.cov(Z.T)); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
    a = int(np.searchsorted(np.cumsum(lam) / lam.sum(), var) + 1)
    P, L = V[:, :a], lam[:a]
    Ct = np.eye(Z.shape[1]) - P @ P.T; Dt = P @ np.diag(1 / L) @ P.T
    t2 = np.einsum("ij,jk,ik->i", Z, Dt, Z); spe = np.einsum("ij,jk,ik->i", Z, Ct, Z)
    Phi = Dt / np.percentile(t2, 99) + Ct / np.percentile(spe, 99)
    return mu, sd, Phi, a


def rbc_rank(A, model):
    mu, sd, Phi, _ = model
    Z = (A - mu) / sd
    return rank(((Z @ Phi) ** 2 / np.diag(Phi)).mean(0))


if __name__ == "__main__":
    ddir = sys.argv[1]
    d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True)
    X, y = d["X"].astype(float), d["y"]
    for seed in [int(a) for a in sys.argv[2:]]:
        Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, 0.3, seed=seed)
        tr, _ = B.split_idx(y, seed)
        N = Xtr[ytr == 0]
        models = {v: pca_model(N, v) for v in (0.80, 0.90, 0.95)}
        res = {"seed": seed, "n_components": {str(v): m[3] for v, m in models.items()}}
        for k in ("ms_test", "mv_test", "ks_test", "pca_test", "pca80", "pca95",
                  "granger_l1", "granger_l3", "granger_a05"):
            res[k] = {}
        for f in F:
            A = Xte[yte == f]
            ms, mv, ks = univariate(A, N)
            res["ms_test"][f] = {"rank": ms, "n": int(len(A))}
            res["mv_test"][f] = {"rank": mv}; res["ks_test"][f] = {"rank": ks}
            res["pca_test"][f] = {"rank": rbc_rank(A, models[0.90])}
            At = Xtr[ytr == f]
            res["pca80"][f] = {"rank": rbc_rank(At, models[0.80])}
            res["pca95"][f] = {"rank": rbc_rank(At, models[0.95])}
            for key, lag, alpha in (("granger_l1", 1, 0.01), ("granger_l3", 3, 0.01), ("granger_a05", 2, 0.05)):
                B.LAG, B.ALPHA = lag, alpha
                res[key][f] = {"rank": B.granger_rank(X, y, tr, f)}
            B.LAG, B.ALPHA = 2, 0.01
        json.dump(res, open(os.path.join(ddir, f"out/fair_seed{seed}.json"), "w"))
        print(ddir, seed, res["n_components"], flush=True)
