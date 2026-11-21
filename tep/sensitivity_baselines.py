"""Sensitivity of the PCA contribution (retained variance x control limit) and Granger ranking (level).

Usage: python sensitivity_baselines.py <root> <seeds>. Output: ROOT/out/sensbase_seed<k>.json.
Paper: Table 6, Supplementary Section S8."""
import json, os, sys
import numpy as np
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS as F
import baselines_extra as B


def rank(s): return B.ranking(np.asarray(s, float))


def pca_model(N, var, pct):
    mu, sd = N.mean(0), N.std(0); sd[sd == 0] = 1
    Z = (N - mu) / sd
    lam, V = np.linalg.eigh(np.cov(Z.T)); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
    a = int(np.searchsorted(np.cumsum(lam) / lam.sum(), var) + 1)
    P, L = V[:, :a], lam[:a]
    Ct = np.eye(Z.shape[1]) - P @ P.T; Dt = P @ np.diag(1 / L) @ P.T
    t2 = np.einsum("ij,jk,ik->i", Z, Dt, Z); spe = np.einsum("ij,jk,ik->i", Z, Ct, Z)
    return mu, sd, Dt / np.percentile(t2, pct) + Ct / np.percentile(spe, pct), a


if __name__ == "__main__":
    ddir = sys.argv[1]
    d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True)
    X, y = d["X"].astype(float), d["y"]
    for seed in [int(a) for a in sys.argv[2:]]:
        Xtr, ytr, _, _, _ = episode_train_test_split(X, y, 0.3, seed=seed)
        tr, _ = B.split_idx(y, seed); N = Xtr[ytr == 0]
        res = {"seed": seed, "pca": {}, "granger": {}, "ncomp": {}}
        for var in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
            for pct in (95, 99, 99.9):
                mu, sd, Phi, a = pca_model(N, var, pct); res["ncomp"][str(var)] = a
                res["pca"][f"{var}|{pct}"] = {f: {"rank": rank(((((Xtr[ytr == f] - mu) / sd) @ Phi) ** 2 / np.diag(Phi)).mean(0))} for f in F}
        for alpha in (0.001, 0.01, 0.05, 0.1):
            B.LAG, B.ALPHA = 2, alpha
            res["granger"][str(alpha)] = {f: {"rank": B.granger_rank(X, y, tr, f)} for f in F}
        json.dump(res, open(os.path.join(ddir, f"out/sensbase_seed{seed}.json"), "w"))
        print(ddir, seed, res["ncomp"], flush=True)
