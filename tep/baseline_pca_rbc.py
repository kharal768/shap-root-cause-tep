"""PCA reconstruction-based contribution (Alcala & Qin, 2009) retaining 90% of the variance.

Usage: python baseline_pca_rbc.py <data_dir> <out_dir>. Output: pca_seed<k>.json. Paper: Table 3."""
import json, sys, numpy as np
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS

data_dir, out_dir = sys.argv[1], sys.argv[2]
d = np.load(f"{data_dir}/tep_clean.npz", allow_pickle=True); X, y = d["X"], d["y"]


def ranking(score):
    order = np.argsort(score)[::-1]
    r = np.empty(len(order), int); r[order] = np.arange(1, len(order) + 1)
    return r.tolist()


for seed in range(10):
    Xtr, ytr, _, _, _ = episode_train_test_split(X, y, 0.3, seed=seed)
    N = Xtr[ytr == 0]; mu, sd = N.mean(0), N.std(0); sd[sd == 0] = 1
    Z = (N - mu) / sd
    lam, V = np.linalg.eigh(np.cov(Z.T)); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
    a = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.90) + 1)
    P, L = V[:, :a], lam[:a]
    Ct = np.eye(Z.shape[1]) - P @ P.T
    Dt = P @ np.diag(1 / L) @ P.T
    t2 = np.einsum("ij,jk,ik->i", Z, Dt, Z); spe = np.einsum("ij,jk,ik->i", Z, Ct, Z)
    tau2, delta2 = np.percentile(t2, 99), np.percentile(spe, 99)
    Phi = Dt / tau2 + Ct / delta2
    res = {"seed": seed, "n_components": a}
    for f in VALIDATABLE_FAULTS:
        F = (Xtr[ytr == f] - mu) / sd
        rbc = (F @ Phi) ** 2 / np.diag(Phi)  # (n, 52): (xi_i' Phi x)^2 / (xi_i' Phi xi_i)
        res[f] = {"rank": ranking(rbc.mean(0))}
    json.dump(res, open(f"{out_dir}/pca_seed{seed}.json", "w"))
    print(seed, a)
