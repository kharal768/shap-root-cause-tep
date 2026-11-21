"""Conditional Shapley values under a multivariate Gaussian model of the inputs (Aas et al., 2021).
Imported by run_reactor.py."""
import json, os, sys, time, warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from data_split import episode_train_test_split
warnings.filterwarnings("ignore")
P, K = 10, 20

def cond_phi(fp, x, mu, C, f, rng):
    d = len(x); phi = np.zeros(d)
    for _ in range(P // 2):
        p = rng.permutation(d)
        for perm in (p, p[::-1]):
            rows = []
            for j in range(d + 1):
                S = perm[:j]; T = perm[j:]
                if len(T) == 0: rows.append(x[None, :]); continue
                if len(S) == 0: m, Sig = mu[T], C[np.ix_(T, T)]
                else:
                    A = np.linalg.solve(C[np.ix_(S, S)], C[np.ix_(S, T)]).T
                    m = mu[T] + A @ (x[S] - mu[S]); Sig = C[np.ix_(T, T)] - A @ C[np.ix_(S, T)]
                w, V = np.linalg.eigh((Sig + Sig.T) / 2); L = V * np.sqrt(np.clip(w, 0, None))
                z = np.empty((K, d)); z[:, S] = x[S]; z[:, T] = m + rng.standard_normal((K, len(T))) @ L.T
                rows.append(z)
            Z = np.vstack([np.repeat(r, K, 0) if len(r) == 1 else r for r in rows])
            v = fp(Z)[:, f].reshape(d + 1, K).mean(1)
            phi[perm] += np.diff(v)
    return phi / P

