"""Conditional Shapley values under a multivariate Gaussian model of the inputs (Aas et al., 2021) for the
MLP, on the same split, samples and class output as the permutation explainer.

Usage: python conditional_shap.py <root> <seeds>. Output: ROOT/out/condshap_seed<k>.json.
Paper: Section 4.3, Supplementary Table S8."""
import json, os, sys, time, warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS as F
from ablation import ranking
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

if __name__ == "__main__":
    ddir = sys.argv[1]
    d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True); X, y = d["X"], d["y"]
    for seed in map(int, sys.argv[2:]):
        t0 = time.time()
        Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
        sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        rng = np.random.default_rng(seed); bg = Xtr_s[rng.choice(len(Xtr_s), 30, replace=False)]
        _ = Xtr_s[np.random.default_rng(1000 + seed).choice(np.where(ytr == 0)[0], 30, replace=False)]
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4, max_iter=300, random_state=seed,
                            early_stopping=True, validation_fraction=0.1, n_iter_no_change=15).fit(Xtr_s, ytr)
        pred = clf.predict(Xte_s); mu = Xtr_s.mean(0); C = np.cov(Xtr_s, rowvar=False) + 1e-3 * np.eye(Xtr_s.shape[1])  # ridge: two channels are constant
        srng = np.random.default_rng(seed); arng = np.random.default_rng(2000 + seed); crng = np.random.default_rng(5000 + seed)
        out = {}
        for f in F:
            idx = np.where((yte == f) & (pred == f))[0]
            alli = np.where(yte == f)[0]; arng.choice(alli, size=min(30, len(alli)), replace=False)
            if len(idx) < 5: out[f] = None; continue
            s = srng.choice(idx, size=min(30, len(idx)), replace=False)
            v = np.mean([np.abs(cond_phi(clf.predict_proba, Xte_s[i], mu, C, f, crng)) for i in s], 0)
            out[f] = {"n": int(len(s)), "rank": ranking(v)}
        json.dump({"seed": seed, "acc": float((pred == yte).mean()), "cond": out}, open(os.path.join(ddir, f"out/condshap_seed{seed}.json"), "w"))
        print(ddir, seed, f"{time.time() - t0:.0f}s", flush=True)
