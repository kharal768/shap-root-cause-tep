"""One-at-a-time sensitivity of the MLP + permutation SHAP pipeline (explained samples, background,
noise, missing values, training fraction).

Usage: python sensitivity_pipeline.py <root> <seeds>. Output: ROOT/out/sens_seed<k>.json. Paper: Supplementary Section S8."""
import json, os, sys, time, warnings
import numpy as np, shap
from sklearn.neural_network import MLPClassifier
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


def mk(seed):
    return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4, max_iter=300,
                         random_state=seed, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15)


def explain(clf, bg, Xs, yte, pred, seed, n=30, ms_from=None, mu0=None, sd0=None):
    out, ms = {}, {}
    srng = np.random.default_rng(seed)
    for f in F:
        idx = np.where((yte == f) & (pred == f))[0]
        if len(idx) < 5: out[f] = None; ms[f] = None; continue
        s = srng.choice(idx, size=min(n, len(idx)), replace=False)
        v = as3(shap.Explainer(clf.predict_proba, bg, algorithm="permutation", seed=0)(Xs[s]).values)
        out[f] = {"n": int(len(s)), "rank": ranking(np.abs(v[:, :, f]).mean(0))}
        if ms_from is not None:
            ms[f] = {"rank": ranking(np.abs(ms_from[s].mean(0) - mu0) / (sd0 + 1e-12))}
    return out, ms


if __name__ == "__main__":
    ddir = sys.argv[1]
    d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True)
    X, y = d["X"], d["y"]
    for seed in [int(a) for a in sys.argv[2:]]:
        t0 = time.time()
        Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
        sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        mu0, sd0 = Xtr[ytr == 0].mean(0), Xtr[ytr == 0].std(0)
        rng = np.random.default_rng(seed); bg = Xtr_s[rng.choice(len(Xtr_s), 30, replace=False)]
        mlp = mk(seed).fit(Xtr_s, ytr); pred = mlp.predict(Xte_s)
        res = {"seed": seed}
        # explained samples (nested)
        nrng = np.random.default_rng(3000 + seed); sel = {}
        for f in F:
            idx = np.where((yte == f) & (pred == f))[0]
            sel[f] = nrng.choice(idx, size=min(100, len(idx)), replace=False) if len(idx) >= 5 else None
        allv = {f: (as3(shap.Explainer(mlp.predict_proba, bg, algorithm="permutation", seed=0)(Xte_s[s]).values)[:, :, f]
                    if s is not None else None) for f, s in sel.items()}
        res["n_explained"] = {str(n): {f: ({"n": int(min(n, len(allv[f]))), "rank": ranking(np.abs(allv[f][:n]).mean(0))}
                                           if allv[f] is not None and len(allv[f]) >= min(n, 5) else None) for f in F}
                              for n in (5, 10, 30, 100)}
        # background size
        res["bg_size"] = {}
        for b in (10, 30, 100):
            bgb = Xtr_s[np.random.default_rng(4000 + seed).choice(len(Xtr_s), b, replace=False)]
            res["bg_size"][str(b)] = explain(mlp, bgb, Xte_s, yte, pred, seed)[0]
        # background composition
        res["bg_normal"] = {}
        nrm, flt = np.where(ytr == 0)[0], np.where(ytr != 0)[0]
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            g = np.random.default_rng(4500 + seed); k = int(round(30 * q))
            ids = np.concatenate([g.choice(nrm, k, replace=False), g.choice(flt, 30 - k, replace=False)])
            res["bg_normal"][str(q)] = explain(mlp, Xtr_s[ids], Xte_s, yte, pred, seed)[0]
        # noise and missing
        prng = np.random.default_rng(5000 + seed)
        res["noise"], res["missing"], res["noise_ms"], res["missing_ms"], res["acc"] = {}, {}, {}, {}, {}
        for s_ in (0.1, 0.3, 0.5, 0.7, 0.9):
            Xn = Xte + prng.normal(0, 1, Xte.shape) * (s_ * sd0); Xn_s = sc.transform(Xn); pn = mlp.predict(Xn_s)
            res["noise"][str(s_)], res["noise_ms"][str(s_)] = explain(mlp, bg, Xn_s, yte, pn, seed, ms_from=Xn, mu0=mu0, sd0=sd0)
            res["acc"][f"noise{s_}"] = float((pn == yte).mean())
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            Xm = Xte.astype(float).copy(); m = prng.random(Xm.shape) < p
            Xm[m] = np.broadcast_to(sc.mean_, Xm.shape)[m]; Xm_s = sc.transform(Xm); pm = mlp.predict(Xm_s)
            res["missing"][str(p)], res["missing_ms"][str(p)] = explain(mlp, bg, Xm_s, yte, pm, seed, ms_from=Xm, mu0=mu0, sd0=sd0)
            res["acc"][f"missing{p}"] = float((pm == yte).mean())
        # training fraction
        res["train_frac"], res["train_frac_ms"] = {}, {}
        g = np.random.default_rng(7000 + seed)
        for fr in (0.1, 0.3, 0.5, 0.7, 0.9):
            keep = np.sort(np.concatenate([g.choice(np.where(ytr == c)[0], max(1, int(round(fr * (ytr == c).sum()))), replace=False)
                                           for c in np.unique(ytr)]))
            Xa, ya = Xtr[keep], ytr[keep]
            sca = StandardScaler().fit(Xa); Xa_s, Xt_s = sca.transform(Xa), sca.transform(Xte)
            bga = Xa_s[np.random.default_rng(seed).choice(len(Xa_s), 30, replace=False)]
            m_ = mk(seed).fit(Xa_s, ya); pa = m_.predict(Xt_s)
            res["train_frac"][str(fr)] = explain(m_, bga, Xt_s, yte, pa, seed)[0]
            mu, sd = Xa[ya == 0].mean(0), Xa[ya == 0].std(0) + 1e-12
            res["train_frac_ms"][str(fr)] = {f: {"rank": ranking(np.abs(Xa[ya == f].mean(0) - mu) / sd)} for f in F}
            res["acc"][f"train{fr}"] = float((pa == yte).mean())
        json.dump(res, open(os.path.join(ddir, f"out/sens_seed{seed}.json"), "w"))
        print(ddir, seed, f"{time.time() - t0:.0f}s", flush=True)
