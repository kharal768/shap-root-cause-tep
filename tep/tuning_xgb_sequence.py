"""Validation-selected XGBoost depth and trees and 1D-CNN/GRU epochs, refitted, with their SHAP rankings.

Usage: python tuning_xgb_sequence.py <root> xgb|cnn|gru <seeds>. Output: ROOT/out/tune_<part>_seed<k>.json.
Paper: Supplementary Section S5."""
import json, os, sys, time, warnings
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS
warnings.filterwarnings("ignore")


def split_idx(y, seed, frac=0.3, sub=None):
    idx = np.arange(len(y))[:, None]
    a = episode_train_test_split(idx, y, frac, seed=seed)
    return a[0][:, 0], a[2][:, 0]


def run_xgb(X, y, seed):
    import xgboost as xgb, shap
    from sklearn.preprocessing import StandardScaler
    from run_mlp_xgboost import explain
    Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
    Xf, yf, Xv, yv, _ = episode_train_test_split(Xtr, ytr, test_frac=0.2, seed=100 + seed)
    s = StandardScaler().fit(Xf); A, V = s.transform(Xf), s.transform(Xv)
    val = {}
    for dp in (4, 6, 8):
        m = xgb.XGBClassifier(n_estimators=400, max_depth=dp, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                              random_state=seed, n_jobs=1, tree_method="hist", early_stopping_rounds=20,
                              eval_metric="mlogloss").fit(A, yf, eval_set=[(V, yv)], verbose=False)
        val[dp] = (f1_score(yv, m.predict(V), average="macro"), int(m.best_iteration) + 1)
    dp = max(val, key=lambda k: val[k][0]); nt = val[dp][1]
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    m = xgb.XGBClassifier(n_estimators=nt, max_depth=dp, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                          random_state=seed, n_jobs=1, tree_method="hist").fit(Xtr_s, ytr)
    p = m.predict(Xte_s)
    return {"val": {str(k): v for k, v in val.items()}, "best": [dp, nt],
            "acc": accuracy_score(yte, p), "macro_f1": f1_score(yte, p, average="macro"),
            "shap": explain(None, None, Xte_s, yte, p, np.random.default_rng(seed), tree=shap.TreeExplainer(m))}


def run_seq(X, y, seed, arch):
    import torch, torch.nn as nn
    import run_sequence_models as S
    torch.set_num_threads(1)
    S.ARCH = arch
    tr_idx, te_idx = split_idx(y, seed)
    yte = y[te_idx]
    a = episode_train_test_split(tr_idx[:, None], y[tr_idx], 0.2, seed=100 + seed)
    fit_idx, val_idx = a[0][:, 0], a[2][:, 0]
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-9
    Xs = ((X - mu) / sd).astype(np.float32)
    lossf = nn.CrossEntropyLoss()

    def train(idx, epochs, eval_idx=None):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        model = S.GRU() if arch == "gru" else S.CNN()
        opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
        hist, best, wait = [], -1, 0
        for ep in range(epochs):
            model.train()
            perm = rng.permutation(len(idx))
            for i in range(0, len(perm), 512):
                b = idx[perm[i:i + 512]]
                xb = torch.as_tensor(S.windows(Xs, b)); yb = torch.as_tensor(y[b], dtype=torch.long)
                opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
            if eval_idx is not None:
                model.eval()
                f = f1_score(y[eval_idx], S.proba(model, S.windows(Xs, eval_idx)).argmax(1), average="macro")
                hist.append(f)
                if f > best: best, wait = f, 0
                else:
                    wait += 1
                    if wait >= 3: break
        model.eval()
        return model, rng, hist

    _, _, hist = train(fit_idx, 12, val_idx)
    n_ep = int(np.argmax(hist)) + 1
    model, rng, _ = train(tr_idx, n_ep)
    p = S.proba(model, S.windows(Xs, te_idx)).argmax(1)
    bg = S.windows(Xs, rng.choice(tr_idx, 30, replace=False))
    sh = {}
    for f in VALIDATABLE_FAULTS:
        ok = np.where((yte == f) & (p == f))[0]
        if len(ok) < 5: sh[f] = None; continue
        s = np.random.default_rng(seed).choice(ok, size=min(30, len(ok)), replace=False)
        xs = S.windows(Xs, te_idx[s])
        v = np.mean([np.abs(S.channel_shap(model, xs[k], bg, f, rng)) for k in range(len(s))], axis=0)
        sh[f] = {"n": int(len(s)), "rank": S.ranking(v)}
    return {"val_hist": hist, "best_epochs": n_ep, "acc": accuracy_score(yte, p),
            "macro_f1": f1_score(yte, p, average="macro"), "shap": sh}


if __name__ == "__main__":
    ddir, part = sys.argv[1], sys.argv[2]
    d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True)
    X, y = d["X"], d["y"]
    for seed in [int(a) for a in sys.argv[3:]]:
        t = time.time()
        res = run_xgb(X, y, seed) if part == "xgb" else run_seq(X, y, seed, part)
        res["seed"] = seed
        json.dump(res, open(os.path.join(ddir, f"out/tune_{part}_seed{seed}.json"), "w"), default=float)
        print(ddir, part, seed, res.get("best", res.get("best_epochs")), round(res["acc"], 4), f"{time.time() - t:.0f}s", flush=True)
