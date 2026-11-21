"""Further comparison rankings: Kolmogorov-Smirnov and Granger causality (classfree), SHAP of an untrained
MLP (mlprand), and Integrated Gradients for the 1D-CNN and GRU (ig_cnn, ig_gru).

Usage: python baselines_extra.py <root> <part> <seeds>. Output: ROOT/out/base2_<part>_seed<k>.json.
Paper: Tables 3 and 5, Supplementary Table S8."""
import json, os, sys, time, warnings, copy
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS

LAG, ALPHA = 2, 0.01


def ranking(score):
    order = np.argsort(score)[::-1]
    r = np.empty(len(order), int); r[order] = np.arange(1, len(order) + 1)
    return r.tolist()


def split_idx(y, seed):
    idx = np.arange(len(y))[:, None]
    a = episode_train_test_split(idx, y, 0.3, seed=seed)
    return a[0][:, 0], a[2][:, 0]


def ks_rank(X, y, tr, f):
    A, B = X[tr][y[tr] == f], X[tr][y[tr] == 0]
    d = np.array([stats.ks_2samp(A[:, j], B[:, j]).statistic if B[:, j].std() > 0 or A[:, j].std() > 0 else 0.0
                  for j in range(X.shape[1])])
    return ranking(d)


def granger_rank(X, y, tr, f):
    trs = np.zeros(len(y), bool); trs[tr] = True
    mask = trs & (y == f)
    keep = np.where(X[y == 0].std(0) > 0)[0]
    mu, sd = X[trs & (y == 0)][:, keep].mean(0), X[trs & (y == 0)][:, keep].std(0)
    Z = (X[:, keep] - mu) / sd
    # contiguous segments of fault-f training samples
    b = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    rows_y, rows_x = [], []
    for s, e in zip(b[::2], b[1::2]):
        if e - s <= LAG + 1: continue
        seg = Z[s:e]
        rows_y.append(seg[LAG:])
        rows_x.append(np.hstack([seg[LAG - l:len(seg) - l] for l in range(1, LAG + 1)]))
    Yt, Xl = np.vstack(rows_y), np.vstack(rows_x)
    Xl = np.hstack([Xl, np.ones((len(Xl), 1))])
    n, p = Xl.shape; m = len(keep)
    G = Xl.T @ Xl; Ginv = np.linalg.pinv(G)
    B = Ginv @ (Xl.T @ Yt)                       # (p, m)
    rss = ((Yt - Xl @ B) ** 2).sum(0)            # (m,)
    F = np.zeros((m, m))                         # F[src, tgt]
    for j in range(m):
        cols = [l * m + j for l in range(LAG)]
        Vinv = np.linalg.pinv(Ginv[np.ix_(cols, cols)])
        bj = B[cols]                             # (LAG, m)
        drss = np.einsum("lt,lk,kt->t", bj, Vinv, bj)
        F[j] = (drss / LAG) / (rss / (n - p))
    np.fill_diagonal(F, 0)
    pv = stats.f.sf(F, LAG, n - p); sig = (pv < ALPHA); np.fill_diagonal(sig, False)
    net = sig.sum(1) - sig.sum(0)
    score_k = net + 1e-3 * np.log1p(F).sum(1) / np.log1p(F).sum()
    score = np.full(X.shape[1], -1e9); score[keep] = score_k
    return ranking(score)


def run_classfree(X, y, seed):
    tr, _ = split_idx(y, seed)
    return {"ks": {f: {"rank": ks_rank(X, y, tr, f)} for f in VALIDATABLE_FAULTS},
            "granger": {f: {"rank": granger_rank(X, y, tr, f)} for f in VALIDATABLE_FAULTS}}


def run_mlprand(X, y, seed):
    import shap
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    rng = np.random.default_rng(seed)
    bg = Xtr_s[rng.choice(len(Xtr_s), 30, replace=False)]
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4,
                        max_iter=300, random_state=seed, early_stopping=True,
                        validation_fraction=0.1, n_iter_no_change=15).fit(Xtr_s, ytr)
    pred = mlp.predict(Xte_s)
    rnd = copy.deepcopy(mlp); g = np.random.default_rng(10_000 + seed)
    for i, Wm in enumerate(rnd.coefs_):
        bound = np.sqrt(6.0 / (Wm.shape[0] + Wm.shape[1]))
        rnd.coefs_[i] = g.uniform(-bound, bound, Wm.shape)
        rnd.intercepts_[i] = g.uniform(-bound, bound, rnd.intercepts_[i].shape)
    out = {}
    srng = np.random.default_rng(seed)
    for f in VALIDATABLE_FAULTS:
        idx = np.where((yte == f) & (pred == f))[0]
        if len(idx) < 5: out[f] = None; continue
        s = srng.choice(idx, size=min(30, len(idx)), replace=False)
        e = shap.Explainer(rnd.predict_proba, bg, algorithm="permutation", seed=0)
        v = e(Xte_s[s]).values[:, :, f]
        out[f] = {"n": int(len(s)), "rank": ranking(np.abs(v).mean(0))}
    return {"mlprand": out, "mlp_acc_check": accuracy_score(yte, pred),
            "mlprand_acc": accuracy_score(yte, rnd.predict(Xte_s))}


def run_ig(X, y, seed, arch):
    import torch
    import run_sequence_models as S
    S.ARCH = arch
    _, ytr, _, yte, _ = episode_train_test_split(X, y, 0.3, seed=seed)
    tr_idx, te_idx = split_idx(y, seed)
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-9
    Xs = ((X - mu) / sd).astype(np.float32)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = S.GRU() if arch == "gru" else S.CNN()
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    for ep in range(S.EPOCHS):
        perm = rng.permutation(len(tr_idx))
        for i in range(0, len(perm), 512):
            b = tr_idx[perm[i:i + 512]]
            xb = torch.as_tensor(S.windows(Xs, b)); yb = torch.as_tensor(y[b], dtype=torch.long)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    model.eval()
    p = S.proba(model, S.windows(Xs, te_idx)).argmax(1)
    acc = float((p == yte).mean())
    steps = 32; alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    out = {}
    for f in VALIDATABLE_FAULTS:
        ok = np.where((yte == f) & (p == f))[0]
        if len(ok) < 5: out[f] = None; continue
        s = np.random.default_rng(seed).choice(ok, size=min(30, len(ok)), replace=False)
        xs = torch.as_tensor(S.windows(Xs, te_idx[s]))          # (n, W, 52), baseline = 0
        n = len(s)
        path = (alphas[:, None, None, None] * xs[None]).reshape(steps * n, *xs.shape[1:]).requires_grad_(True)
        prob = torch.softmax(model(path), 1)[:, f].sum()
        grad, = torch.autograd.grad(prob, path)
        ig = (xs * grad.reshape(steps, n, *xs.shape[1:]).mean(0)).detach().numpy()   # (n, W, 52)
        sc = np.abs(ig.sum(1)).mean(0)
        out[f] = {"n": int(n), "rank": ranking(sc)}
    return {f"ig_{arch}": out, f"{arch}_acc_check": acc}


if __name__ == "__main__":
    ddir, part = sys.argv[1], sys.argv[2]
    d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True)
    X, y = d["X"].astype(float), d["y"]
    for seed in [int(a) for a in sys.argv[3:]]:
        t = time.time()
        fn = os.path.join(ddir, f"out/base2_{part}_seed{seed}.json")
        if part == "classfree": res = run_classfree(X, y, seed)
        elif part == "mlprand": res = run_mlprand(X, y, seed)
        else: res = run_ig(X, y, seed, part.split("_")[1])
        res["seed"] = seed
        json.dump(res, open(fn, "w"))
        print(ddir, part, seed, f"{time.time() - t:.0f}s",
              {k: v for k, v in res.items() if "acc" in k}, flush=True)
