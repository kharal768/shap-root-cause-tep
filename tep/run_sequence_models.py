"""1D-CNN and GRU on causal 20-sample windows, with channel-level permutation Shapley values.

Usage: python run_sequence_models.py cnn|gru <seeds>. Input: ROOT/data/tep_clean.npz.
Output: ROOT/out/<arch>_seed<k>.json. Paper: Table 3."""
import os
import json, sys, time, warnings
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS
ROOT = os.environ.get("TEP_ROOT", "../runs/revised")  # folder with data/tep_clean.npz and out/
warnings.filterwarnings("ignore")
torch.set_num_threads(2)
W, EPOCHS, N_PERM = 20, 6, 6
ARCH = "cnn"


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.GRU(52, 64, batch_first=True); self.o = nn.Linear(64, 21)

    def forward(self, x):  # x: (n, W, 52)
        return self.o(self.g(x)[0][:, -1])


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(52, 64, 5, padding=2), nn.ReLU(), nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
                               nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 21))

    def forward(self, x):  # x: (n, W, 52)
        return self.f(x.transpose(1, 2))


def windows(Xs, idx):
    off = np.arange(-W + 1, 1)
    j = np.clip(idx[:, None] + off[None, :], 0, None)
    return Xs[j]  # (n, W, 52)


def ranking(score):
    order = np.argsort(score)[::-1]
    r = np.empty(len(order), int); r[order] = np.arange(1, len(order) + 1)
    return r.tolist()


def proba(model, x):
    with torch.no_grad():
        out = []
        for i in range(0, len(x), 4096):
            out.append(torch.softmax(model(torch.as_tensor(x[i:i + 4096])), 1).numpy())
        return np.concatenate(out)


def channel_shap(model, x, bg, f, rng):
    """Mean over background of permutation-sampled Shapley values for class f; x: (W,52)."""
    phi = np.zeros(52)
    for _ in range(N_PERM // 2):
        p = rng.permutation(52)
        for perm in (p, p[::-1]):
            cur = bg.copy()  # (B, W, 52)
            batch = [cur.copy()]
            for c in perm:
                cur[:, :, c] = x[:, c]; batch.append(cur.copy())
            pr = proba(model, np.concatenate(batch).astype(np.float32))[:, f].reshape(53, len(bg)).mean(1)
            phi[perm] += np.diff(pr)
    return phi / N_PERM


def run(seed, X, y):
    _, ytr, _, yte, _ = episode_train_test_split(X, y, 0.3, seed=seed)
    # recover indices
    idx_all = np.arange(len(y))
    tr_idx = episode_train_test_split(idx_all[:, None], y, 0.3, seed=seed)[0][:, 0]
    te_idx = episode_train_test_split(idx_all[:, None], y, 0.3, seed=seed)[2][:, 0]
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-9
    Xs = ((X - mu) / sd).astype(np.float32)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = GRU() if ARCH == "gru" else CNN(); opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(); t = time.time()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(tr_idx))
        for i in range(0, len(perm), 512):
            b = tr_idx[perm[i:i + 512]]
            xb = torch.as_tensor(windows(Xs, b)); yb = torch.as_tensor(y[b], dtype=torch.long)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    model.eval(); ttrain = time.time() - t
    p = proba(model, windows(Xs, te_idx)).argmax(1)
    rep = classification_report(yte, p, output_dict=True, zero_division=0)
    bg = windows(Xs, rng.choice(tr_idx, 30, replace=False))
    sh = {}
    for f in VALIDATABLE_FAULTS:
        ok = np.where((yte == f) & (p == f))[0]
        if len(ok) < 5: sh[f] = None; continue
        s = np.random.default_rng(seed).choice(ok, size=min(30, len(ok)), replace=False)
        xs = windows(Xs, te_idx[s])
        v = np.mean([np.abs(channel_shap(model, xs[k], bg, f, rng)) for k in range(len(s))], axis=0)
        sh[f] = {"n": int(len(s)), "rank": ranking(v)}
    res = {"seed": seed, ARCH: {"acc": accuracy_score(yte, p), "macro_f1": f1_score(yte, p, average="macro"),
           "f1": {c: rep[str(c)]["f1-score"] for c in range(21) if str(c) in rep}, "shap": sh,
           "train_s": ttrain, "total_s": time.time() - t}}
    json.dump(res, open(ROOT + f"/out/{ARCH}_seed{seed}.json", "w"))
    print(f"seed {seed}: {ARCH} acc {res[ARCH]['acc']:.4f} macroF1 {res[ARCH]['macro_f1']:.3f} "
          f"(train {ttrain:.0f}s, total {res[ARCH]['total_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    d = np.load(ROOT + "/data/tep_clean.npz", allow_pickle=True)
    args = sys.argv[1:]
    if args and args[0] in ("cnn", "gru"):
        ARCH = args.pop(0)
    for s in [int(a) for a in args]:
        run(s, d["X"], d["y"])
