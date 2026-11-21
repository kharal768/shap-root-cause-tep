"""Model-free deviation ranking of the 1D-CNN's and GRU's explained windows from the background mean.

Usage: python sequence_deviation.py <root> cnn|gru <seeds>. Output: ROOT/out/seqdev_<arch>_seed<k>.json.
Paper: Section 4.4, Supplementary Table S8."""
import json, sys, numpy as np, torch
import run_sequence_models as S
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS
def run(seed, X, y, arch, out):
    S.ARCH = arch
    _, ytr, _, yte, _ = episode_train_test_split(X, y, 0.3, seed=seed)
    ia = np.arange(len(y)); tr = episode_train_test_split(ia[:, None], y, 0.3, seed=seed)[0][:, 0]; te = episode_train_test_split(ia[:, None], y, 0.3, seed=seed)[2][:, 0]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xs = ((X - mu) / sd).astype(np.float32)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = S.GRU() if arch == "gru" else S.CNN(); opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4); lf = torch.nn.CrossEntropyLoss()
    for ep in range(S.EPOCHS):
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), 512):
            b = tr[perm[i:i + 512]]; opt.zero_grad(); lf(model(torch.as_tensor(S.windows(Xs, b))), torch.as_tensor(y[b], dtype=torch.long)).backward(); opt.step()
    model.eval(); p = S.proba(model, S.windows(Xs, te)).argmax(1)
    bg = S.windows(Xs, rng.choice(tr, 30, replace=False)); m = bg.mean((0, 1))
    dev, devall = {}, {}
    for f in VALIDATABLE_FAULTS:
        allw = S.windows(Xs, te[yte == f]); devall[f] = {"rank": S.ranking(np.abs(allw - m).mean((0, 1)))}
        ok = np.where((yte == f) & (p == f))[0]
        if len(ok) < 5: dev[f] = None; continue
        s = np.random.default_rng(seed).choice(ok, size=min(30, len(ok)), replace=False)
        dev[f] = {"rank": S.ranking(np.abs(S.windows(Xs, te[s]) - m).mean((0, 1)))}
    json.dump({"seed": seed, "acc": float((p == yte).mean()), "deviation": dev, "dev_all": devall}, open(f"{out}/seqdev_{arch}_seed{seed}.json", "w"))
    print(out, arch, seed, round(float((p == yte).mean()), 4), flush=True)
if __name__ == "__main__":
    ddir, arch = sys.argv[1], sys.argv[2]
    d = np.load(f"{ddir}/data/tep_clean.npz", allow_pickle=True)
    for s in map(int, sys.argv[3:]): run(s, d["X"], d["y"], arch, f"{ddir}/out")
