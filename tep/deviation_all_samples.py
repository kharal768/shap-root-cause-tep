"""Model-free deviation ranking on all test samples of each fault, without the classifier's selection.

Usage: python deviation_all_samples.py <root>. Output: ROOT/out/devall_seed<k>.json. Paper: Section 4.4."""
import json, os, sys, numpy as np
from sklearn.preprocessing import StandardScaler
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS as F
from ablation import ranking
ddir = sys.argv[1]; d = np.load(os.path.join(ddir, "data/tep_clean.npz"), allow_pickle=True); X, y = d["X"], d["y"]
for seed in range(10):
    Xtr, ytr, Xte, yte, _ = episode_train_test_split(X, y, test_frac=0.3, seed=seed)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    bg = Xtr_s[np.random.default_rng(seed).choice(len(Xtr_s), 30, replace=False)]
    out = {f: {"rank": ranking(np.abs(Xte_s[yte == f] - bg.mean(0)).mean(0))} for f in F}
    json.dump(out, open(os.path.join(ddir, f"out/devall_seed{seed}.json"), "w"))
