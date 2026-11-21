"""Standardized mean shift of every channel and rank of the documented channel for the two sensor faults at
larger fault sizes. Output: ../out/cstr_magnitude.json. Paper: Section 4.7, Supplementary Section S12."""
import numpy as np, json, re
from scipy.stats import ks_2samp
src = open("simulate.py").read().split('if __name__')[0]
out = {}
for tag, a, b in [("bias1.5", "1.5", "0.15"), ("bias4", "4.0", "0.15"), ("bias8", "8.0", "0.15"), ("cap0.4", "1.5", "0.4"), ("cap0.8", "1.5", "0.8")]:
    s = src.replace("(1.5 if f == 10 else 0)", f"({a} if f == 10 else 0)").replace("0.0005 * DT, 0.15)", f"0.0005 * DT * {float(b)/0.15}, {b})")
    g = {}; exec(s, g); X, y = g["simulate"](); N = g["NAMES"]
    N0 = X[y == 0]; mu, sd = N0.mean(0), N0.std(0); out[tag] = {}
    for f, ch in ((10, "T"), (11, "h")):
        F = X[y == f]; ms = np.abs(F.mean(0) - mu) / sd; ks = np.array([ks_2samp(F[:, j], N0[:, j]).statistic for j in range(len(N))])
        j = N.index(ch); out[tag][f] = {"shift": dict(zip(N, ms.round(2).tolist())), "rank_ms": int((ms > ms[j]).sum() + 1), "rank_ks": int((ks > ks[j]).sum() + 1)}
        print(tag, f, ch, "rank ms", out[tag][f]["rank_ms"], "ks", out[tag][f]["rank_ks"], " ".join(f"{n}:{v:.2f}" for n, v in zip(N, ms)))
json.dump(out, open("../out/cstr_magnitude.json", "w"), indent=1)
