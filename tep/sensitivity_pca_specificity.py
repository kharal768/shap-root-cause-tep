"""Hit rates and cross-fault specificity of the PCA contribution at each retained variance.

Usage: python sensitivity_pca_specificity.py <root>. Output: ROOT/out/sens_pca_spec.json. Paper: Section 4.6."""
import json, glob, os, sys
import numpy as np
from fault_metadata import ALL_CHANNEL_NAMES as N, GROUND_TRUTH_CHANNELS as GT
F = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]
ddir = sys.argv[1]; runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ddir, "out/sensbase_seed*.json")))]
out = {}
for var in ("0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.99"):
    m, c, s4 = [], [], 0
    for r in runs:
        e = r["pca"][f"{var}|99"]
        best = lambda f, g: min(e[str(g)]["rank"][N.index(ch)] for ch in GT[f])
        m.append(np.mean([best(f, f) <= 5 for f in F]))
        c.append(np.mean([best(f, g) <= 5 for f in F for g in F if g != f and not (GT[f] & GT[g])]))
        for f in F:
            if f in (1, 2, 7, 8, 10): continue
            top = set(np.argsort(e[str(f)]["rank"])[:5]); s4 += any(N.index(x) in top for x in ("XMEAS(4)", "XMV(4)"))
    out[var] = {"matched5": float(np.mean(m)), "cross5": float(np.mean(c)), "s4_fp": f"{s4}/{7 * len(runs)}", "ncomp": runs[0]["ncomp"][var]}
    print(var, out[var])
json.dump(out, open(os.path.join(ddir, "out/sens_pca_spec.json"), "w"), indent=1)
