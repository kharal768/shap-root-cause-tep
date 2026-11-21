"""Residual-variability ratio of faults that share a channel, correlations and episode counts.

Input: ROOT/data/tep_clean.npz. Output: ROOT/out/residual.json. Paper: Supplementary Section S10."""
import os
import numpy as np, json
from scipy.stats import mannwhitneyu
from data_split import find_episodes
from fault_metadata import ALL_CHANNEL_NAMES as N
ROOT = os.environ.get("TEP_ROOT", "../runs/revised")  # folder with data/tep_clean.npz and out/
d=np.load(ROOT + "/data/tep_clean.npz",allow_pickle=True); X,y=d["X"],d["y"]
eps=find_episodes(y)
sd0=X[y==0].std(0)
out={}
for ch,(fs,fr) in {"XMEAS(21)":(4,11),"XMEAS(22)":(5,12)}.items():
    j=N.index(ch)
    for minlen in (1,5):
        r={f:[np.std(X[s:e,j]-X[s:e,j].mean())/sd0[j] for s,e,l in eps if l==f and e-s>=minlen] for f in (fs,fr)}
        p1=mannwhitneyu(r[fs],r[fr],alternative="less",method="exact").pvalue
        p2=mannwhitneyu(r[fs],r[fr],method="exact").pvalue
        print(ch,"minlen",minlen,{f:(len(v),round(np.mean(v),3)) for f,v in r.items()},"p1 %.4f p2 %.4f"%(p1,p2))
        out[f"{ch}_min{minlen}"]={"step":r[fs],"random":r[fr],"p_one":p1,"p_two":p2}
# correlation
C=np.corrcoef(X[:, sd0>0].T if False else X[:, X.std(0)>0].T)
nc=(X.std(0)>0).sum(); iu=np.triu_indices(nc,1); a=np.abs(C[iu])
print("nonconst",nc,"mean|r| %.3f median %.3f >0.5 %.3f >0.8 %.3f"%(a.mean(),np.median(a),(a>0.5).mean(),(a>0.8).mean()))
for p in [("XMEAS(4)","XMV(4)"),("XMEAS(21)","XMV(10)"),("XMEAS(1)","XMV(3)"),("XMEAS(22)","XMV(11)")]:
    print(p,"r=%.2f"%np.corrcoef(X[:,N.index(p[0])],X[:,N.index(p[1])])[0,1])
print("episodes per class:",{c:sum(1 for e in eps if e[2]==c) for c in range(21)})
json.dump(out,open(ROOT + "/out/residual.json","w"),default=float)
