"""Hit rates, MRR, per-fault frequencies and chance tests of the main rankings; extended channel list.

Input: ROOT/out/*_seed<k>.json. Output: ROOT/out/aggregate.json. Paper: Table 3, Figs. 2 and S2."""
import json, glob, numpy as np
from math import comb
from fault_metadata import ALL_CHANNEL_NAMES as N, GROUND_TRUTH_CHANNELS as GT0
F=[1,2,4,5,6,7,8,10,11,12,13,14]
GT_EXT={**GT0, 4:GT0[4]|{"XMEAS(9)"}, 11:GT0[11]|{"XMEAS(9)"}, 14:{"XMV(10)","XMEAS(9)","XMEAS(21)"}}
def ch(g,k): return 1-comb(52-g,k)/comb(52,k)
def pb(ps,x):
    d=np.zeros(len(ps)+1); d[0]=1
    for p in ps: d[1:]=d[1:]*(1-p)+d[:-1]*p; d[0]*=1-p
    return float(d[x:].sum())
def best_rank(entry,f,GT):
    return min(entry["rank"][N.index(c)] for c in GT[f])
R=[json.load(open(p)) for p in sorted(glob.glob(ROOT + "/out/sweep_seed*.json"),key=lambda s:int(s.split("seed")[1][:-5]))]
import os
ROOT = os.environ.get("TEP_ROOT", "../runs/revised")  # folder with data/tep_clean.npz and out/
for r in R:
    tp=ROOT + f"/out/temporal_seed{r['seed']}.json"
    if os.path.exists(tp):
        T=json.load(open(tp)); r["baseline_var"]=T["baseline_var"]; r["mlp_lag"]=T["mlp_lag"]
R=[r for r in R if "mlp_lag" in r]
for r in R:
    cp=ROOT + f"/out/cnn_seed{r['seed']}.json"
    if os.path.exists(cp): r["cnn"]=json.load(open(cp))["cnn"]
    pp=ROOT + f"/out/pca_seed{r['seed']}.json"
    if os.path.exists(pp): r["pca"]=json.load(open(pp))
    gp=ROOT + f"/out/gru_seed{r['seed']}.json"
    if os.path.exists(gp): r["gru"]=json.load(open(gp))["gru"] if all(os.path.exists(ROOT + f"/out/temporal_seed{r['seed']}.json") for r in R) else R
methods={"MLP+perm":lambda r:r["mlp"]["shap"],"XGB+tree":lambda r:r["xgb"]["shap"],"XGB+perm":lambda r:r["xgb"]["shap_perm"],"Mean shift":lambda r:r["baseline"],"Mean+var":lambda r:r["baseline_var"],"Lagged MLP+perm":lambda r:r["mlp_lag"]["shap"],"CNN+perm":lambda r:r["cnn"]["shap"],"GRU+perm":lambda r:r["gru"]["shap"],"PCA-RBC":lambda r:r["pca"]}
if any("shap_perm" not in r["xgb"] for r in R): methods.pop("XGB+perm")  # not run on the original dataset
summary={}
for GTname,GT in [("main",GT0),("ext",GT_EXT)]:
  for m,get in methods.items():
    rows=[]
    for r in R:
        e=get(r); fs=[f for f in F if e.get(str(f)) is not None]
        br={f:best_rank(e[str(f)],f,GT) for f in fs}
        row={"seed":r["seed"],"n":len(fs),"ranks":br}
        for k in (1,3,5):
            h=sum(br[f]<=k for f in fs); row[f"h{k}"]=h
            row[f"p{k}"]=pb([ch(len(GT[f]),k) for f in fs],h)
            row[f"chance{k}"]=float(np.mean([ch(len(GT[f]),k) for f in fs]))
        row["mrr"]=float(np.mean([1/br[f] for f in fs]))
        rows.append(row)
    summary[(GTname,m)]=rows
def show(GTname):
  print(f"== ground truth: {GTname}")
  for m in methods:
    rows=summary[(GTname,m)]
    for k in (1,3,5):
        rate=[r[f"h{k}"]/r["n"] for r in rows]; ps=[r[f"p{k}"] for r in rows]
        print(f"{m:10s} Hit@{k}: mean {np.mean(rate)*100:5.1f}% sd {np.std(rate,ddof=1)*100 if len(rows)>1 else 0:4.1f} range {min(rate)*100:.1f}-{max(rate)*100:.1f}  chance {np.mean([r[f'chance{k}'] for r in rows])*100:.1f}%  max p {max(ps):.2e}  seeds p<0.05: {sum(p<0.05 for p in ps)}/{len(ps)}")
    print(f"{m:10s} MRR mean {np.mean([r['mrr'] for r in rows]):.3f}")
show("main"); show("ext")
# per-fault hit@5 frequency
print("\nper-fault hit@5 frequency (hits/evaluable seeds), main GT")
freq={}
for m in methods:
    rows=summary[("main",m)]; freq[m]={}
    for f in F:
        ev=[r["ranks"][f] for r in rows if f in r["ranks"]]
        freq[m][f]=(sum(x<=5 for x in ev),len(ev),float(np.median(ev)) if ev else None)
    print(m, {f:f"{a}/{b}" for f,(a,b,_) in freq[m].items()})
# pooled by group
groups={"unshared":[6,13,14],"pairs":[4,5,11,12],"stream4":[1,2,7,8,10]}
for m in methods:
    print(m,{g:"%d/%d"%(sum(freq[m][f][0] for f in fs),sum(freq[m][f][1] for f in fs)) for g,fs in groups.items()})
# classifier
for mk in ("mlp","xgb","mlp_lag","cnn","gru"):
    a=[r[mk]["acc"] for r in R]; mf=[r[mk]["macro_f1"] for r in R]
    worst=[sorted(r[mk]["f1"].items(),key=lambda kv:kv[1])[:4] for r in R]
    print(mk,"acc %.3f±%.3f (%.3f-%.3f) macroF1 %.3f±%.3f"%(np.mean(a),np.std(a,ddof=1) if len(a)>1 else 0,min(a),max(a),np.mean(mf),np.std(mf,ddof=1) if len(mf)>1 else 0))
    f1m={c:np.mean([float(r[mk]["f1"][str(c)]) for r in R]) for c in range(21)}
    print("  mean F1 by class (sorted):",[(c,round(v,2)) for c,v in sorted(f1m.items(),key=lambda kv:kv[1])[:7]])
    print("  seeds where 3,9,15 are the three worst:",sum(set(int(c) for c,_ in w[:3])=={3,9,15} for w in worst),"/",len(R))
    print("  seeds where fault 5 excluded:",sum(r[mk]["shap"]["5"] is None for r in R))
json.dump({"freq":{m:{str(f):v for f,v in d.items()} for m,d in freq.items()},
           "summary":{f"{a}|{b}":v for (a,b),v in summary.items()}},open(ROOT + "/out/aggregate.json","w"),default=float)
