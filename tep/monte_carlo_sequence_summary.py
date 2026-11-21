"""Kendall tau, top-five overlap and hit changes between the stored and recomputed sequence-model rankings.

Input and output in ROOT/out (seqmc_summary.json). Paper: Limitations, Supplementary Section S12."""
import json, sys, numpy as np, os
ROOT = os.environ.get("TEP_ROOT", "../runs/revised")
from scipy.stats import kendalltau
G={1:[3,44],2:[3,44],7:[3,44],8:[3,44],10:[3,44],4:[20,50],11:[20,50],5:[21,51],12:[21,51],6:[0,43],13:[6,7,8],14:[50]}
hit=lambda r,f: min(r[i] for i in G[int(f)])<=5
out={}
for a in ['gru','cnn']:
    tb={'seed1':[],'big':[]}; j5={'seed1':[],'big':[]}; ag={'seed1':0,'big':0}; n=0
    for s in range(3):
        p=f'{ROOT}/out/seqmc_{a}_seed{s}.json'
        if not os.path.exists(p): continue
        M=json.load(open(p)); O=json.load(open(f'{ROOT}/out/{a}_seed{s}.json'))[a]['shap']
        for f,v in O.items():
            if not v or M['big'].get(f) is None: continue
            x=np.array(v['rank']); n+=1
            for k in ('seed1','big'):
                y=np.array(M[k][f]['rank']); tb[k].append(kendalltau(x,y)[0]); j5[k].append(len(set(np.where(x<=5)[0])&set(np.where(y<=5)[0]))/5); ag[k]+=hit(x,f)==hit(y,f)
    if n: out[a]={k:{'tau_median':float(np.median(tb[k])),'top5_overlap':float(np.mean(j5[k])),'hit_agree':[int(ag[k]),int(n)]} for k in tb}; print(a,out[a])
json.dump(out,open(f'{ROOT}/out/seqmc_summary.json','w'),indent=1)
