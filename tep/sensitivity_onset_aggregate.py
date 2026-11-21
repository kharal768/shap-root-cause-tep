"""Aggregates the label-onset runs.

Usage: python sensitivity_onset_aggregate.py <root> [<root> ...]. Output: <root>/out/onsetagg.json. Paper: Supplementary Section S8."""
import json, glob, os, sys, numpy as np
from math import comb
from fault_metadata import ALL_CHANNEL_NAMES as N, GROUND_TRUTH_CHANNELS as GT
F=[1,2,4,5,6,7,8,10,11,12,13,14]
def ch(g,k): return 1-comb(52-g,k)/comb(52,k)
def pb(ps,x):
    d=np.zeros(len(ps)+1); d[0]=1
    for p in ps: d[1:]=d[1:]*(1-p)+d[:-1]*p; d[0]*=1-p
    return float(d[x:].sum())
out={}
for D in sys.argv[1:]:
    runs=[json.load(open(f)) for f in sorted(glob.glob(os.path.join(D,'out/onset_seed*.json')))]
    out[D]={}
    for meth in ('shap','ms'):
        for m in ('0','20','60','120'):
            h1=[];h5=[];nf=[];sig=0
            for r in runs:
                e=r[meth][m]; br={f:min(e[str(f)]['rank'][N.index(c)] for c in GT[f]) for f in F if e.get(str(f))}
                fs=list(br); h1.append(np.mean([br[f]<=1 for f in fs])); h5.append(np.mean([br[f]<=5 for f in fs])); nf.append(len(fs))
                for k in (1,5): sig+=pb([ch(len(GT[f]),k) for f in fs], sum(br[f]<=k for f in fs))<0.05
            out[D][f'{meth}{m}']={'hit1':float(np.mean(h1)),'hit5':float(np.mean(h5)),'nf':float(np.mean(nf)),'sig':f'{sig}/{2*len(runs)}'}
            print(D[-5:],meth,m,{k:(round(v*100,1) if k.startswith('hit') else v) for k,v in out[D][f'{meth}{m}'].items()})
    json.dump(out[D],open(os.path.join(D,'out/onsetagg.json'),'w'),indent=1)
