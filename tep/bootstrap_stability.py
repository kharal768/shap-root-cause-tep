"""Bootstrap stability of the MLP's SHAP ranking over resamples of 30 explained samples (seed 0).

Input: ROOT/data/tep_clean.npz. Output: ROOT/out/bootstrap_seed0.json. Paper: Supplementary Fig. S3."""
import os
import json, warnings, numpy as np, shap
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from data_split import episode_train_test_split
from fault_metadata import ALL_CHANNEL_NAMES as N, VALIDATABLE_FAULTS, GROUND_TRUTH_CHANNELS as GT
ROOT = os.environ.get("TEP_ROOT", "../runs/revised")  # folder with data/tep_clean.npz and out/
warnings.filterwarnings("ignore")
d=np.load(ROOT + "/data/tep_clean.npz",allow_pickle=True); X,y=d["X"],d["y"]
Xtr,ytr,Xte,yte,_=episode_train_test_split(X,y,0.3,seed=0)
sc=StandardScaler().fit(Xtr); Xtr_s,Xte_s=sc.transform(Xtr),sc.transform(Xte)
rng=np.random.default_rng(0); bg=Xtr_s[rng.choice(len(Xtr_s),30,replace=False)]
mlp=MLPClassifier(hidden_layer_sizes=(128,64),alpha=1e-4,max_iter=300,random_state=0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=15).fit(Xtr_s,ytr)
p=mlp.predict(Xte_s); out={}
B=1000
for f in VALIDATABLE_FAULTS:
    idx=np.where((yte==f)&(p==f))[0]
    if len(idx)<5: out[f]=None; print(f,"skip"); continue
    s=rng.choice(idx,size=min(100,len(idx)),replace=False)
    v=np.abs(shap.Explainer(mlp.predict_proba,bg,algorithm="permutation",seed=0)(Xte_s[s]).values[:,:,f])
    gi=[N.index(c) for c in GT[f]]
    hits=[];best=[]
    for b in range(B):
        m=v[rng.integers(0,len(s),30)].mean(0); order=np.argsort(m)[::-1]
        r=min(int(np.where(order==g)[0][0])+1 for g in gi); best.append(r); hits.append(r<=5)
    out[f]={"n_pool":int(len(s)),"hit5_freq":float(np.mean(hits)),"rank_median":float(np.median(best)),
            "rank_q05":float(np.percentile(best,5)),"rank_q95":float(np.percentile(best,95))}
    print(f,out[f],flush=True)
json.dump(out,open(ROOT + "/out/bootstrap_seed0.json","w"))
