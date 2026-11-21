"""Monte Carlo error of the MLP's permutation SHAP: another explainer seed and a tenfold budget.

Usage: python monte_carlo_mlp.py <root> <seeds>. Output: ROOT/out/budget_seed<k>.json.
Paper: Limitations, Supplementary Section S4."""
import json, os, sys, time, warnings
import numpy as np, shap
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from data_split import episode_train_test_split
from fault_metadata import VALIDATABLE_FAULTS as F
from ablation import ranking, as3
warnings.filterwarnings("ignore")
ddir=sys.argv[1]
d=np.load(os.path.join(ddir,"data/tep_clean.npz"),allow_pickle=True); X,y=d["X"],d["y"]
for seed in [int(a) for a in sys.argv[2:]]:
    t0=time.time()
    Xtr,ytr,Xte,yte,_=episode_train_test_split(X,y,test_frac=0.3,seed=seed)
    sc=StandardScaler().fit(Xtr); Xtr_s,Xte_s=sc.transform(Xtr),sc.transform(Xte)
    rng=np.random.default_rng(seed); bg=Xtr_s[rng.choice(len(Xtr_s),30,replace=False)]
    clf=MLPClassifier(hidden_layer_sizes=(128,64),activation="relu",solver="adam",alpha=1e-4,max_iter=300,random_state=seed,
                      early_stopping=True,validation_fraction=0.1,n_iter_no_change=15).fit(Xtr_s,ytr)
    pred=clf.predict(Xte_s); srng=np.random.default_rng(seed)
    out={"seed1":{},"big":{},"dev_all":{},"dev_train":{}}
    for f in F:
        alli=np.where(yte==f)[0]
        out["dev_all"][f]={"rank":ranking(np.abs(Xte_s[alli]-bg.mean(0)).mean(0))}
        tri=np.where(ytr==f)[0]
        out["dev_train"][f]={"rank":ranking(np.abs(Xtr_s[tri].mean(0)-Xtr_s[ytr==0].mean(0)))}
        idx=np.where((yte==f)&(pred==f))[0]
        if len(idx)<5: out["seed1"][f]=out["big"][f]=None; continue
        s=srng.choice(idx,size=min(30,len(idx)),replace=False)
        v1=as3(shap.Explainer(clf.predict_proba,bg,algorithm="permutation",seed=1)(Xte_s[s]).values)
        vb=as3(shap.Explainer(clf.predict_proba,bg,algorithm="permutation",seed=0)(Xte_s[s],max_evals=5000).values)
        out["seed1"][f]={"rank":ranking(np.abs(v1[:,:,f]).mean(0))}
        out["big"][f]={"rank":ranking(np.abs(vb[:,:,f]).mean(0))}
    json.dump(out,open(os.path.join(ddir,f"out/budget_seed{seed}.json"),"w"))
    print(ddir,seed,f"{time.time()-t0:.0f}s",flush=True)
