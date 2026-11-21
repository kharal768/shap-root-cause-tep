"""SHAP values of the MLP for fault 14 at split seed 0 (Supplementary Fig. S5).

Input: TEP_ROOT_REVISED/data/tep_clean.npz. Output: FIG_DIR/fig_shap_beeswarm_fault14.pdf."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tep"))
REV = os.environ.get("TEP_ROOT_REVISED", "../runs/revised")
OUT = os.environ.get("FIG_DIR", "../figures_out"); os.makedirs(OUT, exist_ok=True)
import numpy as np, shap, warnings, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from data_split import episode_train_test_split
from fault_metadata import ALL_CHANNEL_NAMES as N
d=np.load(REV + "/data/tep_clean.npz",allow_pickle=True); X,y=d["X"],d["y"]
Xtr,ytr,Xte,yte,_=episode_train_test_split(X,y,0.3,seed=0)
sc=StandardScaler().fit(Xtr); Xtr_s,Xte_s=sc.transform(Xtr),sc.transform(Xte)
rng=np.random.default_rng(0); bg=Xtr_s[rng.choice(len(Xtr_s),30,replace=False)]
mlp=MLPClassifier(hidden_layer_sizes=(128,64),alpha=1e-4,max_iter=300,random_state=0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=15).fit(Xtr_s,ytr)
p=mlp.predict(Xte_s); f=14
idx=np.where((yte==f)&(p==f))[0]; s=np.random.default_rng(0).choice(idx,30,replace=False)
sv=shap.Explainer(mlp.predict_proba,bg,algorithm="permutation",seed=0,feature_names=N)(Xte_s[s])
e=sv[:,:,f]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8})
shap.plots.beeswarm(e,max_display=12,show=False,plot_size=(5.5,3.6))
fig=plt.gcf(); ax=fig.axes[0]
for t in ax.get_yticklabels():
    t.set_fontsize(8)
    if t.get_text() in ("XMV(10)","XMEAS(21)","XMEAS(9)"): t.set_color("#008300"); t.set_fontweight("bold")
for t in ax.get_xticklabels(): t.set_fontsize(7.5)
ax.set_xlabel("SHAP value for the fault-14 class probability",fontsize=8)
for cax in fig.axes[1:]:
    cax.tick_params(labelsize=7.5); cax.set_ylabel(cax.get_ylabel(),fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "/fig_shap_beeswarm_fault14.pdf")
r=np.argsort(np.abs(e.values).mean(0))[::-1]; print([N[i] for i in r[:6]])
