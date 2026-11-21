"""Draws the paper's figures at printed width (7-8 pt text).

Main text: Fig. 1 fig_flowchart, Fig. 2 fig_perfault_heatmap, Fig. 3 fig_hit_vs_k.
Supplement: Fig. S1 fig_confusion_matrix, Fig. S2 fig_hitrates_seeds, Fig. S3 fig_bootstrap,
Fig. S4 fig_perfault_heatmap_orig.

Inputs: aggregate.json, sens_eval.json, bootstrap_seed0.json and tep_model_artifacts.npz in the out/ folders
of the two dataset roots (TEP_ROOT_REVISED, TEP_ROOT_ORIGINAL). Output: FIG_DIR (default ../figures_out)."""
import os
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})
REV = os.environ.get("TEP_ROOT_REVISED", "../runs/revised")
ORI = os.environ.get("TEP_ROOT_ORIGINAL", "../runs/original")
OUT = os.environ.get("FIG_DIR", "../figures_out") + "/"
os.makedirs(OUT, exist_ok=True)
M = ["MLP+perm", "Lagged MLP+perm", "CNN+perm", "GRU+perm", "XGB+tree", "XGB+perm", "PCA-RBC", "Mean shift", "Mean+var"]
LAB = {"MLP+perm": "MLP · permutation SHAP", "Lagged MLP+perm": "Lagged MLP · permutation SHAP", "CNN+perm": "1D-CNN · channel SHAP",
       "GRU+perm": "GRU · channel SHAP", "XGB+tree": "XGBoost · TreeExplainer", "XGB+perm": "XGBoost · permutation SHAP",
       "PCA-RBC": "PCA contribution (RBC)", "Mean shift": "Mean shift", "Mean+var": "Mean/variance"}
SHORT = {"MLP+perm": "MLP", "Lagged MLP+perm": "Lagged MLP", "CNN+perm": "1D-CNN", "GRU+perm": "GRU", "XGB+tree": "XGB tree",
         "XGB+perm": "XGB perm.", "PCA-RBC": "PCA-RBC", "Mean shift": "Mean shift", "Mean+var": "Mean/var."}
COL = {"MLP+perm": "#2a78d6", "XGB+tree": "#eb6834", "XGB+perm": "#1baf7a", "Mean shift": "#52514e", "Lagged MLP+perm": "#6250d6",
       "Mean+var": "#a3a29c", "PCA-RBC": "#008300", "CNN+perm": "#e87ba4", "GRU+perm": "#eda100"}
MK = {"MLP+perm": "o", "XGB+tree": "s", "XGB+perm": "D", "Mean shift": "^", "Lagged MLP+perm": "P", "Mean+var": "v", "PCA-RBC": "h", "CNN+perm": "X", "GRU+perm": "*"}
F = ["1", "2", "7", "8", "10", "4", "11", "5", "12", "6", "13", "14"]

# ---------- Fig. 1: flowchart (main, 5.4 in) ----------
def box(ax, x, y, w, h, txt, fc="#f0efec"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=INK2, lw=0.8))
    ax.text(x, y, txt, ha="center", va="center", fontsize=7.5, color=INK, linespacing=1.25)
    return dict(x=x, y=y, w=w, h=h)
def arrow(ax, a, b, sa="bottom", sb="top", ls="-"):
    p = lambda bx, s: {"bottom": (bx["x"], bx["y"] - bx["h"] / 2), "top": (bx["x"], bx["y"] + bx["h"] / 2),
                       "left": (bx["x"] - bx["w"] / 2, bx["y"]), "right": (bx["x"] + bx["w"] / 2, bx["y"])}[s]
    ax.annotate("", xy=p(b, sb), xytext=p(a, sa), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, ls=ls, mutation_scale=8, shrinkA=0, shrinkB=0))
W, H = 5.4, 3.1
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
A = box(ax, 2.7, 2.89, 3.0, 0.38, "Two one-year TEP simulations (revised and original)\n52 channels · 21 classes")
B = box(ax, 2.7, 2.42, 2.4, 0.26, "Episode-level 70/30 split, 10 seeds")
xs = [0.68, 1.98, 3.28]
C = [box(ax, xs[0], 1.92, 1.22, 0.42, "MLP (128, 64)\nstatic or lagged"), box(ax, xs[1], 1.92, 1.22, 0.42, "XGBoost\n(200 trees, depth 6)"),
     box(ax, xs[2], 1.92, 1.22, 0.42, "1D-CNN and GRU\n(1-h window)")]
E = [box(ax, xs[0], 1.38, 1.22, 0.42, "Permutation SHAP\n(trained, untrained)"), box(ax, xs[1], 1.38, 1.22, 0.42, "TreeExplainer,\npermutation SHAP"),
     box(ax, xs[2], 1.38, 1.22, 0.42, "Channel SHAP and\nIntegrated Gradients")]
Bf = box(ax, 4.68, 1.65, 1.3, 1.0, "Classifier-free\nrankings:\nmean shift,\nmean/variance, KS,\nPCA (RBC), Granger")
D = box(ax, 0.68, 0.81, 1.22, 0.42, "Documented\nchannels (Table 2)", fc="#cde2fb")
G = box(ax, 2.98, 0.81, 2.6, 0.42, "Rank of documented channels per fault:\nHit-at-1/3/5 and MRR")
T = [box(ax, 0.95, 0.23, 1.75, 0.42, "Exact chance test\n(Poisson-binomial)"), box(ax, 2.75, 0.23, 1.7, 0.42, "Paired, cross-fault and\nbootstrap comparisons"),
     box(ax, 4.5, 0.23, 1.65, 0.42, "Ablation, robustness\nand sensitivity")]
arrow(ax, A, B)
for c in C: arrow(ax, B, c)
arrow(ax, B, Bf)
for c, e in zip(C, E): arrow(ax, c, e)
for e in E: arrow(ax, e, G)
arrow(ax, Bf, G, "bottom", "right"); arrow(ax, D, G, "right", "left")
for t in T: arrow(ax, G, t)
fig.savefig(OUT + "fig_flowchart.pdf"); plt.close(fig)

# ---------- heatmap (main revised 5.4 in; supplement original 6.5 in) ----------
cmap = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])
def heatmap(agg, models, width, name):
    A = json.load(open(agg)); n = len(models)
    fig, ax = plt.subplots(figsize=(width, 0.3 + 0.228 * n + 0.42))
    Z = np.full((n, len(F)), np.nan)
    for i, m in enumerate(models):
        for j, f in enumerate(F):
            a, b, _ = A["freq"][m][f]
            if b: Z[i, j] = a / b
    ax.imshow(Z, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, m in enumerate(models):
        for j, f in enumerate(F):
            a, b, _ = A["freq"][m][f]
            if b: ax.text(j, i, f"{a}" if b == 10 else f"{a}/{b}", ha="center", va="center", fontsize=7.5, color="white" if a / b > 0.6 else INK)
            else:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#f0efec", zorder=0)); ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(F))); ax.set_xticklabels(F); ax.set_yticks(range(n)); ax.set_yticklabels([LAB[m] for m in models])
    ax.set_xlabel("Fault")
    for x in (4.5, 8.5): ax.axvline(x, color="white", lw=3)
    for x, t in [(2, "Stream-4 group"), (6.5, "Step/random pairs"), (10, "Unshared channels")]: ax.text(x, -0.75, t, ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    fig.tight_layout(pad=0.3, rect=(0, 0, 0.985, 1)); fig.savefig(OUT + name); plt.close(fig)
heatmap(REV + "/out/aggregate.json", M, 5.4, "fig_perfault_heatmap.pdf")
heatmap(ORI + "/out/aggregate.json", [m for m in M if m != "XGB+perm"], 6.5, "fig_perfault_heatmap_orig.pdf")

# ---------- hit vs k (main, 5.4 in) ----------
SHAP_C, BASE_C = "#2a78d6", "#eb6834"
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), sharey=True)
for ax, (Dd, title) in zip(axes, ((REV, "Revised dataset"), (ORI, "Original dataset"))):
    r = json.load(open(f"{Dd}/out/sens_eval.json"))["k"]; ks = sorted(int(k) for k in r)
    shap_m = [m for m in r["1"] if "SHAP" in m]; base_m = [m for m in r["1"] if m not in shap_m and m not in ("best_shap", "best_base")]
    for ms, c, lab in ((shap_m, SHAP_C, "SHAP rankings"), (base_m, BASE_C, "Classifier-free rankings")):
        lo = [min(r[str(k)][m]["hit"] for m in ms) * 100 for k in ks]; hi = [max(r[str(k)][m]["hit"] for m in ms) * 100 for k in ks]
        ax.fill_between(ks, lo, hi, color=c, alpha=0.16, linewidth=0); ax.plot(ks, hi, color=c, lw=1.8, label=lab)
    ch = [r[str(k)][shap_m[0]]["chance"] * 100 for k in ks]
    ax.plot(ks, ch, color=INK2, lw=1.1, ls="--", label="Chance")
    ax.axvline(5, color="#c3c2b7", lw=0.8, zorder=0)
    ax.set_title(title, loc="left", color=INK); ax.set_xticks(ks); ax.set_xlim(1, 10); ax.set_ylim(0, 100)
    ax.set_xlabel("Cut-off $k$ (channels inspected)"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("Hit-at-$k$ (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.5)
fig.tight_layout(rect=(0, 0.09, 1, 1), pad=0.3); fig.savefig(OUT + "fig_hit_vs_k.pdf"); plt.close(fig)

# ---------- per-split hit rates (supplement, 6.5 in) ----------
A = json.load(open(REV + "/out/aggregate.json"))
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.75), sharey=True)
for ax, k in zip(axs, (1, 3, 5)):
    for i, m in enumerate(M):
        rows = A["summary"][f"main|{m}"]; v = np.array([rr[f"h{k}"] / rr["n"] for rr in rows]) * 100
        x = i + np.linspace(-0.22, 0.22, len(v))
        ax.scatter(x, v, s=13, marker=MK[m], color=COL[m], edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot([i - .3, i + .3], [v.mean()] * 2, color=INK, lw=1.4, zorder=4)
    chv = np.mean([rr[f"chance{k}"] for rr in A["summary"]["main|MLP+perm"]]) * 100
    ax.axhline(chv, color="#e34948", lw=1.0, ls="--"); ax.text(8.45, chv + 2, "chance", color=INK2, fontsize=7, ha="right")
    ax.set_title(f"Hit-at-{k}"); ax.set_xticks(range(len(M))); ax.set_xticklabels([SHORT[m] for m in M], rotation=55, ha="right", rotation_mode="anchor", fontsize=7)
    ax.set_ylim(0, 100); ax.set_xlim(-.6, 8.6); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel("Faults with a documented\nchannel in the top $k$ (%)")
fig.tight_layout(pad=0.3); fig.savefig(OUT + "fig_hitrates_seeds.pdf"); plt.close(fig)

# ---------- bootstrap (supplement, 6.5 in) ----------
Bt = json.load(open(REV + "/out/bootstrap_seed0.json"))
fs = [f for f in ["1", "2", "4", "6", "7", "8", "10", "11", "12", "13", "14"] if Bt.get(f)]
fig, ax = plt.subplots(figsize=(6.5, 2.45))
for j, f in enumerate(fs):
    b = Bt[f]; c = "#2a78d6" if b["hit5_freq"] >= 0.5 else "#e34948"
    ax.plot([j, j], [b["rank_q05"], b["rank_q95"]], color=c, lw=2.0, solid_capstyle="round")
    ax.scatter(j, b["rank_median"], s=30, color=c, edgecolor="white", zorder=3)
ax.axhline(5.5, color=INK2, ls="--", lw=0.9); ax.text(len(fs) - .55, 4.9, "top-5 cut-off", fontsize=7, ha="right", color=INK2)
ax.set_xticks(range(len(fs))); ax.set_xticklabels([f"{f}\n{Bt[f]['hit5_freq']*100:.0f}%" for f in fs]); ax.set_xlabel("Fault (share of resamples with a hit in the top five)"); ax.set_ylabel("Rank of best documented\nchannel (1 = top)")
ax.set_yticks([1, 5, 10, 15, 20]); ax.set_ylim(21, 0); ax.set_xlim(-0.8, len(fs) - 0.4); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(pad=0.3); fig.savefig(OUT + "fig_bootstrap.pdf"); plt.close(fig)

# ---------- confusion matrix (supplement, 5.1 in) ----------
cm = np.load(REV + "/out/tep_model_artifacts.npz")["confusion_matrix"].astype(float); cmn = cm / cm.sum(1, keepdims=True)
cmap2 = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])
fig, ax = plt.subplots(figsize=(5.1, 4.5))
im = ax.imshow(cmn, cmap=cmap2, vmin=0, vmax=1)
ax.set_xticks(range(21)); ax.set_yticks(range(21)); ax.tick_params(labelsize=7, length=0)
ax.set_xlabel("Predicted class (0 = normal operation)"); ax.set_ylabel("True class (0 = normal operation)")
for f in [3, 9, 15]: ax.add_patch(plt.Rectangle((f - .5, f - .5), 1, 1, fill=False, ec="#e34948", lw=1.5))
for f in [5, 16]: ax.add_patch(plt.Rectangle((f - .5, f - .5), 1, 1, fill=False, ec="#eda100", lw=1.5, ls="--"))
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02); cb.set_label("Share of true-class test samples"); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
fig.tight_layout(pad=0.3); fig.savefig(OUT + "fig_confusion_matrix.pdf"); plt.close(fig)
print("done")
