import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from .config import REGIMES

COL = {"Bull": "#2a9d8f", "Bear": "#e76f51", "HighVol": "#6d597a"}
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9})


def regime_timeline(close, regimes, splits, path):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(close.index, close.values, color="black", lw=0.7)
    for c, r in enumerate(REGIMES):
        m = regimes == c
        ax.fill_between(close.index, close.min(), close.max(), where=m, color=COL[r], alpha=0.25, lw=0, label=r)
    for d in splits:
        ax.axvline(d, color="grey", ls="--", lw=0.8)
    ax.set_ylabel("Close (adj.)"); ax.legend(loc="upper left", ncol=3, frameon=False)
    fig.savefig(path); plt.close(fig)


def class_distribution(counts: dict, path):
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(counts)); w = 0.27
    for c, r in enumerate(REGIMES):
        ax.bar(x + (c - 1) * w, [v[c] for v in counts.values()], w, color=COL[r], label=r)
    ax.set_xticks(x, list(counts)); ax.set_ylabel("Trading days"); ax.legend(frameon=False)
    fig.savefig(path); plt.close(fig)


def tsne_real_vs_synth(real, synth, cls_name, path, seed=0):
    n = min(len(real), len(synth), 800)
    rng = np.random.default_rng(seed)
    R = real[rng.choice(len(real), n, replace=False)]; S = synth[rng.choice(len(synth), n, replace=False)]
    Z = TSNE(2, random_state=seed, perplexity=min(30, n // 4)).fit_transform(np.vstack([R, S]))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(*Z[:n].T, s=5, alpha=0.6, label="Real", color=COL[cls_name])
    ax.scatter(*Z[n:].T, s=5, alpha=0.6, label="Synthetic", color="grey")
    ax.legend(frameon=False); ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path); plt.close(fig)


def confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(3.4, 3))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(3), REGIMES); ax.set_yticks(range(3), REGIMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    fig.savefig(path); plt.close(fig)


def recall_comparison(df, path):
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(df)), 3.4))
    x = np.arange(len(df)); w = 0.27
    for c, r in enumerate(REGIMES):
        ax.bar(x + (c - 1) * w, df[f"recall_{r}"], w, color=COL[r], label=r)
    ax.set_xticks(x, df["model"], rotation=25, ha="right"); ax.set_ylabel("Recall"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(path); plt.close(fig)


def modality_by_regime(tab, path):
    fig, ax = plt.subplots(figsize=(5, 3))
    bottom = np.zeros(len(tab))
    for col, colr in zip(tab.columns, ["#264653", "#e9c46a", "#8ab17d"]):
        ax.bar(tab.index, tab[col], bottom=bottom, label=col, color=colr); bottom += tab[col].values
    ax.set_ylabel("Share of |attribution| (%)"); ax.legend(frameon=False, bbox_to_anchor=(1, 1))
    fig.savefig(path); plt.close(fig)


def bar_importance(series, xlabel, path, color="#264653"):
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(5, 0.25 * len(s) + 1))
    ax.barh(s.index, s.values, color=color); ax.set_xlabel(xlabel)
    fig.savefig(path); plt.close(fig)


def gradcam_grid(images, cams, titles, path):
    k = len(images)
    fig, axes = plt.subplots(2, k, figsize=(2.2 * k, 4.4))
    axes = np.atleast_2d(axes).reshape(2, k)
    for i in range(k):
        axes[0, i].imshow(images[i], cmap="gray"); axes[0, i].set_title(titles[i], fontsize=8)
        axes[1, i].imshow(images[i], cmap="gray"); axes[1, i].imshow(cams[i], cmap="jet", alpha=0.45)
        for a in axes[:, i]: a.axis("off")
    fig.savefig(path); plt.close(fig)
