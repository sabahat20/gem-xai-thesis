"""Additional analyses added in the final run (v2):

* Moving-block bootstrap confidence intervals for macro-F1 and accuracy, and paired
  bootstrap differences between GEM-XAI and each baseline (blocks keep the serial
  dependence of daily regimes).
* Probability calibration: multiclass Brier score, expected calibration error (ECE)
  and a reliability diagram.
* SHAP TreeExplainer for the XGBoost baseline and the rank correlation between its
  feature importances and those of GEM-XAI's numerical branch.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from .config import REGIMES


def _block_indices(n, block, rng):
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def _macro_f1(y, pred):
    present = [c for c in range(3) if (y == c).any()]
    return f1_score(y, pred, average="macro", labels=present, zero_division=0)


def bootstrap_table(y, probs: dict, ref="GEM-XAI (full)", n_boot=1000, block=20, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = [_block_indices(n, block, rng) for _ in range(n_boot)]
    preds = {k: p.argmax(1) for k, p in probs.items()}
    rows = []
    boot = {}
    for name, pr in preds.items():
        f1s = np.array([_macro_f1(y[i], pr[i]) for i in idx])
        acc = np.array([(y[i] == pr[i]).mean() for i in idx])
        boot[name] = f1s
        rows.append({"model": name, "macro_f1": _macro_f1(y, pr),
                     "macro_f1_ci_low": np.percentile(f1s, 2.5), "macro_f1_ci_high": np.percentile(f1s, 97.5),
                     "accuracy": (y == pr).mean(),
                     "accuracy_ci_low": np.percentile(acc, 2.5), "accuracy_ci_high": np.percentile(acc, 97.5)})
    ci = pd.DataFrame(rows)
    diffs = []
    if ref in boot:
        for name in boot:
            if name == ref:
                continue
            d = boot[ref] - boot[name]
            diffs.append({"comparison": f"{ref} minus {name}",
                          "macro_f1_diff": _macro_f1(y, preds[ref]) - _macro_f1(y, preds[name]),
                          "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                          "share_boot_ref_better": float((d > 0).mean())})
    return ci, pd.DataFrame(diffs)


def calibration_table(y, probs: dict, n_bins=10):
    rows, curves = [], {}
    Y = np.eye(3)[y]
    for name, p in probs.items():
        brier = float(((p - Y) ** 2).sum(1).mean())
        conf, pred = p.max(1), p.argmax(1)
        correct = (pred == y).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        which = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
        ece, pts = 0.0, []
        for b in range(n_bins):
            m = which == b
            if m.any():
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
                pts.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
        rows.append({"model": name, "brier": brier, "ece": float(ece), "mean_confidence": float(conf.mean()),
                     "accuracy": float(correct.mean())})
        curves[name] = pts
    return pd.DataFrame(rows), curves


def reliability_plot(curves: dict, names, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8, label="Perfect calibration")
    colors = ["#c0603f", "#264653", "#2a9d8f", "#6d597a"]
    for k, n in enumerate(names):
        if n not in curves:
            continue
        pts = np.array([(a, b) for a, b, _ in curves[n]])
        ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=4, lw=1.5, color=colors[k % len(colors)], label=n)
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0, 1.0); ax.legend(frameon=False, fontsize=7)
    fig.savefig(path); plt.close(fig)


def xgb_shap(xg_model, X_test, feat_names):
    try:
        import shap
        sv = shap.TreeExplainer(xg_model).shap_values(X_test)
        if isinstance(sv, list):                  # older SHAP: list over classes
            arr = np.stack(sv, -1)
        else:
            arr = np.asarray(sv)
            if arr.ndim == 2:
                arr = arr[..., None]
    except (ValueError, TypeError):
        # Some SHAP versions cannot read the multi-class base_score written by XGBoost >= 3.
        # Fall back to XGBoost's own exact TreeSHAP values (same algorithm).
        import xgboost as xgb
        contrib = np.asarray(xg_model.get_booster().predict(
            xgb.DMatrix(np.asarray(X_test, dtype=float)), pred_contribs=True))
        if contrib.ndim == 3:                     # (n, classes, features + bias)
            arr = contrib[:, :, :-1].transpose(0, 2, 1)
        else:                                     # (n, features + bias)
            arr = contrib[:, :-1, None]
    imp = np.abs(arr).mean(axis=(0, 2))
    return pd.Series(imp, index=feat_names).sort_values(ascending=False)


def importance_agreement(imp_a: pd.Series, imp_b: pd.Series, k=5):
    common = imp_a.index.intersection(imp_b.index)
    rho, p = spearmanr(imp_a[common], imp_b[common])
    top_a, top_b = set(imp_a.index[:k]), set(imp_b.index[:k])
    return {"spearman_rho": float(rho), "p_value": float(p), f"top{k}_overlap": len(top_a & top_b),
            f"top{k}_gemxai": list(imp_a.index[:k]), f"top{k}_xgboost": list(imp_b.index[:k])}
