"""Metrics: macro F1, per-class recall, macro AUC-ROC, and DeLong's test."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score

from .config import REGIMES


def robust_macro_auc(y, proba):
    """One-vs-rest AUC averaged over classes that have both positives and negatives in y."""
    aucs = [roc_auc_score((y == c).astype(int), proba[:, c]) for c in range(proba.shape[1])
            if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def metrics(y, proba, name):
    pred = proba.argmax(1)
    present = [c for c in range(3) if (y == c).any()]
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0).astype(float)
    rec[[c for c in range(3) if c not in present]] = np.nan   # recall undefined for an absent class
    auc = robust_macro_auc(y, proba)
    return {"model": name, "macro_f1": f1_score(y, pred, average="macro", labels=present, zero_division=0),
            "accuracy": (pred == y).mean(), "macro_auc": auc,
            **{f"recall_{r}": v for r, v in zip(REGIMES, rec)}}


# ---- Fast DeLong (Sun & Xu, 2014) ----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N); out[J] = T
    return out


def _delong_cov(preds, m):
    pos, neg = preds[:, :m], preds[:, m:]
    k, n = preds.shape[0], preds.shape[1] - m
    tx = np.array([_midrank(r) for r in pos]); ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    return aucs, np.cov(v01) / m + np.cov(v10) / n


def delong_test(y_bin, p1, p2):
    order = np.argsort(-y_bin, kind="stable")
    m = int(y_bin.sum())
    if m == 0 or m == len(y_bin):
        return np.nan, np.nan, np.nan
    aucs, cov = _delong_cov(np.vstack([p1, p2])[:, order], m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], float(2 * norm.sf(abs(z)))


def delong_table(y, ref_proba, others: dict, ref_name="GEM-XAI"):
    rows = []
    for name, p in others.items():
        for c, r in enumerate(REGIMES):
            a1, a2, pv = delong_test((y == c).astype(int), ref_proba[:, c], p[:, c])
            rows.append({"comparison": f"{ref_name} vs {name}", "class": r, "auc_gemxai": a1,
                         "auc_baseline": a2, "p_value": pv, "significant_0.05": pv < 0.05 if pv == pv else False})
    return pd.DataFrame(rows)


def conf_mat(y, proba):
    return confusion_matrix(y, proba.argmax(1), labels=[0, 1, 2])
