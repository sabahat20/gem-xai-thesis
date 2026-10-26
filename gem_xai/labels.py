"""Phase 1b: three-state Gaussian HMM regime labels with no look-ahead.

* The HMM is fitted (Baum-Welch EM) on the TRAINING partition only.
* Observation vector: [daily log-return, 21-day realised volatility].
* States are mapped deterministically. On equity data the lowest-return state is usually
  also the highest-volatility state, so the proposal's two rules collide. Default
  ("return_first"): lowest mean return -> Bear; of the remaining two, highest mean
  volatility -> HighVol; the last -> Bull. This guarantees the Bear state always has the
  lowest (typically negative) mean return. "vol_first" is kept for the sensitivity analysis.
* Labels are decoded with the causal FORWARD filter: the label for day t uses only
  observations up to day t. Viterbi decoding is also reported for comparison, because
  Viterbi uses the full sequence and therefore contains look-ahead.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from .utils import log


def _obs(feats: pd.DataFrame) -> np.ndarray:
    return feats[["log_return", "rv_21"]].values


def fit_hmm(train_feats: pd.DataFrame, seed: int = 42, n_restarts: int = 10, mapping_rule="return_first"):
    X = _obs(train_feats)
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    best, best_ll = None, -np.inf
    for r in range(n_restarts):
        m = GaussianHMM(n_components=3, covariance_type="full", n_iter=500, random_state=seed + r)
        m.fit(Xs)
        ll = m.score(Xs)
        if ll > best_ll:
            best, best_ll = m, ll
    means = best.means_ * sd + mu
    if mapping_rule == "return_first":
        bear = int(np.argmin(means[:, 0]))
        rest = [i for i in range(3) if i != bear]
        hv = max(rest, key=lambda i: means[i, 1])
        bull = [i for i in rest if i != hv][0]
    else:
        hv = int(np.argmax(means[:, 1]))
        rest = [i for i in range(3) if i != hv]
        bull = max(rest, key=lambda i: means[i, 0])
        bear = [i for i in rest if i != bull][0]
    mapping = {bull: 0, bear: 1, hv: 2}
    log.info("HMM fitted, logL=%.1f; state means (ret, rv21): %s", best_ll, np.round(means, 4).tolist())
    return {"model": best, "mu": mu, "sd": sd, "mapping": mapping, "state_means": means, "loglik": best_ll}


def forward_filter(hmm, feats: pd.DataFrame) -> np.ndarray:
    """Causal filtered state probabilities P(s_t | x_1..x_t), mapped to regime order."""
    m = hmm["model"]
    Xs = (_obs(feats) - hmm["mu"]) / hmm["sd"]
    logB = np.column_stack([multivariate_normal(m.means_[k], m.covars_[k]).logpdf(Xs) for k in range(3)])
    logA = np.log(m.transmat_)
    la = np.log(m.startprob_ + 1e-300) + logB[0]
    out = np.zeros_like(logB)
    out[0] = la - logsumexp(la)
    for t in range(1, len(Xs)):
        la = logsumexp(out[t - 1][:, None] + logA, axis=0) + logB[t]
        out[t] = la - logsumexp(la)
    probs = np.exp(out)
    mapped = np.zeros_like(probs)
    for s, r in hmm["mapping"].items():
        mapped[:, r] = probs[:, s]
    return mapped


def viterbi_labels(hmm, feats):
    Xs = (_obs(feats) - hmm["mu"]) / hmm["sd"]
    states = hmm["model"].predict(Xs)
    return np.array([hmm["mapping"][s] for s in states])
