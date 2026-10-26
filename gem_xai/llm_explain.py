"""Phase 5a: faithfulness-constrained LLM regime explanations.

For each selected test prediction an evidence block is built from the model's own
attributions. The LLM may only cite features in that block. A verifier then checks:
  * every indicator name mentioned exists in the evidence block,
  * the stated regime matches the predicted regime,
  * every number quoted appears in the evidence block (rounded),
  * the uncertainty note is present iff the uncertainty flag is set.
Non-compliant outputs are regenerated (up to 3 attempts); the pass rate is reported.

Backends: "openai" (needs OPENAI_API_KEY), "hf" (local Llama-3-8B-Instruct via
transformers), or "template" (deterministic, no LLM - for testing the pipeline only;
do NOT report template outputs as LLM results).
"""
import json
import os
import re

import numpy as np
import pandas as pd

from .config import REGIMES
from .utils import log

SYSTEM = ("You are a financial risk analyst. Explain a market-regime classification using ONLY the "
          "evidence provided. Do not mention any indicator, number, event or macro narrative that is "
          "not in the evidence block. Output exactly four labelled parts.")

TEMPLATE = """EVIDENCE BLOCK (JSON):
{evidence}

Write the explanation in this exact format:
1. Predicted regime and confidence: <regime>, <confidence as given>
2. Key drivers: <reference only feature names from top_features, with their observed values>
3. Supporting evidence: <relate the drivers, the net sentiment score and the chart region to typical {regime} characteristics>
4. Uncertainty note: <if uncertainty_flag is true, state that confidence is below {thr}; otherwise write "None">"""


def evidence_block(i, proba, shap_row, x_raw_row, sent_row, cam_label, thr):
    c = int(np.argmax(proba))
    top = shap_row.abs().sort_values(ascending=False).index[:5]
    return {
        "predicted_regime": REGIMES[c],
        "confidence": round(float(proba[c]), 3),
        "class_probabilities": {r: round(float(p), 3) for r, p in zip(REGIMES, proba)},
        "top_features": [{"feature": f, "shap_value": round(float(shap_row[f]), 4),
                          "observed_value": round(float(x_raw_row[f]), 4)} for f in top],
        "net_sentiment": round(float(sent_row["net_sentiment"]), 3),
        "dominant_chart_region": cam_label,
        "uncertainty_flag": bool(proba[c] < thr),
    }


def _call(backend, prompt, cfg, _cache={}):
    if backend == "openai":
        from openai import OpenAI
        client = _cache.setdefault("openai", OpenAI())
        r = client.chat.completions.create(model=cfg.openai_model, temperature=0.2,
                                           messages=[{"role": "system", "content": SYSTEM},
                                                     {"role": "user", "content": prompt}])
        return r.choices[0].message.content
    if backend == "hf":
        if "hf" not in _cache:
            from transformers import pipeline
            import torch
            _cache["hf"] = pipeline("text-generation", model=cfg.hf_model, device_map="auto",
                                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        out = _cache["hf"]([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                           max_new_tokens=350, do_sample=False)
        return out[0]["generated_text"][-1]["content"]
    raise ValueError(backend)


def _template(ev, thr):
    drivers = "; ".join(f"{t['feature']} = {t['observed_value']} (SHAP {t['shap_value']})" for t in ev["top_features"][:3])
    unc = f"Confidence is below {thr}." if ev["uncertainty_flag"] else "None"
    return (f"1. Predicted regime and confidence: {ev['predicted_regime']}, {ev['confidence']}\n"
            f"2. Key drivers: {drivers}\n"
            f"3. Supporting evidence: net sentiment of {ev['net_sentiment']} and attention on {ev['dominant_chart_region']}.\n"
            f"4. Uncertainty note: {unc}")


FEEDBACK = ("\n\nYour previous answer was rejected by the automatic verifier for these reasons: {issues}. "
            "Rewrite the explanation following the rules exactly, using only the evidence block.")


def verify(text, ev, all_features, thr):
    issues = []
    text = re.sub(r"[*#`_]{2,}|[*#`]", "", text)   # ignore markdown formatting (bold, headers)
    allowed = {t["feature"] for t in ev["top_features"]}
    for f in all_features:
        if re.search(rf"\b{re.escape(f)}\b", text) and f not in allowed:
            issues.append(f"mentions feature not in evidence: {f}")
    stated = [r for r in REGIMES if re.search(rf"\b{r}\b", text, re.I)]
    if ev["predicted_regime"] not in stated:
        issues.append("predicted regime not stated")
    nums_allowed = {str(thr), str(ev["confidence"]), str(ev["net_sentiment"])}
    nums_allowed |= {str(v) for v in ev["class_probabilities"].values()}
    for t in ev["top_features"]:
        nums_allowed |= {str(t["shap_value"]), str(t["observed_value"])}
    allowed_f = {float(n) for n in nums_allowed}
    for n in re.findall(r"-?\d+\.\d+", text):
        if not any(abs(float(n) - a) < 5e-3 for a in allowed_f):
            issues.append(f"unsupported number: {n}")
    has_unc = not re.search(r"Uncertainty note:\s*None", text, re.I)
    if has_unc != ev["uncertainty_flag"]:
        issues.append("uncertainty note inconsistent with flag")
    return issues


def generate_explanations(idx, dates, proba, shap_df, X_raw, sent_df, cam_labels, all_features, cfg):
    backend = cfg.llm_backend
    if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY not set - falling back to template backend (pipeline test only).")
        backend = "template"
    rows = []
    for i in idx:
        ev = evidence_block(i, proba[i], shap_df.iloc[i], X_raw.iloc[i], sent_df.iloc[i], cam_labels[i],
                            cfg.uncertainty_threshold)
        prompt = TEMPLATE.format(evidence=json.dumps(ev, indent=2), regime=ev["predicted_regime"],
                                 thr=cfg.uncertainty_threshold)
        first_pass, issues, text = None, [], ""
        for attempt in range(3):
            p_ = prompt if attempt == 0 else prompt + FEEDBACK.format(issues="; ".join(issues))
            text = _template(ev, cfg.uncertainty_threshold) if backend == "template" else _call(backend, p_, cfg)
            issues = verify(text, ev, all_features, cfg.uncertainty_threshold)
            if first_pass is None:
                first_pass = not issues
            if not issues:
                break
        rows.append({"date": str(dates[i].date()), "predicted_regime": ev["predicted_regime"],
                     "confidence": ev["confidence"], "evidence": json.dumps(ev), "explanation": text,
                     "passed_first_attempt": first_pass, "passed_final": not issues,
                     "attempts": attempt + 1, "issues": "; ".join(issues), "backend": backend})
    df = pd.DataFrame(rows)
    log.info("Faithfulness: first-attempt pass %.1f%%, final pass %.1f%%",
             100 * df.passed_first_attempt.mean(), 100 * df.passed_final.mean())
    return df


def expert_rating_sheet(df):
    """Blank rubric for 3-5 domain experts (1-5 Likert)."""
    sheet = df[["date", "predicted_regime", "explanation"]].copy()
    for c in ["factual_accuracy", "coherence", "specificity", "actionability"]:
        sheet[c] = ""
    return sheet


def krippendorff_alpha_ordinal(ratings: np.ndarray) -> float:
    """ratings: raters x items (NaN = missing). Ordinal Krippendorff's alpha."""
    R = ratings
    vals = np.unique(R[~np.isnan(R)])
    idx = {v: k for k, v in enumerate(vals)}
    n_v = len(vals)
    O = np.zeros((n_v, n_v))
    for item in R.T:
        x = item[~np.isnan(item)]
        m = len(x)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    O[idx[x[a]], idx[x[b]]] += 1 / (m - 1)
    n_c = O.sum(1); n = n_c.sum()
    D = np.zeros((n_v, n_v))
    for c in range(n_v):
        for k in range(n_v):
            lo, hi = min(c, k), max(c, k)
            D[c, k] = (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
    Do = (O * D).sum() / n
    De = (np.outer(n_c, n_c) * D).sum() / (n * (n - 1))
    return float(1 - Do / De) if De > 0 else float("nan")
