"""Re-run ONLY the LLM explanation step on saved evidence blocks (no retraining).

The main run saves every evidence block in results/explanations/llm_explanations.csv.
This script sends those same evidence blocks to the chosen LLM, re-runs the
faithfulness verifier and overwrites the explanations + summary.

    export OPENAI_API_KEY=sk-...
    python run_llm_only.py --results results --llm-backend openai
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from gem_xai.config import Config
from gem_xai.data import FEATURES
from gem_xai.llm_explain import FEEDBACK, TEMPLATE, _call, _template, expert_rating_sheet, verify
from gem_xai.utils import log, save_json


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--results", default="results")
    a.add_argument("--llm-backend", default="openai", choices=["openai", "hf"])
    a.add_argument("--openai-model", default="gpt-4o")
    a.add_argument("--hf-model", default=None)
    args = a.parse_args()
    cfg = Config(out_dir=args.results, llm_backend=args.llm_backend, openai_model=args.openai_model)
    if args.hf_model:
        cfg.hf_model = args.hf_model
    path = Path(args.results) / "explanations" / "llm_explanations.csv"
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        ev = json.loads(r["evidence"])
        prompt = TEMPLATE.format(evidence=json.dumps(ev, indent=2), regime=ev["predicted_regime"],
                                 thr=cfg.uncertainty_threshold)
        first, issues, text = None, [], ""
        for attempt in range(3):
            p_ = prompt if attempt == 0 else prompt + FEEDBACK.format(issues="; ".join(issues))
            text = _call(cfg.llm_backend, p_, cfg)
            issues = verify(text, ev, FEATURES, cfg.uncertainty_threshold)
            first = (not issues) if first is None else first
            if not issues:
                break
        rows.append({**r.to_dict(), "explanation": text, "passed_first_attempt": first,
                     "passed_final": not issues, "attempts": attempt + 1, "issues": "; ".join(issues),
                     "backend": f"{cfg.llm_backend}:{cfg.openai_model if cfg.llm_backend == 'openai' else cfg.hf_model}"})
        log.info("%s %s first-pass=%s", r["date"], ev["predicted_regime"], first)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    expert_rating_sheet(out).to_csv(path.parent / "expert_rating_sheet.csv", index=False)
    save_json({"faithfulness_first_attempt": float(out.passed_first_attempt.mean()),
               "faithfulness_final": float(out.passed_final.mean()),
               "mean_attempts": float(out.attempts.mean()), "backend": out.backend.iloc[0]},
              Path(args.results) / "tables" / "faithfulness_summary.json")
    log.info("First-attempt pass %.1f%%, final pass %.1f%%",
             100 * out.passed_first_attempt.mean(), 100 * out.passed_final.mean())


if __name__ == "__main__":
    main()
