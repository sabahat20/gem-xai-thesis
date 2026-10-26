# Changes in the final run (v2)

These changes were made after the first full run (results v1). They fix two design
problems diagnosed in v1, not test-set scores: the sentiment modality was empty, and the
2023 validation year contained no Bear days, so early stopping and Optuna could not
reward Bear detection. The test period (Jan 2024 – Jun 2025) and the HMM labels are unchanged.

| Area | v1 | v2 |
|---|---|---|
| News sentiment | none (all zeros) | 73,353 Guardian business-section headlines (2010–Jun 2025) scored with ProsusAI/finbert, after-close news moved to the next session (`cache/daily_sentiment.csv`) |
| HMM labels | fitted on 2010–2022 | unchanged (fitted on 2010–2022) |
| Model training data | 2010–2022 | 2010–2021 |
| Validation | 2023 (no Bear days) | 2022–2023 (all three regimes) |
| Fusion regularisation | none | modality dropout (tuned by Optuna, 0–0.5) |
| Optuna | 10 trials, lr + dropout | 40 trials, lr + dropout + modality dropout |
| Baselines | HMM, XGBoost, MLP, ResNet-18, multimodal w/o augmentation | + FinBERT sentiment only, + persistence (today's regime) |
| Ablations | – | indicators + sentiment, indicators + chart, GEM-XAI without modality dropout |
| Statistics | DeLong, 5 seeds | + moving-block bootstrap CIs (block 20, 1,000 reps) and paired differences |
| Calibration | – | Brier score, ECE, reliability diagram |
| Explanations | template backend | Qwen2.5-3B-Instruct (open model) via `run_llm_only.py --llm-backend hf` |
| XGBoost explainability | – | SHAP TreeExplainer + rank agreement with GEM-XAI's SHAP |

Run (Colab T4 or AWS GPU):
```bash
python run_all.py --news-csv cache/daily_sentiment.csv --model-train-end 2021-12-31 \
       --val-end 2023-12-31 --optuna-trials 40 --seeds 5 --out results_v2
python run_llm_only.py --results results_v2 --llm-backend hf --hf-model Qwen/Qwen2.5-3B-Instruct
```
