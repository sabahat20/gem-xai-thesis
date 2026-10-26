# GEM-XAI: Generative and Explainable Multimodal AI for Financial Market Regime Classification

Code, data and results for the MSc thesis in Machine Learning and Artificial Intelligence at
Liverpool John Moores University (LJMU), by **Sabahat Zartar Jamil**, supervised by **Dr. K. Lokeshwaran**.

GEM-XAI forecasts the next-day market regime (**Bull**, **Bear** or **HighVol**) of the SPDR S&P 500 ETF (SPY),
2010 – June 2025. It fuses three modalities:
- 20 technical indicators;
- daily FinBERT sentiment from 73,353 Guardian business headlines;
- 30-day candlestick images.

The modalities are fused with additive attention and modality dropout, and every prediction is explained with
architecture-matched methods and verified LLM explanations.

## Main results (final run, test period Jan 2024 – Jun 2025, 373 trading days)

| Model | Macro-F1 (seed 42) | Macro-F1 (5 seeds) | Macro AUC |
|---|---|---|---|
| HMM (filtered + transition), reference | 0.958 | – | 0.992 |
| **GEM-XAI (full)** | **0.905** | **0.869 ± 0.036** | **0.983** |
| XGBoost (indicators) | 0.880 | – | 0.981 |
| Indicators + Sentiment | 0.843 | 0.858 ± 0.033 | 0.957 |
| Indicators + Chart | 0.853 | 0.846 ± 0.023 | 0.971 |
| MLP (indicators only) | 0.848 | 0.836 ± 0.024 | 0.971 |
| ResNet-18 chart only | 0.510 | 0.504 ± 0.023 | 0.731 |
| FinBERT sentiment only | 0.215 | 0.261 ± 0.036 | 0.509 |

- **Against the single-modality models:** GEM-XAI is far better in every run.
- **Against the MLP:** significantly better in the primary run (DeLong p ≤ 0.004; block-bootstrap 95% CI for the macro-F1 gain: 0.025 to 0.110), but not across seeds (paired p = 0.14) or in the AWS re-run.
- **Against XGBoost:** statistically comparable, with a lower Brier score in the primary run (0.151 vs 0.175) and under-confident rather than over-confident probabilities.
- **WGAN-GP augmentation:** failed the pre-specified quality gate (gradient-boosting discriminator 80.4% > 60%), so it was not used.
- **LLM explanations (Qwen2.5-3B-Instruct):** 82% passed the faithfulness verifier at the first attempt and 94% after feedback-driven regeneration.
- **Cross-platform re-run** (`results/aws_cpu_check`, AWS SageMaker ml.t3.xlarge CPU, fixed hyperparameters, seed 42): the labels, data and HMM references reproduced exactly (0.958), but GEM-XAI (0.860), the MLP (0.861) and XGBoost (0.864) were tied. GEM-XAI's lead in the primary run therefore partly reflects training variability; see Section 5.6.4 of the thesis.
- **First run** (`results/first_run_v1`, no news data, 2023-only validation): GEM-XAI scored 0.724. The changes between the two runs are listed in [`CHANGES_v2.md`](CHANGES_v2.md).

## Repository structure

```
gem_xai/                 pipeline package
  data.py                SPY download (yfinance) and the 20 technical features
  labels.py              3-state Gaussian HMM, causal forward-filter labels, Viterbi comparison
  charts.py              64x64 candlestick rasteriser (and optional mplfinance renderer)
  sentiment.py           FinBERT scoring and daily aggregation (reads data/daily_sentiment.csv via cache/)
  gan.py                 WGAN-GP and the MMD / t-SNE / discriminator quality gate
  models.py, train.py    three-branch network, attention fusion, modality dropout, focal loss, Optuna
  explain.py             SHAP GradientExplainer, Integrated Gradients (Captum), Grad-CAM
  llm_explain.py         evidence blocks, constrained prompt, faithfulness verifier
  evaluate.py, extra.py  metrics, DeLong test, block bootstrap, calibration, XGBoost SHAP
run_all.py               end-to-end experiment
run_llm_only.py          re-run only the LLM explanation step on saved evidence blocks
news/                    scripts that built the sentiment data (Guardian API -> FinBERT -> daily features)
notebooks/               Google Colab notebooks used for the news data and the final run
data/daily_sentiment.csv daily sentiment features 2010-01-04 .. 2025-06-30 (scores only, no article text)
results/final_run_v2/    tables, figures, explanations and log of the final run (reported in the thesis)
results/first_run_v1/    outputs of the first run, kept for transparency
results/aws_cpu_check/   independent CPU re-run on AWS SageMaker (fixed hyperparameters, one seed)
```

## How to reproduce

GPU recommended (tested on a Google Colab T4).

```bash
pip install -r requirements.txt
mkdir -p cache && cp data/daily_sentiment.csv cache/daily_sentiment.csv

# Main experiment, about 30 minutes on a T4
python run_all.py --news-csv cache/daily_sentiment.csv --model-train-end 2021-12-31 \
       --val-end 2023-12-31 --optuna-trials 40 --seeds 5 --out results_v2

# LLM explanations with an open model, about 15 minutes on a T4
python run_llm_only.py --results results_v2 --llm-backend hf --hf-model Qwen/Qwen2.5-3B-Instruct
```

Or open `notebooks/GEM-XAI_final_run_colab.ipynb` in Google Colab and choose **Runtime → Run all**.

To rebuild the sentiment data from scratch, you need a free Guardian Open Platform key:
```bash
python news/1_download_news.py --source guardian --key YOUR_KEY
python news/2_finbert_daily_features.py --headlines headlines_guardian.csv --out daily_sentiment.csv
```

Smoke test without internet: `python run_all.py --synthetic --fast --no-pretrained --out results_smoke`

The trained weights (`gem_xai.pt`, about 45 MB) are not included because of GitHub's upload size limit. They are reproduced by the commands above.

## Data sources

| Data | Source |
|---|---|
| SPY daily prices (adjusted) | Yahoo Finance via `yfinance` |
| News headlines | [The Guardian Open Platform](https://open-platform.theguardian.com/), business section, used to compute aggregate sentiment scores only |
| Sentiment model | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) |
| Explanation LLM | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| Image backbone | ResNet-18, ImageNet weights (torchvision) |

## Leakage controls
- **Causal labels:** regime labels come from the causal forward filter; Viterbi is reported only for comparison.
- **Genuine horizon:** the target is the regime at t+1.
- **Purged splits:** the last day of each partition is purged.
- **Training-only fitting:** scalers are fitted on the training partition only.
- **Timely news:** after-close news is moved to the next session.

## Disclaimer
This is research code for an academic thesis. It is not investment advice.
