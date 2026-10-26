"""Central configuration for the GEM-XAI pipeline.

All dates follow the walk-forward protocol in the research proposal:
train 2010-2022, validation 2023, test Jan 2024 - Jun 2025.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    ticker: str = "SPY"
    # The HMM that generates the labels is always fitted on 2010-2022 (labels are therefore
    # identical across protocol versions). The model-training / validation boundary can be
    # moved so that the validation window contains all three regimes (2023 has no Bear days).
    hmm_train_end: str = "2022-12-31"
    # Download starts one year early so 200-day indicators are fully warmed up by 2010.
    download_start: str = "2009-01-01"
    download_end: str = "2025-07-01"
    train_start: str = "2010-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2025-06-30"

    # Prediction horizon in trading days. The label is the HMM regime at t+horizon,
    # predicted from information available at the close of day t. A horizon of 0
    # would be near-trivial because the HMM observations (log-return, RV21) are
    # also input features.
    horizon: int = 1

    chart_window: int = 30   # candles per chart image
    image_size: int = 64     # 64x64 greyscale

    # Model
    num_emb: int = 128
    text_emb: int = 64
    img_emb: int = 128
    fusion_dim: int = 128
    dropout: float = 0.3
    focal_gamma: float = 2.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 40
    patience: int = 8
    pretrained_resnet: bool = True
    # Modality dropout: during training each non-empty modality is hidden with this
    # probability (at least one modality is always kept). 0 disables it.
    modality_dropout: float = 0.0

    # WGAN-GP
    gan_latent: int = 32
    gan_epochs: int = 2000
    gan_lambda_gp: float = 10.0
    gan_n_critic: int = 5
    gan_target_ratio: float = 0.6   # grow each minority class to 60% of the majority count
    gan_require_pass: bool = True   # drop synthetic samples that fail the <60% discriminator test

    # Explainability / LLM
    n_explanations: int = 50
    shap_background: int = 100
    uncertainty_threshold: float = 0.65
    llm_backend: str = "template"   # "openai", "hf" or "template"
    openai_model: str = "gpt-4o"
    hf_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    seed: int = 42
    out_dir: Path = field(default_factory=lambda: Path("results"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    news_csv: str | None = None   # path to FNSPID-style news CSV (Date, Article_title)
    synthetic: bool = False        # smoke-test mode with simulated prices

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.cache_dir = Path(self.cache_dir)
        for sub in ["", "figures", "tables", "models", "explanations"]:
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


REGIMES = ["Bull", "Bear", "HighVol"]
