"""Phase 1d: daily FinBERT sentiment features.

Input: a news CSV with at least a date column and a headline column (FNSPID format:
`Date`, `Article_title`, optionally `Stock_symbol`). Each headline is scored with
ProsusAI/finbert, then aggregated per trading day into four features:
mean_pos, mean_neg, net_sentiment (= pos - neg), sentiment_std.

Headlines published on a non-trading day (or after the close) are rolled forward to
the next trading day, so day-t features never contain news from after day t.
Days with no headlines get neutral values and a coverage flag is logged.
"""
import numpy as np
import pandas as pd

from .utils import device, log

SENT_FEATURES = ["mean_pos", "mean_neg", "net_sentiment", "sentiment_std"]


def score_headlines(texts, batch_size=64):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device()).eval()
    labels = [model.config.id2label[i].lower() for i in range(3)]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(device())
            out.append(torch.softmax(model(**enc).logits, -1).cpu().numpy())
            if i % (batch_size * 200) == 0:
                log.info("FinBERT scored %d / %d headlines", i, len(texts))
    p = np.concatenate(out)
    return pd.DataFrame({"pos": p[:, labels.index("positive")], "neg": p[:, labels.index("negative")]})


def build_sentiment(trading_days: pd.DatetimeIndex, cfg) -> pd.DataFrame:
    cache = cfg.cache_dir / "daily_sentiment.csv"
    if cfg.synthetic or not cfg.news_csv:
        if not cfg.synthetic:
            log.warning("No news CSV supplied - sentiment features set to neutral zeros. "
                        "Pass --news-csv to use FinBERT.")
        rng = np.random.default_rng(cfg.seed)
        z = pd.DataFrame(0.0, index=trading_days, columns=SENT_FEATURES)
        if cfg.synthetic:
            z["mean_pos"] = rng.uniform(0.2, 0.5, len(z))
            z["mean_neg"] = rng.uniform(0.2, 0.5, len(z))
            z["net_sentiment"] = z.mean_pos - z.mean_neg
            z["sentiment_std"] = rng.uniform(0.1, 0.3, len(z))
        return z
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True)
        return s.reindex(trading_days).fillna({"mean_pos": 0, "mean_neg": 0, "net_sentiment": 0, "sentiment_std": 0})

    news = pd.read_csv(cfg.news_csv)
    dcol = next(c for c in news.columns if c.lower() in ("date", "datetime", "published"))
    tcol = next(c for c in news.columns if c.lower() in ("article_title", "headline", "title"))
    news = news[[dcol, tcol]].dropna()
    ts = pd.to_datetime(news[dcol], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    news = news.assign(ts=ts).dropna(subset=["ts"])
    # After 16:00 ET -> belongs to the next session
    day = news.ts.dt.normalize().dt.tz_localize(None)
    day = day.where(news.ts.dt.hour < 16, day + pd.Timedelta(days=1))
    pos = trading_days.searchsorted(day)
    keep = pos < len(trading_days)
    news = news[keep].assign(day=trading_days[pos[keep]])
    news = news[(news.day >= trading_days[0]) & (news.day <= trading_days[-1])]
    log.info("Scoring %d headlines with FinBERT", len(news))
    sc = score_headlines(news[tcol].astype(str).tolist())
    sc["day"] = news.day.values
    sc["net"] = sc.pos - sc.neg
    g = sc.groupby("day")
    daily = pd.DataFrame({"mean_pos": g.pos.mean(), "mean_neg": g.neg.mean(),
                          "net_sentiment": g.net.mean(), "sentiment_std": g.net.std().fillna(0)})
    daily.to_csv(cache)
    cov = daily.index.isin(trading_days).sum() / len(trading_days)
    log.info("News coverage: %.1f%% of trading days", 100 * cov)
    return daily.reindex(trading_days).fillna(0.0)
