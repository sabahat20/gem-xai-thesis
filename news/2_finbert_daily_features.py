"""
Step 2 - Score headlines with FinBERT and build the four daily sentiment features.

Run on AWS (GPU recommended; CPU works but is slower). Needs internet once to download
the ProsusAI/finbert model from Hugging Face.

Usage:
  pip install torch transformers pandas pandas_market_calendars
  python 2_finbert_daily_features.py --headlines headlines_guardian.csv --out daily_sentiment.csv

Leakage rule (same as the thesis): news published at or after the 16:00 New York close
is assigned to the NEXT trading session, so day t only uses news known by the close of day t.

Output columns (one row per NYSE trading day 2010-01-04 .. 2025-06-30):
  date, mean_pos, mean_neg, net_sentiment, sentiment_std, n_headlines
Days with no headlines get zeros and n_headlines = 0.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas_market_calendars as mcal


def trading_days():
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date="2009-12-01", end_date="2025-07-15")
    return pd.DatetimeIndex(sched.index.date)


def assign_session(ts_utc, days):
    ny = ts_utc.tz_convert("America/New_York")
    d = pd.DatetimeIndex(ny.date)
    after_close = np.asarray(ny.hour >= 16)
    pos = days.searchsorted(d, side="left")            # first trading day >= calendar date
    same = (pos < len(days)) & (days[np.minimum(pos, len(days) - 1)] == d)
    pos = np.where(same & after_close, pos + 1, pos)    # after close on a trading day -> next session
    pos = np.minimum(pos, len(days) - 1)
    return days[pos]


def score(headlines, batch=64):
    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    out = np.zeros((len(headlines), 3), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(headlines), batch):
            enc = tok(headlines[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
            p = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            out[i:i + batch, 0] = p[:, lab["positive"]]
            out[i:i + batch, 1] = p[:, lab["negative"]]
            out[i:i + batch, 2] = p[:, lab["neutral"]]
            if (i // batch) % 200 == 0:
                print(f"scored {i}/{len(headlines)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headlines", required=True)
    ap.add_argument("--out", default="daily_sentiment.csv")
    ap.add_argument("--max_per_day", type=int, default=0, help="optional cap per session for speed (0 = no cap)")
    a = ap.parse_args()

    df = pd.read_csv(a.headlines).dropna(subset=["headline"])
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
    days = trading_days()
    df["date"] = assign_session(pd.DatetimeIndex(df["published_utc"]), days)
    df = df[(df["date"] >= "2010-01-01") & (df["date"] <= "2025-06-30")]
    if a.max_per_day:
        df = df.groupby("date", group_keys=False).apply(lambda g: g.sample(min(len(g), a.max_per_day), random_state=42))
    print("Headlines to score:", len(df))

    p = score(df["headline"].astype(str).tolist())
    df["pos"], df["neg"] = p[:, 0], p[:, 1]
    df["net"] = df["pos"] - df["neg"]
    daily = df.groupby("date").agg(mean_pos=("pos", "mean"), mean_neg=("neg", "mean"),
                                   net_sentiment=("net", "mean"), sentiment_std=("net", "std"),
                                   n_headlines=("net", "size"))
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    full = pd.DataFrame(index=days[(days >= "2010-01-01") & (days <= "2025-06-30")])
    full = full.join(daily).fillna({"mean_pos": 0, "mean_neg": 0, "net_sentiment": 0, "sentiment_std": 0, "n_headlines": 0})
    full.index.name = "date"
    full.to_csv(a.out)
    cov = (full["n_headlines"] > 0).mean()
    print(f"Saved {a.out}: {len(full)} trading days, {cov:.1%} with at least one headline")
    print(full.groupby(full.index.year)["n_headlines"].agg(["mean", "min"]))


if __name__ == "__main__":
    main()
