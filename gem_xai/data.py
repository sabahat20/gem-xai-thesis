"""Phase 1a: SPY price data and the 20 engineered technical features."""
import numpy as np
import pandas as pd

from .utils import log

FEATURES = [
    "log_return", "return_5d", "return_21d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_pct_b", "bb_width",
    "sma_10_ratio", "sma_20_ratio", "sma_50_ratio", "sma_200_ratio",
    "atr_14_pct", "obv_z", "rv_5", "rv_21", "volume_z",
    "high_low_range", "close_open_gap",
]


def download_prices(cfg) -> pd.DataFrame:
    cache = cfg.cache_dir / f"{cfg.ticker}_ohlcv.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        log.info("Loaded cached prices %s", df.shape)
        return df
    import yfinance as yf
    df = yf.download(cfg.ticker, start=cfg.download_start, end=cfg.download_end,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise RuntimeError("yfinance returned no data - check network access.")
    df.to_csv(cache)
    log.info("Downloaded %s rows for %s", len(df), cfg.ticker)
    return df


def synthetic_prices(cfg) -> pd.DataFrame:
    """Regime-switching GBM used ONLY to smoke-test the code without network access."""
    rng = np.random.default_rng(cfg.seed)
    idx = pd.bdate_range(cfg.download_start, cfg.download_end)
    params = {0: (0.0006, 0.008), 1: (-0.0010, 0.014), 2: (0.0000, 0.025)}
    P = np.array([[0.985, 0.010, 0.005], [0.020, 0.965, 0.015], [0.030, 0.020, 0.950]])
    s, rets = 0, []
    for _ in idx:
        mu, sd = params[s]
        rets.append(rng.normal(mu, sd))
        s = rng.choice(3, p=P[s])
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, 0.003, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, len(idx))))
    vol = rng.lognormal(18, 0.3, len(idx)) * (1 + 5 * np.abs(np.array(rets)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features at row t use data up to and including the close of day t only."""
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    lr = np.log(c).diff()
    f["log_return"] = lr
    f["return_5d"] = c.pct_change(5)
    f["return_21d"] = c.pct_change(21)

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"], f["macd_signal"], f["macd_hist"] = macd / c, sig / c, (macd - sig) / c

    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    upper, lower = sma20 + 2 * sd20, sma20 - 2 * sd20
    f["bb_pct_b"] = (c - lower) / (upper - lower)
    f["bb_width"] = (upper - lower) / sma20

    for w in (10, 20, 50, 200):
        f[f"sma_{w}_ratio"] = c / c.rolling(w).mean() - 1

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c

    obv = (np.sign(delta).fillna(0) * v).cumsum()
    f["obv_z"] = (obv - obv.rolling(63).mean()) / obv.rolling(63).std()

    f["rv_5"] = lr.rolling(5).std() * np.sqrt(252)
    f["rv_21"] = lr.rolling(21).std() * np.sqrt(252)
    f["volume_z"] = (v - v.rolling(63).mean()) / v.rolling(63).std()
    f["high_low_range"] = (h - l) / c
    f["close_open_gap"] = (o - c.shift()) / c.shift()

    assert list(f.columns) == FEATURES
    return f.replace([np.inf, -np.inf], np.nan)
