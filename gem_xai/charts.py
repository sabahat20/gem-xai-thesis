"""Phase 1c: rolling 30-day candlestick chart images (64x64 greyscale).

Two renderers are provided:
* `raster` (default): a fast, deterministic NumPy rasteriser (2 px per candle,
  wick = 1 px line, bullish body = white, bearish body = grey). ~4,000 images in seconds.
* `mplfinance`: the library named in the proposal; slower (~minutes) but standard.
Both use only the 30 candles ending at day t, so there is no look-ahead.
"""
import io

import numpy as np
import pandas as pd

from .utils import log


def _raster(win: pd.DataFrame, size: int) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    lo, hi = win["Low"].min(), win["High"].max()
    scale = (size - 1) / (hi - lo + 1e-12)
    to_row = lambda p: int(round((hi - p) * scale))
    n = len(win)
    width = size // n
    for i, (_, r) in enumerate(win.iterrows()):
        x = i * width
        img[to_row(r.High):to_row(r.Low) + 1, x] = 0.5
        top, bot = to_row(max(r.Open, r.Close)), to_row(min(r.Open, r.Close))
        img[top:bot + 1, x:x + width] = 1.0 if r.Close >= r.Open else 0.6
    return img


def _mpf(win: pd.DataFrame, size: int) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import mplfinance as mpf
    from PIL import Image
    buf = io.BytesIO()
    style = mpf.make_mpf_style(base_mpf_style="classic", gridstyle="", facecolor="black",
                               marketcolors=mpf.make_marketcolors(up="white", down="grey", inherit=True))
    mpf.plot(win, type="candle", style=style, axisoff=True, figsize=(1, 1),
             savefig=dict(fname=buf, dpi=size, pad_inches=0))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L").resize((size, size)), dtype=np.float32) / 255.0


def build_chart_images(prices: pd.DataFrame, dates: pd.DatetimeIndex, cfg, renderer="raster") -> np.ndarray:
    cache = cfg.cache_dir / f"charts_{renderer}_{cfg.image_size}_{'syn' if cfg.synthetic else 'real'}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if len(d["dates"]) == len(dates) and (pd.DatetimeIndex(d["dates"]) == dates).all():
            log.info("Loaded cached chart images")
            return d["images"]
    fn = _raster if renderer == "raster" else _mpf
    pos = prices.index.get_indexer(dates)
    imgs = np.stack([fn(prices.iloc[p - cfg.chart_window + 1:p + 1], cfg.image_size) for p in pos])
    np.savez_compressed(cache, images=imgs, dates=dates.values)
    log.info("Rendered %d chart images (%s)", len(imgs), renderer)
    return imgs
