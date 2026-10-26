"""Phase 4: architecturally consistent explainability.

* Branch A (MLP, numerical)    -> SHAP GradientExplainer (expected gradients)
* Branch B (sentiment linear)  -> Captum Integrated Gradients over the 4 daily sentiment scores
* Branch C (ResNet-18, image)  -> Grad-CAM on the last residual block (layer4)
* Modality-level attribution   -> SHAP GradientExplainer run jointly over all three inputs,
                                  |attribution| summed per modality and normalised to 100%.
"""
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .train import to_tensors
from .utils import device, log


def _shap_for_class(sv, cls_idx):
    """Normalise SHAP output across versions to: list(per input) of arrays for chosen class per sample."""
    # New SHAP: list over inputs, each array (..., n_classes). Old SHAP: list over classes of list over inputs.
    if isinstance(sv, list) and isinstance(sv[0], list):
        return [np.stack([sv[c][k][i] for i, c in enumerate(cls_idx)]) for k in range(len(sv[0]))]
    return [np.stack([a[i, ..., c] for i, c in enumerate(cls_idx)]) for a in sv]


def shap_all_modalities(model, bg, data, pred):
    dev = device()
    model.eval()
    bgt = [t.to(dev) for t in to_tensors(*bg)]
    dt = [t.to(dev) for t in to_tensors(*data)]
    exp = shap.GradientExplainer(model, bgt)
    sv = exp.shap_values(dt, nsamples=100)
    return _shap_for_class(sv, pred)  # [num (N,20), text (N,4), img (N,1,64,64)]


def modality_shares(shap_parts, method="net"):
    """Per-sample share of attribution by modality.
    net  : |sum of signed SHAP values within the modality| (default). Expected-gradients SHAP
           values add up to the prediction change, so the signed sum is the modality's net
           contribution and is not inflated by the number of input dimensions.
    gross: sum of |SHAP| over all inputs of the modality. Biased towards high-dimensional
           inputs (the chart has 4,096 pixels vs 20 indicators) - reported for comparison only."""
    if method == "net":
        mags = np.column_stack([np.abs(p.reshape(len(p), -1).sum(1)) for p in shap_parts])
    else:
        mags = np.column_stack([np.abs(p).reshape(len(p), -1).sum(1) for p in shap_parts])
    return mags / np.clip(mags.sum(1, keepdims=True), 1e-12, None)


class _TextWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, xt, xn, xi): return self.m(xn, xt, xi)


def integrated_gradients_text(model, data, pred):
    dev = device()
    model.eval()
    xn, xt, xi = [t.to(dev) for t in to_tensors(*data)]
    ig = IntegratedGradients(_TextWrapper(model))
    out = []
    for i in range(0, len(xn), 32):
        s = slice(i, i + 32)
        a = ig.attribute(xt[s], baselines=torch.zeros_like(xt[s]), target=torch.tensor(pred[s], device=dev),
                         additional_forward_args=(xn[s], xi[s]), n_steps=50)
        out.append(a.detach().cpu().numpy())
    return np.concatenate(out)


class _ImgWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m; self.xn = self.xt = None
    def forward(self, xi): return self.m(self.xn, self.xt, xi)


def gradcam_images(model, data, pred):
    dev = device()
    model.eval()
    xn, xt, xi = [t.to(dev) for t in to_tensors(*data)]
    w = _ImgWrapper(model)
    cams = []
    with GradCAM(model=w, target_layers=[model.img.backbone.layer4[-1]]) as cam:
        for i in range(0, len(xi), 32):
            s = slice(i, i + 32)
            w.xn, w.xt = xn[s], xt[s]
            cams.append(cam(input_tensor=xi[s], targets=[ClassifierOutputTarget(int(c)) for c in pred[s]]))
    return np.concatenate(cams)


def cam_region_label(cam):
    """Human-readable dominant region: which part of the 30-day window and price range."""
    H, W = cam.shape
    t = [cam[:, : W // 3].sum(), cam[:, W // 3: 2 * W // 3].sum(), cam[:, 2 * W // 3:].sum()]
    p = [cam[: H // 2].sum(), cam[H // 2:].sum()]
    time = ["early-window (days 1-10)", "mid-window (days 11-20)", "recent (days 21-30)"][int(np.argmax(t))]
    price = ["upper price range", "lower price range"][int(np.argmax(p))]
    return f"{time} candles, {price}"


def run_explainability(model, train_data, test_data, pred, feat_names, sent_names, cfg):
    rng = np.random.default_rng(cfg.seed)
    bi = rng.choice(len(train_data[0]), min(cfg.shap_background, len(train_data[0])), replace=False)
    bg = [a[bi] for a in train_data[:3]]
    log.info("SHAP GradientExplainer over %d test samples", len(test_data[0]))
    parts = shap_all_modalities(model, bg, test_data[:3], pred)
    shares = modality_shares(parts, "net")
    shares_gross = modality_shares(parts, "gross")
    log.info("Integrated Gradients (sentiment branch)")
    ig = integrated_gradients_text(model, test_data[:3], pred)
    log.info("Grad-CAM (image branch)")
    cams = gradcam_images(model, test_data[:3], pred)
    return {
        "shap_num": pd.DataFrame(parts[0], columns=feat_names),
        "shap_text": parts[1], "shap_img": parts[2],
        "modality_shares": pd.DataFrame(shares, columns=["Numerical", "Sentiment", "Chart"]),
        "modality_shares_gross": pd.DataFrame(shares_gross, columns=["Numerical", "Sentiment", "Chart"]),
        "ig_text": pd.DataFrame(ig, columns=sent_names),
        "gradcam": cams,
        "cam_labels": [cam_region_label(c) for c in cams],
    }
