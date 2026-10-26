"""End-to-end GEM-XAI experiment.

Usage (real data):
    python run_all.py --news-csv data/fnspid_headlines.csv --llm-backend openai --optuna-trials 30
Smoke test (no network, simulated prices):
    python run_all.py --synthetic --fast
"""
import argparse
import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from gem_xai import plots
from gem_xai.charts import build_chart_images
from gem_xai.config import REGIMES, Config
from gem_xai.data import FEATURES, compute_features, download_prices, synthetic_prices
from gem_xai.evaluate import conf_mat, delong_table, metrics
from gem_xai import extra
from gem_xai.explain import run_explainability
from gem_xai.gan import augment
from gem_xai.labels import fit_hmm, forward_filter, viterbi_labels
from gem_xai.llm_explain import expert_rating_sheet, generate_explanations
from gem_xai.sentiment import SENT_FEATURES, build_sentiment
from gem_xai.train import optuna_search, predict_proba, train_model
from gem_xai.utils import log, save_json, set_seed


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--synthetic", action="store_true")
    a.add_argument("--fast", action="store_true", help="few epochs, for smoke testing")
    a.add_argument("--news-csv", default=None)
    a.add_argument("--llm-backend", default="template", choices=["template", "openai", "hf"])
    a.add_argument("--hf-model", default=None, help="local instruct model for --llm-backend hf")
    a.add_argument("--mapping-rule", default="return_first", choices=["return_first", "vol_first"])
    a.add_argument("--seeds", type=int, default=5, help="repeat neural models over N seeds; report mean/std")
    a.add_argument("--optuna-trials", type=int, default=0)
    a.add_argument("--horizon", type=int, default=1)
    a.add_argument("--renderer", default="raster", choices=["raster", "mplfinance"])
    a.add_argument("--no-pretrained", action="store_true")
    a.add_argument("--allow-failed-gan", action="store_true",
                   help="use synthetic samples even if they fail the discriminator test (report as ablation)")
    a.add_argument("--gan-epochs", type=int, default=None)
    a.add_argument("--model-train-end", default="2022-12-31",
                   help="last day of the MODEL training partition (HMM is always fitted on 2010-2022)")
    a.add_argument("--val-end", default="2023-12-31")
    a.add_argument("--modality-dropout", type=float, default=0.2)
    a.add_argument("--bootstrap", type=int, default=1000)
    a.add_argument("--out", default="results")
    a.add_argument("--seed", type=int, default=42)
    return a.parse_args()


def main():
    args = parse()
    cfg = Config(synthetic=args.synthetic, news_csv=args.news_csv, llm_backend=args.llm_backend,
                 horizon=args.horizon, out_dir=args.out, seed=args.seed,
                 pretrained_resnet=not args.no_pretrained)
    cfg.gan_require_pass = not args.allow_failed_gan
    cfg.train_end, cfg.val_end = args.model_train_end, args.val_end
    cfg.val_start = str((pd.Timestamp(cfg.train_end) + pd.Timedelta(days=1)).date())
    cfg.modality_dropout = args.modality_dropout
    if args.hf_model:
        cfg.hf_model = args.hf_model
    if args.fast:
        cfg.epochs, cfg.gan_epochs, cfg.patience, cfg.shap_background, cfg.n_explanations = 4, 300, 3, 30, 20
    if args.gan_epochs:
        cfg.gan_epochs = args.gan_epochs
    set_seed(cfg.seed)
    F, T = cfg.out_dir / "figures", cfg.out_dir / "tables"

    # ---------------- Phase 1: data, features, labels ----------------
    prices = synthetic_prices(cfg) if cfg.synthetic else download_prices(cfg)
    feats = compute_features(prices)
    pos = np.arange(len(prices))
    ok = feats.notna().all(1).values & (pos >= cfg.chart_window - 1)
    ok &= (prices.index >= cfg.train_start) & (prices.index <= cfg.test_end)
    dates = prices.index[ok]
    feats = feats.loc[dates]
    sent = build_sentiment(dates, cfg)[SENT_FEATURES]

    train_mask_all = dates <= cfg.hmm_train_end
    hmm = fit_hmm(feats[train_mask_all], seed=cfg.seed, mapping_rule=args.mapping_rule)
    filt = forward_filter(hmm, feats)          # causal P(regime_t | x_1..t)
    regime = filt.argmax(1)
    vit = viterbi_labels(hmm, feats)
    log.info("Filtered vs Viterbi label agreement: %.1f%%", 100 * (regime == vit).mean())

    h = cfg.horizon
    y_all = np.r_[regime[h:], np.full(h, -1)] if h > 0 else regime.copy()
    split = np.where(dates <= cfg.train_end, 0, np.where(dates <= cfg.val_end, 1, 2))
    # Purge the last h days of each partition so no target crosses a split boundary
    keep = y_all >= 0
    for s in (0, 1):
        last = np.where(split == s)[0][-h:] if h > 0 else []
        keep[last] = False
    images = build_chart_images(prices, dates, cfg, args.renderer)

    def part(s):
        m = keep & (split == s)
        return m
    mtr, mva, mte = part(0), part(1), part(2)
    sc_n = StandardScaler().fit(feats.values[mtr])
    sc_t = StandardScaler().fit(sent.values[mtr])
    im_mu, im_sd = images[mtr].mean(), images[mtr].std()
    Xn = sc_n.transform(feats.values).astype(np.float32)
    Xt = np.nan_to_num(sc_t.transform(sent.values)).astype(np.float32)
    Xi = ((images - im_mu) / im_sd).astype(np.float32)
    mk = lambda m: (Xn[m], Xt[m], Xi[m], y_all[m])
    train, val, test = mk(mtr), mk(mva), mk(mte)
    test_dates = dates[mte]

    counts = {n: np.bincount(d[3], minlength=3) for n, d in zip(["Train", "Validation", "Test"], [train, val, test])}
    pd.DataFrame(counts, index=REGIMES).T.to_csv(T / "table_class_distribution.csv")
    hmm_tab = pd.DataFrame(hmm["state_means"], columns=["mean_log_return", "mean_rv21"])
    hmm_tab["regime"] = [REGIMES[hmm["mapping"][s]] for s in range(3)]
    hmm_tab.to_csv(T / "table_hmm_states.csv", index=False)
    plots.regime_timeline(prices.loc[dates, "Close"], regime,
                          [pd.Timestamp(cfg.val_start), pd.Timestamp(cfg.test_start)], F / "fig_regime_timeline.png")
    plots.class_distribution(counts, F / "fig_class_distribution.png")
    save_json({"filtered_vs_viterbi_agreement": float((regime == vit).mean()), "hmm_loglik": hmm["loglik"],
               "transition_matrix": hmm["model"].transmat_.tolist(), "mapping": hmm["mapping"],
               "n_train": int(mtr.sum()), "n_val": int(mva.sum()), "n_test": int(mte.sum()),
               "horizon": h, "hmm_train_end": cfg.hmm_train_end, "model_train_end": cfg.train_end,
               "val_start": cfg.val_start, "val_end": cfg.val_end,
               "sentiment_nonzero_days_share": float((sent.abs().sum(1) > 0).mean())},
              cfg.out_dir / "data_summary.json")

    # ---------------- Phase 2: WGAN-GP augmentation (train only) ----------------
    Xs, ys, gan_rep = augment(train[0], train[3], cfg)
    rng = np.random.default_rng(cfg.seed)
    pair_idx = np.array([rng.choice(np.where(train[3] == c)[0]) for c in ys], dtype=int)
    train_aug = (np.vstack([train[0], Xs]).astype(np.float32),
                 np.vstack([train[1], train[1][pair_idx]]) if len(ys) else train[1],
                 np.concatenate([train[2], train[2][pair_idx]]) if len(ys) else train[2],
                 np.concatenate([train[3], ys]).astype(int))
    for c, rep in gan_rep.items():
        plots.tsne_real_vs_synth(train[0][train[3] == c], rep.pop("_samples"), REGIMES[c],
                                 F / f"fig_tsne_{REGIMES[c]}.png", cfg.seed)
    pd.DataFrame({REGIMES[c]: {k: v for k, v in r.items() if k != "loss_history"} for c, r in gan_rep.items()}).to_csv(
        T / "table_gan_quality.csv")

    # ---------------- Phase 3: models + baselines ----------------
    if args.optuna_trials > 0:
        study = optuna_search(cfg, train_aug, val, args.optuna_trials)
        cfg.lr, cfg.dropout = study.best_params["lr"], study.best_params["dropout"]
        cfg.modality_dropout = study.best_params.get("modality_dropout", cfg.modality_dropout)
        study.trials_dataframe().to_csv(T / "table_optuna_trials.csv", index=False)
    save_json({"lr": cfg.lr, "dropout": cfg.dropout, "modality_dropout": cfg.modality_dropout,
               "optuna_trials": args.optuna_trials, "seeds": args.seeds, "epochs": cfg.epochs,
               "patience": cfg.patience, "batch_size": cfg.batch_size, "news_csv": cfg.news_csv,
               "model_train_end": cfg.train_end, "val_start": cfg.val_start, "val_end": cfg.val_end,
               "pretrained_resnet": cfg.pretrained_resnet}, T / "run_config.json")

    probs = {}
    A = hmm["model"].transmat_
    perm = [s for s, _ in sorted(hmm["mapping"].items(), key=lambda kv: kv[1])]
    A_m = A[np.ix_(perm, perm)]
    probs["HMM (filtered + transition)"] = filt[mte] @ np.linalg.matrix_power(A_m, max(h, 1))
    # Persistence reference: tomorrow's regime = today's filtered regime probabilities
    probs["Persistence (today's regime)"] = filt[mte].copy()

    xg = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, eval_metric="mlogloss", random_state=cfg.seed)
    xg.fit(train[0], train[3], sample_weight=compute_sample_weight("balanced", train[3]))
    probs["XGBoost (indicators)"] = xg.predict_proba(test[0])

    models = {}
    specs = [("MLP (indicators only)", ("num",), train, None),
             ("FinBERT sentiment only", ("text",), train, None),
             ("ResNet-18 chart only", ("img",), train, None),
             ("Indicators + Sentiment", ("num", "text"), train, None),
             ("Indicators + Chart", ("num", "img"), train, None),
             ("Multimodal (no augmentation)", ("num", "text", "img"), train, None),
             ("GEM-XAI (no modality dropout)", ("num", "text", "img"), train_aug, 0.0),
             ("GEM-XAI (full)", ("num", "text", "img"), train_aug, None)]
    if cfg.news_csv is None and not cfg.synthetic:
        # Without news the sentiment inputs are constant, so a sentiment-only model is meaningless.
        specs = [s_ for s_ in specs if s_[0] not in ("FinBERT sentiment only", "Indicators + Sentiment")]
    for name, mods, data, md in specs:
        set_seed(cfg.seed)
        m, vauc = train_model(cfg, data, val, modalities=mods, modality_dropout=md)
        models[name] = m
        probs[name] = predict_proba(m, *test[:3])
        log.info("%s: best val AUC %.4f", name, vauc)

    # ---- Seed robustness: neural models are retrained with several seeds ----
    nn_specs = specs
    seed_rows = []
    for k in range(args.seeds):
        sd = cfg.seed + k
        for name, mods, data, md in nn_specs:
            if k == 0:
                p = probs[name]
            else:
                set_seed(sd)
                mk_, _ = train_model(cfg, data, val, modalities=mods, verbose=False, modality_dropout=md)
                p = predict_proba(mk_, *test[:3])
            seed_rows.append({**metrics(test[3], p, name), "seed": sd})
        log.info("Seed %d done", sd)
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(T / "table_seed_runs.csv", index=False)
    num_cols = [c for c in seed_df.columns if c not in ("model", "seed")]
    agg = seed_df.groupby("model", sort=False)[num_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.round(4).to_csv(T / "table_test_results_seeds.csv")
    log.info("\n%s", agg[[c for c in agg.columns if c.startswith("macro_f1")]].round(4).to_string())

    res = pd.DataFrame([metrics(test[3], p, n) for n, p in probs.items()])
    res.to_csv(T / "table_test_results.csv", index=False)
    log.info("\n%s", res.round(4).to_string(index=False))
    gem = probs["GEM-XAI (full)"]
    delong_table(test[3], gem, {k: v for k, v in probs.items() if k != "GEM-XAI (full)"}).to_csv(
        T / "table_delong.csv", index=False)
    plots.recall_comparison(res, F / "fig_recall_comparison.png")

    # ---- Extra analyses: bootstrap CIs, calibration, XGBoost SHAP ----
    ci, diffs = extra.bootstrap_table(test[3], probs, n_boot=args.bootstrap, seed=cfg.seed)
    ci.to_csv(T / "table_bootstrap_ci.csv", index=False)
    diffs.to_csv(T / "table_bootstrap_diff.csv", index=False)
    cal, curves = extra.calibration_table(test[3], probs)
    cal.to_csv(T / "table_calibration.csv", index=False)
    extra.reliability_plot(curves, ["GEM-XAI (full)", "XGBoost (indicators)", "MLP (indicators only)"],
                           F / "fig_reliability.png")
    xg_imp = extra.xgb_shap(xg, test[0], FEATURES)
    xg_imp.to_csv(T / "table_shap_xgboost.csv", header=["mean_abs_shap"])
    plots.bar_importance(xg_imp, "Mean |SHAP| (XGBoost)", F / "fig_shap_xgboost.png", "#6d597a")
    for n in ["GEM-XAI (full)", "Multimodal (no augmentation)", "XGBoost (indicators)", "MLP (indicators only)"]:
        plots.confusion(conf_mat(test[3], probs[n]), n, F / f"fig_cm_{n.split()[0].lower()}.png")
    import torch
    torch.save(models["GEM-XAI (full)"].state_dict(), cfg.out_dir / "models" / "gem_xai.pt")

    # ---------------- Phase 4: explainability ----------------
    model = models["GEM-XAI (full)"]
    pred = gem.argmax(1)
    ex = run_explainability(model, train, test, pred, FEATURES, SENT_FEATURES, cfg)
    sh = ex["modality_shares"] * 100
    sh["Predicted regime"] = [REGIMES[c] for c in pred]
    mod_tab = sh.groupby("Predicted regime").mean().reindex([r for r in REGIMES if r in set(sh["Predicted regime"])])
    mod_tab.loc["All"] = sh.drop(columns="Predicted regime").mean()
    mod_tab.round(2).to_csv(T / "table_modality_attribution.csv")
    shg = ex["modality_shares_gross"] * 100
    shg["Predicted regime"] = sh["Predicted regime"]
    gtab = shg.groupby("Predicted regime").mean()
    gtab.loc["All"] = shg.drop(columns="Predicted regime").mean()
    gtab.round(2).to_csv(T / "table_modality_attribution_gross.csv")
    plots.modality_by_regime(mod_tab, F / "fig_modality_attribution.png")
    plots.bar_importance(ex["shap_num"].abs().mean(), "Mean |SHAP|", F / "fig_shap_numerical.png")
    gem_imp = ex["shap_num"].abs().mean().sort_values(ascending=False)
    gem_imp.to_csv(T / "table_shap_gemxai.csv", header=["mean_abs_shap"])
    save_json(extra.importance_agreement(gem_imp, xg_imp), T / "shap_agreement.json")
    ig = ex["ig_text"].assign(regime=[REGIMES[c] for c in pred]).groupby("regime").mean()
    ig.round(5).to_csv(T / "table_ig_sentiment.csv")
    plots.bar_importance(ex["ig_text"].abs().mean(), "Mean |Integrated Gradients|", F / "fig_ig_sentiment.png", "#e9c46a")
    pd.Series(ex["cam_labels"]).groupby([REGIMES[c] for c in pred]).value_counts().to_csv(T / "table_gradcam_regions.csv")
    sel = [int(np.argmax(np.where(pred == c, gem[:, c], -1))) for c in range(3) if (pred == c).any()]
    plots.gradcam_grid([images[mte][i] for i in sel], [ex["gradcam"][i] for i in sel],
                       [f"{REGIMES[pred[i]]} {test_dates[i].date()}" for i in sel], F / "fig_gradcam_examples.png")
    with torch.no_grad():
        predict_proba(model, *test[:3], bs=len(test[0]))
        att = model.last_attention.cpu().numpy()
    pd.DataFrame(att, columns=["Numerical", "Sentiment", "Chart"]).assign(
        regime=[REGIMES[c] for c in pred]).groupby("regime").mean().round(4).to_csv(T / "table_fusion_attention.csv")

    # ---------------- Phase 5: LLM explanations ----------------
    k = cfg.n_explanations
    per = max(1, k // 3)
    idx = []
    for c in range(3):
        cand = np.where(pred == c)[0]
        idx += list(rng.choice(cand, min(per, len(cand)), replace=False)) if len(cand) else []
    rest = np.setdiff1d(np.arange(len(pred)), idx)
    idx += list(rng.choice(rest, max(0, k - len(idx)), replace=False))
    raw = pd.DataFrame(feats.values[mte], columns=FEATURES)
    expl = generate_explanations(sorted(idx), test_dates, gem, ex["shap_num"], raw,
                                 sent.iloc[np.where(mte)[0]].reset_index(drop=True), ex["cam_labels"], FEATURES, cfg)
    expl.to_csv(cfg.out_dir / "explanations" / "llm_explanations.csv", index=False)
    expert_rating_sheet(expl).to_csv(cfg.out_dir / "explanations" / "expert_rating_sheet.csv", index=False)
    save_json({"faithfulness_first_attempt": float(expl.passed_first_attempt.mean()),
               "faithfulness_final": float(expl.passed_final.mean()),
               "mean_attempts": float(expl.attempts.mean()), "backend": expl.backend.iloc[0]},
              T / "faithfulness_summary.json")
    log.info("Done. Results in %s", cfg.out_dir.resolve())


if __name__ == "__main__":
    main()
