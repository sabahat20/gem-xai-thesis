"""Training loop with early stopping on validation macro AUC, plus an Optuna search."""
import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .models import GEMXAI, FocalLoss
from .utils import device, log


def to_tensors(Xn, Xt, Xi, y=None):
    t = [torch.tensor(Xn, dtype=torch.float32), torch.tensor(Xt, dtype=torch.float32),
         torch.tensor(Xi, dtype=torch.float32).unsqueeze(1)]
    if y is not None:
        t.append(torch.tensor(y, dtype=torch.long))
    return t


@torch.no_grad()
def predict_proba(model, Xn, Xt, Xi, bs=256):
    model.eval()
    dev = device()
    tn, tt, ti = to_tensors(Xn, Xt, Xi)
    out = []
    for i in range(0, len(tn), bs):
        out.append(torch.softmax(model(tn[i:i + bs].to(dev), tt[i:i + bs].to(dev), ti[i:i + bs].to(dev)), -1).cpu())
    return torch.cat(out).numpy()


def macro_auc(y, p):
    """Validation score for early stopping: OvR AUC over classes present in y.
    Falls back to macro-F1 if the validation set contains a single class."""
    from .evaluate import robust_macro_auc
    from sklearn.metrics import f1_score
    auc = robust_macro_auc(y, p)
    return auc if auc == auc else f1_score(y, p.argmax(1), average="macro")


def class_alpha(y):
    c = np.bincount(y, minlength=3).astype(float)
    a = c.sum() / (3 * np.maximum(c, 1))
    return torch.tensor(a / a.mean(), dtype=torch.float32)


def train_model(cfg, train, val, modalities=("num", "text", "img"), lr=None, dropout=None,
                epochs=None, use_focal=True, verbose=True, modality_dropout=None):
    dev = device()
    if dropout is not None or modality_dropout is not None:
        cfg = copy.copy(cfg)
        if dropout is not None:
            cfg.dropout = dropout
        if modality_dropout is not None:
            cfg.modality_dropout = modality_dropout
    model = GEMXAI(cfg, n_num=train[0].shape[1], modalities=modalities).to(dev)
    alpha = class_alpha(train[3])
    crit = FocalLoss(alpha.to(dev), cfg.focal_gamma) if use_focal else torch.nn.CrossEntropyLoss(weight=alpha.to(dev))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr or cfg.lr, weight_decay=cfg.weight_decay)
    dl = DataLoader(TensorDataset(*to_tensors(*train)), batch_size=cfg.batch_size, shuffle=True)
    best, best_auc, wait = None, -1, 0
    for ep in range(epochs or cfg.epochs):
        model.train()
        tot = 0
        for xn, xt, xi, yb in dl:
            xn, xt, xi, yb = xn.to(dev), xt.to(dev), xi.to(dev), yb.to(dev)
            loss = crit(model(xn, xt, xi), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb)
        auc = macro_auc(val[3], predict_proba(model, *val[:3]))
        if verbose:
            log.info("[%s] epoch %02d loss %.4f val macro-AUC %.4f", "+".join(modalities), ep, tot / len(dl.dataset), auc)
        if auc > best_auc:
            best_auc, best, wait = auc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best is not None:
        model.load_state_dict(best)
    return model, best_auc


def optuna_search(cfg, train, val, n_trials=20):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        dp = trial.suggest_float("dropout", 0.1, 0.5)
        md = trial.suggest_float("modality_dropout", 0.0, 0.5)
        _, auc = train_model(cfg, train, val, lr=lr, dropout=dp, modality_dropout=md,
                             epochs=min(cfg.epochs, 15), verbose=False)
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=n_trials)
    log.info("Optuna best %s (val AUC %.4f)", study.best_params, study.best_value)
    return study
