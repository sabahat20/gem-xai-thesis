"""Phase 2: WGAN-GP augmentation of minority-class numerical feature vectors.

One WGAN-GP is trained per minority regime (Bear, HighVol) on the standardised
20-dimensional TRAINING feature vectors of that class. Synthetic quality is checked with:
  1. MMD (RBF kernel, median heuristic) between real and synthetic samples
     (compared with the MMD between two halves of the real data as a reference),
  2. a t-SNE plot of real vs synthetic,
  3. a real-vs-synthetic discriminator (5-fold logistic regression + gradient boosting);
     accuracy < 0.60 is the acceptance target from the proposal.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .utils import device, log


class Generator(nn.Module):
    def __init__(self, z, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z, 128), nn.LeakyReLU(0.2), nn.Linear(128, 128),
                                 nn.LeakyReLU(0.2), nn.Linear(128, d))

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.LeakyReLU(0.2), nn.Linear(128, 128),
                                 nn.LeakyReLU(0.2), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x)


def _gp(critic, real, fake):
    a = torch.rand(real.size(0), 1, device=real.device)
    x = (a * real + (1 - a) * fake).requires_grad_(True)
    g = torch.autograd.grad(critic(x).sum(), x, create_graph=True)[0]
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(X: np.ndarray, cfg, lambda_gp=None):
    dev = device()
    lambda_gp = lambda_gp or cfg.gan_lambda_gp
    d = X.shape[1]
    G, D = Generator(cfg.gan_latent, d).to(dev), Critic(d).to(dev)
    oG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    oD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    bs = min(64, len(X))
    hist = []
    for ep in range(cfg.gan_epochs):
        for _ in range(cfg.gan_n_critic):
            real = Xt[torch.randint(0, len(Xt), (bs,), device=dev)]
            fake = G(torch.randn(bs, cfg.gan_latent, device=dev)).detach()
            lossD = D(fake).mean() - D(real).mean() + lambda_gp * _gp(D, real, fake)
            oD.zero_grad(); lossD.backward(); oD.step()
        lossG = -D(G(torch.randn(bs, cfg.gan_latent, device=dev))).mean()
        oG.zero_grad(); lossG.backward(); oG.step()
        if ep % 200 == 0:
            hist.append((ep, lossD.item(), lossG.item()))
    return G, hist


def sample(G, n, cfg):
    with torch.no_grad():
        return G(torch.randn(n, cfg.gan_latent, device=device())).cpu().numpy()


def mmd_rbf(X, Y, max_n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(len(X), max_n), replace=False)]
    Y = Y[rng.choice(len(Y), min(len(Y), max_n), replace=False)]
    Z = np.vstack([X, Y])
    d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    sigma2 = np.median(d2[d2 > 0])
    K = np.exp(-d2 / sigma2)
    n = len(X)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())


def discriminator_accuracy(real, synth, seed=0):
    n = min(len(real), len(synth))
    rng = np.random.default_rng(seed)
    Xr = real[rng.choice(len(real), n, replace=False)]
    Xs = synth[rng.choice(len(synth), n, replace=False)]
    X = np.vstack([Xr, Xs]); y = np.r_[np.zeros(n), np.ones(n)]
    lr = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5).mean()
    gb = cross_val_score(GradientBoostingClassifier(random_state=seed), X, y, cv=5).mean()
    return float(lr), float(gb)


def augment(Xtr, ytr, cfg, classes=(1, 2)):
    """Returns synthetic X, y and a quality report. Synthetic rows never touch val/test."""
    counts = np.bincount(ytr, minlength=3)
    target = int(cfg.gan_target_ratio * counts.max())
    Xs, ys, report = [], [], {}
    for c in classes:
        Xc = Xtr[ytr == c]
        need = target - len(Xc)
        if need <= 0 or len(Xc) < 20:
            continue
        G, hist = None, None
        for lam in (cfg.gan_lambda_gp, 20.0):   # contingency: raise lambda to 20 on failure
            G, hist = train_wgan_gp(Xc, cfg, lambda_gp=lam)
            syn = sample(G, max(need, len(Xc)), cfg)
            acc_lr, acc_gb = discriminator_accuracy(Xc, syn)
            if max(acc_lr, acc_gb) < 0.60:
                break
        half = len(Xc) // 2
        ref = mmd_rbf(Xc[:half], Xc[half:])
        rep = {"n_real": int(len(Xc)), "n_synthetic": int(need), "lambda_gp": lam,
               "mmd_real_vs_synth": mmd_rbf(Xc, syn), "mmd_real_half_vs_half": ref,
               "disc_acc_logreg": acc_lr, "disc_acc_gboost": acc_gb,
               "passes_disc_test": bool(max(acc_lr, acc_gb) < 0.60), "loss_history": hist}
        report[c] = rep
        log.info("WGAN-GP class %d: %s", c, {k: v for k, v in rep.items() if k != "loss_history"})
        report[c]["_samples"] = syn[:need]
        # Proposal contingency: if quality tests still fail, fall back to focal loss only.
        rep["used_for_training"] = bool(rep["passes_disc_test"] or not cfg.gan_require_pass)
        if rep["used_for_training"]:
            Xs.append(syn[:need]); ys.append(np.full(need, c))
        else:
            log.warning("Class %d synthetic data failed the discriminator test - not used (focal loss only).", c)
    if not Xs:
        return np.empty((0, Xtr.shape[1])), np.empty(0, int), report
    return np.vstack(Xs), np.concatenate(ys), report
