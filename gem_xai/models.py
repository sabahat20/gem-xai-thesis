"""Phase 3: three-branch multimodal classifier with additive attention fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class NumBranch(nn.Module):
    """Branch A: 3-layer MLP on the 20 technical indicators -> 128-d."""
    def __init__(self, d_in, d_out=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(128, d_out), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class TextBranch(nn.Module):
    """Branch B: linear projection of the 4 daily FinBERT sentiment scores -> 64-d."""
    def __init__(self, d_in=4, d_out=64):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, x):
        return F.relu(self.proj(x))


class ImageBranch(nn.Module):
    """Branch C: ResNet-18; only layer3, layer4 and the projection head are trainable."""
    def __init__(self, d_out=128, pretrained=True):
        super().__init__()
        w = None
        if pretrained:
            try:
                w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                net = torchvision.models.resnet18(weights=w)
            except Exception:  # offline
                net = torchvision.models.resnet18(weights=None)
        else:
            net = torchvision.models.resnet18(weights=None)
        # Greyscale input: average the pretrained RGB filters into one channel.
        conv = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        conv.weight.data = net.conv1.weight.data.mean(1, keepdim=True)
        net.conv1 = conv
        net.fc = nn.Identity()
        for name, p in net.named_parameters():
            p.requires_grad = name.startswith(("layer3", "layer4"))
        self.backbone = net
        self.head = nn.Sequential(nn.Linear(512, d_out), nn.ReLU())

    def forward(self, x):
        return self.head(self.backbone(x))


class AttentionFusion(nn.Module):
    """Additive (Bahdanau-style) attention over modality embeddings."""
    def __init__(self, dims, d=128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(k, d) for k in dims])
        self.W = nn.Linear(d, d)
        self.v = nn.Linear(d, 1, bias=False)

    def forward(self, embs, keep=None):
        H = torch.stack([p(e) for p, e in zip(self.proj, embs)], 1)       # B x M x d
        s = self.v(torch.tanh(self.W(H))).squeeze(-1)                    # B x M
        if keep is not None:                                             # modality dropout mask
            s = s.masked_fill(~keep, float("-inf"))
        a = torch.softmax(s, 1)
        return (a.unsqueeze(-1) * H).sum(1), a


class GEMXAI(nn.Module):
    """modalities: any subset of ('num','text','img'); single-modality models are baselines."""
    def __init__(self, cfg, n_num=20, n_text=4, n_classes=3, modalities=("num", "text", "img")):
        super().__init__()
        self.modalities = tuple(modalities)
        dims = []
        if "num" in self.modalities:
            self.num = NumBranch(n_num, cfg.num_emb, cfg.dropout); dims.append(cfg.num_emb)
        if "text" in self.modalities:
            self.text = TextBranch(n_text, cfg.text_emb); dims.append(cfg.text_emb)
        if "img" in self.modalities:
            self.img = ImageBranch(cfg.img_emb, cfg.pretrained_resnet); dims.append(cfg.img_emb)
        self.fusion = AttentionFusion(dims, cfg.fusion_dim)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.fusion_dim, n_classes))
        self.modality_dropout = float(getattr(cfg, "modality_dropout", 0.0))
        self.last_attention = None

    def forward(self, x_num, x_text, x_img):
        embs = []
        if "num" in self.modalities: embs.append(self.num(x_num))
        if "text" in self.modalities: embs.append(self.text(x_text))
        if "img" in self.modalities: embs.append(self.img(x_img))
        keep = None
        M = len(embs)
        if self.training and self.modality_dropout > 0 and M > 1:
            B = embs[0].shape[0]
            keep = torch.rand(B, M, device=embs[0].device) >= self.modality_dropout
            none = ~keep.any(1)                                    # never drop every modality
            if none.any():
                pick = torch.randint(0, M, (int(none.sum()),), device=keep.device)
                keep[none] = torch.nn.functional.one_hot(pick, M).bool()
        z, a = self.fusion(embs, keep)
        self.last_attention = a.detach()
        return self.head(z)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else torch.ones(3))
        self.gamma = gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits, -1)
        logpt = logp.gather(1, y[:, None]).squeeze(1)
        pt = logpt.exp()
        return (-self.alpha[y] * (1 - pt) ** self.gamma * logpt).mean()
