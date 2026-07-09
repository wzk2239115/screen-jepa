"""Predictive JEPA: masked feature prediction with an EMA target encoder.

Forces the encoder to encode enough about each word that its feature at a masked
location can be predicted from surrounding context — the mechanism through which
word-level semantics can emerge (unlike the global-invariance objective).

Pipeline (on convnext, which exposes a 14x14 spatial feature map):
  context  = encoder(masked_image)        -> (B, N, d)
  target   = ema_encoder(full_image)      -> (B, N, d)  [stop-grad]
  predictor input: context tokens with masked cells replaced by a learned
                   [MASK] token + 2D pos emb; predict target features at masked cells
  loss = MSE(predictor[mask], target[mask].detach())
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn

from backbones import TransformerBlock, build_encoder


def _ema_update(encoder, target, tau):
    with torch.no_grad():
        for p, t in zip(encoder.parameters(), target.parameters()):
            t.data.mul_(tau).add_(p.data, alpha=1 - tau)


class PredictiveJEPA(nn.Module):

    def __init__(self, arch="convnext", img_size=224, hidden=768, layers=12,
                 heads=12, patch=16, pred_depth=4, ema_tau=0.996, lam_sig=0.0):
        super().__init__()
        self.encoder = build_encoder(arch, img_size=img_size, dim=hidden,
                                     patch=patch, depth=layers, heads=heads)
        self.grid = getattr(self.encoder, "feature_grid", img_size // patch)
        self.feat_dim = getattr(self.encoder, "feature_dim", self.encoder.out_dim)
        self.arch = arch
        self.ema_tau = ema_tau
        self.lam_sig = lam_sig

        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, self.feat_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.feat_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.predictor = nn.Sequential(
            *[TransformerBlock(self.feat_dim, heads=max(1, self.feat_dim // 64))
              for _ in range(pred_depth)])
        self.pred_norm = nn.LayerNorm(self.feat_dim)

        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def encode(self, x):
        """Global embedding for eval/probe: mean of the trained stage features."""
        return self.encoder(x, return_map=True).mean(dim=1)

    def _features(self, enc, x):
        if hasattr(enc, "feature_dim"):
            return enc(x, return_map=True)
        h = enc(x)
        return h.unsqueeze(1).expand(-1, self.grid * self.grid, -1)

    def update_ema(self):
        _ema_update(self.encoder, self.target_encoder, self.ema_tau)

    def forward(self, full, masked, cell_mask, lam=None):
        lam = self.lam_sig if lam is None else lam
        with torch.no_grad():
            tgt = self.target_encoder(full, return_map=True).detach()
        ctx = self.encoder(masked, return_map=True)
        B, N, D = ctx.shape
        inp = ctx + self.pos
        m = cell_mask.unsqueeze(-1)
        inp = torch.where(m, self.mask_token.expand(B, N, D), inp)
        out = self.pred_norm(self.predictor(inp))

        mask = cell_mask
        pred = out[mask]
        target = tgt[mask]
        pred_loss = F.mse_loss(pred, target)

        stats = {
            "pred": pred_loss.detach(),
            "cos": F.cosine_similarity(pred, target, dim=-1).mean().detach(),
            "tgt_std": tgt.std(dim=0).mean().detach(),
            "mask_frac": cell_mask.float().mean().detach(),
        }
        if lam > 0:
            from model import SIGReg
            if not hasattr(self, "_sig"):
                self._sig = SIGReg().to(ctx.device)
            reg = self._sig(ctx.transpose(0, 1))
            stats["reg"] = reg.detach()
            return pred_loss + lam * reg, stats
        return pred_loss, stats


if __name__ == "__main__":
    m = PredictiveJEPA(arch="convnext", img_size=224, hidden=384, layers=6, heads=6)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    full = torch.randn(2, 3, 224, 224)
    masked = torch.randn(2, 3, 224, 224)
    cm = torch.zeros(2, m.grid * m.grid, dtype=torch.bool)
    cm[:, :20] = True
    loss, stats = m(full, masked, cm)
    loss.backward()
    print(f"trainable params(M)={n:.1f} grid={m.grid} feat_dim={m.feat_dim} "
          f"loss={loss.item():.4f} { {k: round(float(v),3) for k,v in stats.items()} }")
