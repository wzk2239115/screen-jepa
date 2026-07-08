import torch
import torch.nn.functional as F
from torch import nn
from transformers import ViTConfig, ViTModel


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (ported from lewm/module.py)."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj: (T, B, D)"""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None,
                 norm_fn=nn.BatchNorm1d, act_fn=nn.GELU):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class TextJEPA(nn.Module):
    """Two-view invariance world model.

    view_full  : the rendered sentence image
    view_masked: the same image with a few words erased

    Objective (VICReg-style): align z_full and z_masked while SIGReg keeps the
    embedding distribution from collapsing.
    """

    def __init__(self, hidden=384, layers=8, heads=6, mlp_dim=2048,
                 patch=16, img_size=224, embed_dim=None):
        super().__init__()
        cfg = ViTConfig(
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            intermediate_size=hidden * 4,
            patch_size=patch,
            image_size=img_size,
            num_channels=3,
        )
        self.encoder = ViTModel(cfg)
        self.projector = MLP(hidden, mlp_dim, embed_dim or hidden, norm_fn=nn.BatchNorm1d)
        self.sigreg = SIGReg()

    def encode(self, x):
        h = self.encoder(pixel_values=x).last_hidden_state[:, 0]
        return self.projector(h)

    def forward(self, full, masked):
        z_full = self.encode(full)
        z_masked = self.encode(masked)
        return z_full, z_masked

    def loss(self, full, masked, lam=0.1):
        z_full, z_masked = self.forward(full, masked)
        inv = F.mse_loss(z_full, z_masked)
        z = torch.stack([z_full, z_masked])  # (2, B, D)
        reg = self.sigreg(z)
        return inv + lam * reg, {
            "inv": inv.detach(),
            "reg": reg.detach(),
            "z_std": z_full.std(dim=0).mean().detach(),
        }


if __name__ == "__main__":
    m = TextJEPA(hidden=192, layers=4, heads=3, mlp_dim=1024, embed_dim=192)
    n = sum(p.numel() for p in m.parameters())
    print("params(M):", round(n / 1e6, 2))
    full = torch.randn(8, 3, 224, 224)
    masked = torch.randn(8, 3, 224, 224)
    loss, stats = m.loss(full, masked)
    print("loss:", float(loss), {k: float(v) for k, v in stats.items()})
