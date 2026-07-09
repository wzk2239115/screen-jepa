"""Encoder backbones for screen-jepa, all from-scratch (no pretrained), pure torch.

Each exposes .out_dim and forward(x) -> (B, out_dim) global embedding.

Architectures:
  - convnext : pure ConvNeXt (hierarchical conv, strongest local features, simplest)
  - convvit  : conv stem (compress /16) + transformer over tokens
  - windowvit: SAM-like: conv patchify + windowed attention + periodic global
"""
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim, eps=eps)

    def forward(self, x):
        return self.norm(x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Linear(dim, mlp_ratio * dim)
        self.pw2 = nn.Linear(mlp_ratio * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x):
        r = x
        x = self.norm(self.dw(x))
        x = x.permute(0, 2, 3, 1)
        x = self.pw2(F.gelu(self.pw1(x)))
        x = x.permute(0, 3, 1, 2)
        return r + self.gamma.view(1, -1, 1, 1) * x


class ConvNeXt(nn.Module):
    """Pure ConvNeXt: hierarchical conv stages -> global avg pool."""

    def __init__(self, img_size=224, dims=(64, 128, 256, 512), depths=(2, 2, 6, 2), in_chans=3):
        super().__init__()
        self.out_dim = dims[-1]
        self.down_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], 4, stride=4), LayerNorm2d(dims[0]))
        self.down_layers.append(stem)
        for i in range(3):
            self.down_layers.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], 2, stride=2), LayerNorm2d(dims[i + 1])))
        self.stages = nn.ModuleList([
            nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            for i in range(4)])
        self.final_norm = LayerNorm2d(dims[-1])

    def forward(self, x):
        for i in range(4):
            x = self.down_layers[i](x)
            x = self.stages[i](x)
        x = self.final_norm(x)
        return x.mean(dim=(2, 3))


class Attention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=64):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.proj = nn.Linear(inner, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange_t(t, self.heads) for t in qkv)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.proj(o)


def rearrange_t(t, h):
    return t.reshape(t.shape[0], t.shape[1], h, -1).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=6, dim_head=64, mlp_ratio=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


def conv_stem(in_chans, dim, img_size, down=16):
    """Conv frontend compressing spatial by `down`. Returns (module, out_tokens_grid)."""
    assert img_size % down == 0, f"img_size {img_size} not divisible by {down}"
    layers = []
    c = in_chans
    cur = 1
    while cur < down:
        step = min(2, down // cur)
        layers += [nn.Conv2d(c, dim, step, stride=step), LayerNorm2d(dim)]
        c = dim
        cur *= step
    return nn.Sequential(*layers), img_size // down


class ConvViT(nn.Module):
    """Conv stem (compress /16) -> transformer over tokens + CLS."""

    def __init__(self, img_size=224, dim=384, depth=8, heads=6, dim_head=64, in_chans=3):
        super().__init__()
        self.out_dim = dim
        self.stem, g = conv_stem(in_chans, dim, img_size, down=16)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, g * g + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, dim_head) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        f = self.stem(x)
        t = f.flatten(2).transpose(1, 2)
        c = self.cls.expand(t.shape[0], -1, -1)
        t = torch.cat([c, t], dim=1) + self.pos
        t = self.blocks(t)
        return self.norm(t[:, 0])


class WindowAttention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=64, ws=7):
        super().__init__()
        self.ws = ws
        self.heads = heads
        inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.proj = nn.Linear(inner, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        ws = self.ws
        x = x.transpose(1, 2).reshape(B, C, H, W)
        # pad to multiple of ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.reshape(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B * (Hp // ws) * (Wp // ws), ws * ws, C)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2) for t in qkv)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        o = self.proj(o)
        o = o.reshape(B, Hp // ws, Wp // ws, ws, ws, -1)
        o = o.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, Hp * Wp)
        if pad_h or pad_w:
            o = o[:, :, : H * W]
        return o.transpose(1, 2)


class WindowBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ws=7, mlp_ratio=4, windowed=True):
        super().__init__()
        self.windowed = windowed
        self.ws = ws
        self.n1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, dim_head, ws) if windowed else Attention(dim, heads, dim_head)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x, H, W):
        cls, spatial = x[:, :1], x[:, 1:]
        if self.windowed:
            spatial = spatial + self.attn(self.n1(spatial), H, W)
        else:
            both = torch.cat([cls, spatial], dim=1)
            both = both + self.attn(self.n1(both))
            cls, spatial = both[:, :1], both[:, 1:]
        both = torch.cat([cls, spatial], dim=1)
        both = both + self.mlp(self.n2(both))
        return both


class WindowViT(nn.Module):
    """SAM-like: conv patchify + windowed attention + periodic global attention."""

    def __init__(self, img_size=224, patch=16, dim=384, depth=8, heads=6, dim_head=64,
                 ws=7, global_every=4, in_chans=3):
        super().__init__()
        assert img_size % patch == 0
        self.out_dim = dim
        g = img_size // patch
        self.grid = g
        self.patch = nn.Conv2d(in_chans, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, g * g + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            WindowBlock(dim, heads, dim_head, ws, windowed=((i + 1) % global_every != 0))
            for i in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        f = self.patch(x)
        t = f.flatten(2).transpose(1, 2)
        c = self.cls.expand(t.shape[0], -1, -1)
        t = torch.cat([c, t], dim=1) + self.pos
        for blk in self.blocks:
            t = blk(t, self.grid, self.grid)
        return self.norm(t[:, 0])


def build_encoder(arch, img_size=224, dim=384, patch=16, depth=8, heads=6, in_chans=3):
    arch = arch.lower()
    if arch == "convnext":
        return ConvNeXt(img_size=img_size)
    if arch == "convvit":
        return ConvViT(img_size=img_size, dim=dim, depth=depth, heads=heads)
    if arch == "windowvit":
        return WindowViT(img_size=img_size, patch=patch, dim=dim, depth=depth,
                         heads=heads, ws=7, global_every=4)
    raise ValueError(f"unknown arch: {arch}")


if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)
    for name in ["convnext", "convvit", "windowvit"]:
        m = build_encoder(name, img_size=224, dim=384, patch=16, depth=8, heads=6)
        n = sum(p.numel() for p in m.parameters())
        y = m(x)
        print(f"{name:10s} out={tuple(y.shape)} out_dim={m.out_dim} params(M)={n/1e6:.2f}")
