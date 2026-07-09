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

    def __init__(self, img_size=224, base_dim=96, depths=(2, 2, 6, 2), in_chans=3):
        super().__init__()
        dims = (base_dim, base_dim * 2, base_dim * 4, base_dim * 8)
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


def merge_2x2(x, H, W):
    """x: (B, H*W, C) -> (B, (H/2)*(W/2), 4C)."""
    B, N, C = x.shape
    x = x.transpose(1, 2).reshape(B, C, H, W)
    x = x.reshape(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, (H // 2) * (W // 2), 4 * C)


class Hiera(nn.Module):
    """Pyramidal transformer with global attention + patch merging (Hiera-style)."""

    def __init__(self, img_size=224, embed=96, depths=(2, 3, 18, 3),
                 num_heads=(3, 6, 12, 24), in_chans=3):
        super().__init__()
        self.out_dim = embed * 8
        self.patch = nn.Conv2d(in_chans, embed, 4, stride=4)
        self.stages = nn.ModuleList()
        self.down = nn.ModuleList()
        c = embed
        for i, (d, h) in enumerate(zip(depths, num_heads)):
            self.stages.append(nn.Sequential(
                *[TransformerBlock(c, heads=h, dim_head=64) for _ in range(d)]))
            if i < len(depths) - 1:
                self.down.append(nn.Linear(4 * c, 2 * c))
                c *= 2
        self.norm = nn.LayerNorm(c)
        self.grids = [img_size // (4 * (2 ** i)) for i in range(4)]

    def forward(self, x):
        x = self.patch(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for i, (stage, grid) in enumerate(zip(self.stages, self.grids)):
            x = stage(x)
            if i < len(self.stages) - 1:
                x = self.down[i](merge_2x2(x, grid, grid))
        return self.norm(x).mean(dim=1)


class SRAAttention(nn.Module):
    """Spatial Reduction Attention (PVT)."""

    def __init__(self, dim, heads=6, dim_head=64, sr_ratio=4):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.inner = inner
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.proj = nn.Linear(inner, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(inner, inner, sr_ratio, stride=sr_ratio)
            self.norm = LayerNorm2d(inner)

    def forward(self, x, H, W):
        B, N, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # each (B, N, inner)
        q = rearrange_t(q, self.heads)
        if self.sr_ratio > 1:
            k = self.norm(self.sr(k.transpose(1, 2).reshape(B, self.inner, H, W)))
            v = self.norm(self.sr(v.transpose(1, 2).reshape(B, self.inner, H, W)))
            k = k.flatten(2).transpose(1, 2)
            v = v.flatten(2).transpose(1, 2)
        k = rearrange_t(k, self.heads)
        v = rearrange_t(v, self.heads)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, N, -1)
        return self.proj(o)


class PVTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head=64, sr_ratio=4, mlp_ratio=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = SRAAttention(dim, heads, dim_head, sr_ratio)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x, H, W):
        x = x + self.attn(self.n1(x), H, W)
        x = x + self.mlp(self.n2(x))
        return x


class PVT(nn.Module):
    """Pyramidal Vision Transformer with Spatial Reduction Attention."""

    def __init__(self, img_size=224, dims=(80, 160, 400, 640),
                 depths=(3, 6, 14, 3), num_heads=(2, 4, 8, 10),
                 sr_ratios=(8, 4, 2, 1), in_chans=3):
        super().__init__()
        self.out_dim = dims[-1]
        self.patch = nn.Conv2d(in_chans, dims[0], 7, stride=4, padding=3)
        self.stages = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(
                *[PVTBlock(dims[i], num_heads[i], 64, sr_ratios[i]) for _ in range(depths[i])]))
            if i < 3:
                self.down.append(nn.Linear(4 * dims[i], dims[i + 1]))
        self.norm = nn.LayerNorm(dims[-1])
        self.grids = [img_size // (4 * (2 ** i)) for i in range(4)]

    def forward(self, x):
        x = self.patch(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for i, (stage, grid) in enumerate(zip(self.stages, self.grids)):
            x = stage_apply(stage, x, grid, grid)
            if i < 3:
                x = self.down[i](merge_2x2(x, grid, grid))
        return self.norm(x).mean(dim=1)


def stage_apply(stage, x, H, W):
    for blk in stage:
        x = blk(x, H, W)
    return x


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1), nn.SiLU(), nn.Conv2d(c // r, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(dim=(2, 3), keepdim=True))


class MBConv(nn.Module):
    def __init__(self, c_in, c_out, expand=4, stride=1, k=3):
        super().__init__()
        mid = c_in * expand
        self.pw1 = nn.Conv2d(c_in, mid, 1)
        self.dw = nn.Conv2d(mid, mid, k, stride, k // 2, groups=mid)
        self.se = SE(mid)
        self.pw2 = nn.Conv2d(mid, c_out, 1)
        self.skip = (c_in == c_out) and (stride == 1)

    def forward(self, x):
        r = x
        x = F.silu(self.pw1(x))
        x = F.silu(self.dw(x))
        x = self.se(x)
        x = self.pw2(x)
        return x + r if self.skip else x


class Efficient(nn.Module):
    """EfficientNet-style: MBConv + SE, pyramidal conv stages."""

    def __init__(self, img_size=224, widths=(80, 128, 256, 448, 816),
                 depths=(2, 4, 8, 10, 3), in_chans=3):
        super().__init__()
        self.out_dim = widths[-1]
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, widths[0], 3, stride=2, padding=1), LayerNorm2d(widths[0]))
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.downs.append(nn.Sequential(
                nn.Conv2d(widths[i], widths[i + 1], 2, stride=2), LayerNorm2d(widths[i + 1])))
            self.stages.append(nn.Sequential(
                *[MBConv(widths[i + 1], widths[i + 1]) for _ in range(depths[i + 1])]))
        self.norm = LayerNorm2d(widths[-1])

    def forward(self, x):
        x = self.stem(x)
        for d, s in zip(self.downs, self.stages):
            x = d(x)
            x = s(x)
        return self.norm(x).mean(dim=(2, 3))


class Retina(nn.Module):
    """Dual-path: high-res conv (fovea) + low-res transformer (periphery), fused."""

    def __init__(self, img_size=224, dim=384, depth=6, heads=6, in_chans=3):
        super().__init__()
        self.out_dim = dim * 2
        c = dim // 2
        self.local_stem = nn.Sequential(
            nn.Conv2d(in_chans, c, 4, stride=4), LayerNorm2d(c))
        self.local = nn.Sequential(*[ConvNeXtBlock(c) for _ in range(6)])
        # global path: conv stem /16 -> transformer
        self.glob_stem, g = conv_stem(in_chans, dim, img_size, down=16)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, g * g + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.glob = nn.Sequential(*[TransformerBlock(dim, heads=heads) for _ in range(depth)])
        self.glob_norm = nn.LayerNorm(dim)
        self.fuse = nn.Linear(c + dim, self.out_dim)

    def forward(self, x):
        l = self.local(self.local_stem(x)).mean(dim=(2, 3))
        gg = self.glob_stem(x).flatten(2).transpose(1, 2)
        c = self.cls.expand(gg.shape[0], -1, -1)
        gg = torch.cat([c, gg], dim=1) + self.pos
        gg = self.glob_norm(self.glob(gg)[:, 0])
        return self.fuse(torch.cat([l, gg], dim=-1))


class S4D(nn.Module):
    """Diagonal state-space block (S4D), pure-torch, conv-mode."""

    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d = d_model
        log_A = torch.randn(d_state) * 0.1 - 5
        C = torch.randn(d_model, d_state)
        self.log_A = nn.Parameter(log_A)
        self.C = nn.Parameter(C)
        self.D = nn.Parameter(torch.ones(d_model))
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, L, d)
        z = self.in_proj(x)
        L = z.shape[1]
        A = (-self.log_A.exp()).exp()  # A in (0,1), per state
        powers = A.unsqueeze(0) ** torch.arange(L, device=x.device).unsqueeze(1)  # (L, d_state)
        k = z.new_zeros(L, self.d)
        for s in range(A.shape[0]):
            k = k + torch.outer(powers[:, s], self.C[:, s])  # (L, d)
        y = F.conv1d(z.transpose(1, 2), k.transpose(0, 1).unsqueeze(1).expand(self.d, 1, L),
                     groups=self.d, padding=L - 1)[..., :L].transpose(1, 2)
        y = y + self.D * z
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=64, mlp_ratio=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fwd = S4D(dim, d_state)
        self.bwd = S4D(dim, d_state)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x):
        h = self.norm(x)
        x = x + self.fwd(h) + self.bwd(torch.flip(h, dims=[1])).flip(dims=[1])
        x = x + self.mlp(self.norm2(x))
        return x


class MambaViT(nn.Module):
    """State-space (S4D) backbone over serialized patches, bidirectional."""

    def __init__(self, img_size=224, patch=16, dim=384, depth=12, in_chans=3):
        super().__init__()
        self.out_dim = dim
        g = img_size // patch
        self.patch = nn.Conv2d(in_chans, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, g * g, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.Sequential(*[MambaBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        t = self.blocks(t)
        return self.norm(t).mean(dim=1)


def build_encoder(arch, img_size=224, dim=384, patch=16, depth=8, heads=6, in_chans=3):
    arch = arch.lower()
    if arch == "convnext":
        return ConvNeXt(img_size=img_size, base_dim=max(48, dim // 4))
    if arch == "convvit":
        return ConvViT(img_size=img_size, dim=dim, depth=depth, heads=heads)
    if arch == "windowvit":
        return WindowViT(img_size=img_size, patch=patch, dim=dim, depth=depth,
                         heads=heads, ws=7, global_every=4)
    if arch == "hiera":
        return Hiera(img_size=img_size, embed=96)
    if arch == "pvt":
        return PVT(img_size=img_size)
    if arch == "retina":
        return Retina(img_size=img_size, dim=dim, depth=max(6, depth - 2), heads=heads)
    if arch == "efficient":
        return Efficient(img_size=img_size)
    if arch == "mamba":
        return MambaViT(img_size=img_size, patch=patch, dim=dim, depth=depth)
    raise ValueError(f"unknown arch: {arch}")


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    for name in ["convnext", "convvit", "windowvit", "hiera", "pvt", "retina",
                 "efficient", "mamba"]:
        try:
            m = build_encoder(name, img_size=224, dim=768, patch=16, depth=12, heads=12)
            n = sum(p.numel() for p in m.parameters())
            y = m(x)
            print(f"{name:10s} out={y.shape[-1]:4d} params(M)={n/1e6:6.1f}")
        except Exception as e:
            print(f"{name:10s} ERROR {type(e).__name__}: {e}")
