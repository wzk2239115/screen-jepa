"""TC-JEPA: Text-Conditional Joint-Embedding Predictive Architecture.

Reference: "Text-Conditional JEPA for Learning Semantically Rich Visual
Representations" (ICML 2025).

Architecture:
    image ──► context encoder f_θ ──► z_x (context patch features)
                                        │
    caption ──► T5 (frozen) ──► t ──────┤
                                        ▼
    mask tokens (target pos) ──► predictor g_φ (cross-attn to t at every layer)
                                        │
                                        ▼ ẑ_y (predicted target features)

    image ──► target encoder f_θ̄ (EMA) ──► z_y (ground-truth, stop-grad)

    Loss = L2(ẑ_y, z_y) + λ·L_sparse + β·L_consistency
"""

import math
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================= Vision Transformer =======================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, bias=True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, N, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Standard ViT encoder (no CLS token)."""

    def __init__(self, img_size=224, patch_size=16, dim=768, depth=12, heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x, target_mask=None):
        x = self.patch_embed(x) + self.pos_embed
        if target_mask is not None:
            m = target_mask.unsqueeze(-1)
            x = torch.where(m, self.mask_token.expand_as(x), x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ===================== Text-Conditioned Predictor =====================

class CrossAttention(nn.Module):
    """Cross-attention: patch features (Q) attend to text tokens (K, V).

    Returns (output, q_proj, k_proj) — the latter two are needed for the
    sparsity / consistency regularisation losses."""

    def __init__(self, dim, text_dim, heads):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(text_dim, dim, bias=False)
        self.w_v = nn.Linear(text_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, text, text_mask=None):
        q = self.w_q(x)       # (B, N, dim)
        k = self.w_k(text)    # (B, S, dim)
        v = self.w_v(text)    # (B, S, dim)

        B, N, D = q.shape
        S = k.shape[1]
        qh = q.reshape(B, N, self.heads, self.dim_head).transpose(1, 2)
        kh = k.reshape(B, S, self.heads, self.dim_head).transpose(1, 2)
        vh = v.reshape(B, S, self.heads, self.dim_head).transpose(1, 2)

        if text_mask is not None:
            valid = text_mask.bool()[:, None, None, :].expand(B, self.heads, N, S)
            out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=valid)
        else:
            out = F.scaled_dot_product_attention(qh, kh, vh)

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.to_out(out), q, k


class PredictorBlock(nn.Module):
    """Self-attention → cross-attention(text) → MLP."""

    def __init__(self, dim, text_dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, text_dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, text, text_mask):
        x = x + self.self_attn(self.norm1(x))
        cross_out, q, k = self.cross_attn(self.norm2(x), text, text_mask)
        x = x + cross_out
        x = x + self.mlp(self.norm3(x))
        return x, q, k


class TextCondPredictor(nn.Module):
    """Narrow ViT predictor with word-level cross-attention at every layer."""

    def __init__(self, dim, text_dim, depth, heads, num_patches, encoder_dim):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.ctx_proj = nn.Linear(encoder_dim, dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList(
            [PredictorBlock(dim, text_dim, heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, encoder_dim)

    def forward(self, ctx_features, target_mask, text, text_mask):
        B, N, _ = ctx_features.shape
        x = self.ctx_proj(ctx_features) + self.pos_embed      # context features
        token = self.mask_token + self.pos_embed               # target mask tokens
        m = target_mask.unsqueeze(-1)
        x = torch.where(m, token.expand(B, -1, -1), x)

        qs, ks = [], []
        for blk in self.blocks:
            x, q, k = blk(x, text, text_mask)
            qs.append(q)
            ks.append(k)

        x = self.norm(x)
        x = self.out_proj(x)
        return x, qs, ks


# ========================= T5 Text Encoder =========================

class T5TextEncoder(nn.Module):
    """Frozen T5 encoder → word-level embeddings for cross-attention."""

    def __init__(self, model_name="t5-small"):
        super().__init__()
        from transformers import T5Tokenizer, T5EncoderModel
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.dim = self.model.config.d_model
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (B, S, D_t)


# ====================== Multi-Block Masking ======================

def gen_target_mask(grid, num_blocks=4, scale_range=(0.10, 0.25), device="cpu"):
    """I-JEPA style: place *num_blocks* non-overlapping rectangular target
    blocks on a *grid*×*grid* lattice.  Context = complement of target."""
    occupied = torch.zeros(grid, grid, dtype=torch.bool, device=device)
    for _ in range(num_blocks):
        scale = random.uniform(*scale_range)
        area = max(1, int(grid * grid * scale))
        aspect = random.uniform(0.5, 2.0)
        h = max(1, min(grid, int(round(math.sqrt(area / aspect)))))
        w = max(1, min(grid, int(round(math.sqrt(area * aspect)))))
        top = random.randint(0, grid - h)
        left = random.randint(0, grid - w)
        occupied[top:top + h, left:left + w] = True
    return occupied.flatten()


def gen_batch_masks(batch_size, grid, device="cpu", **kw):
    return torch.stack(
        [gen_target_mask(grid, device=device, **kw) for _ in range(batch_size)]
    )


# ============================== TC-JEPA ==============================

class TCJEPA(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        pred_dim=384,
        pred_depth=6,
        pred_heads=12,
        t5_model="t5-small",
        lam_sparse=0.1,
        lam_consistency=0.5,
        lam_reg=1.0,
        ema_tau=0.996,
        target_scale=(0.10, 0.25),
        num_target_blocks=4,
        normalize_target=True,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            img_size, patch_size, encoder_dim, encoder_depth, encoder_heads
        )
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2

        # EMA target encoder (stop-grad)
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # T5 (frozen)
        self.text_encoder = T5TextEncoder(t5_model)

        # Text-conditioned predictor
        self.predictor = TextCondPredictor(
            pred_dim, self.text_encoder.dim, pred_depth, pred_heads,
            self.num_patches, encoder_dim,
        )

        self.lam_sparse = lam_sparse
        self.lam_consistency = lam_consistency
        self.lam_reg = lam_reg
        self.ema_tau = ema_tau
        self.target_scale = target_scale
        self.num_target_blocks = num_target_blocks
        self.encoder_dim = encoder_dim
        self.normalize_target = normalize_target

    # ---- VICReg-style anti-collapse --------------------------------
    @staticmethod
    def _vicreg(z):
        """Scale-invariant variance + covariance regularization.
        z: (B, N, D) → normalised, then checked for isotropy."""
        z = F.normalize(z.reshape(-1, z.size(-1)), dim=-1)
        D = z.size(1)
        N = max(z.size(0) - 1, 1)
        std = z.std(dim=0)
        var = F.relu(1.0 / math.sqrt(D) - std).mean()
        zc = z - z.mean(dim=0)
        cov = zc.T @ zc / N
        cov_loss = cov.fill_diagonal_(0).pow(2).sum() / D
        return var + cov_loss

    # ---- EMA -------------------------------------------------------
    @torch.no_grad()
    def update_ema(self, epoch=0, total_epochs=1):
        progress = min(epoch / max(total_epochs, 1), 1.0)
        tau = self.ema_tau + (1.0 - self.ema_tau) * (1 - math.cos(math.pi * progress)) / 2
        for p, t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            t.data.mul_(tau).add_(p.data, alpha=1 - tau)

    # ---- Loss helpers ----------------------------------------------
    @staticmethod
    def _rectified_cos(q, k, text_mask):
        """Rectified cosine similarity  O_{i,s} = max(cos(q_i, k_s), 0).

        q: (B, N, D)  k: (B, S, D)  text_mask: (B, S)
        → (B, N, S)   Memory-efficient: normalized bmm instead of broadcast."""
        qn = F.normalize(q.float(), dim=-1)         # (B, N, D)
        kn = F.normalize(k.float(), dim=-1)          # (B, S, D)
        cos = torch.bmm(qn, kn.transpose(1, 2))      # (B, N, S)
        cos = F.relu(cos)
        if text_mask is not None:
            cos = cos * text_mask.unsqueeze(1).float()
        return cos

    # ---- Forward ---------------------------------------------------
    def forward(self, images, input_ids, attention_mask, lam_sparse=None,
                lam_consistency=None, lam_reg=None):
        B = images.size(0)
        device = images.device
        lam_sparse = self.lam_sparse if lam_sparse is None else lam_sparse
        lam_consistency = self.lam_consistency if lam_consistency is None else lam_consistency
        lam_reg = self.lam_reg if lam_reg is None else lam_reg

        # 1. masks
        target_masks = gen_batch_masks(
            B, self.grid, device=device,
            num_blocks=self.num_target_blocks,
            scale_range=self.target_scale,
        )  # (B, N) bool

        # 2. context encoder (target patches masked out)
        z_ctx = self.encoder(images, target_mask=target_masks)   # (B, N, D)

        # 3. target encoder (full image, stop-grad)
        with torch.no_grad():
            z_tgt = self.target_encoder(images)                   # (B, N, D)
            if self.normalize_target:
                z_tgt = F.normalize(z_tgt, dim=-1)

        # 4. text embeddings (T5 frozen)
        with torch.no_grad():
            text_emb = self.text_encoder(input_ids, attention_mask)  # (B, S, D_t)

        # 5. predictor
        pred, qs, ks = self.predictor(z_ctx, target_masks, text_emb, attention_mask)

        # 6. L2 predictive loss (only at target positions)
        se = ((pred - z_tgt.detach()) ** 2).mean(dim=-1)  # (B, N)
        n_tgt = target_masks.sum().clamp(min=1)
        l2_loss = (se * target_masks).sum() / n_tgt

        # cosine similarity between pred and target (diagnostic)
        with torch.no_grad():
            cos_pt = F.cosine_similarity(
                pred[target_masks], z_tgt[target_masks], dim=-1
            ).mean()

        # 7. VICReg anti-collapse on encoder output
        reg_loss = self._vicreg(z_ctx)

        # 8. sparsity + consistency over all layers
        Os = [self._rectified_cos(q, k, attention_mask) for q, k in zip(qs, ks)]
        L = len(Os)

        sparse_loss = sum(O.sum(dim=-1).mean() for O in Os) / L

        O_mean = sum(Os) / L
        consistency_loss = sum(
            (O - O_mean).abs().sum(dim=-1).mean() for O in Os
        ) / L

        loss = l2_loss + lam_reg * reg_loss + lam_sparse * sparse_loss + lam_consistency * consistency_loss

        stats = {
            "l2": l2_loss.detach(),
            "sparse": sparse_loss.detach(),
            "consistency": consistency_loss.detach(),
            "reg": reg_loss.detach(),
            "cos_pt": cos_pt.detach(),
            "tgt_ratio": target_masks.float().mean().detach(),
        }
        return loss, stats
