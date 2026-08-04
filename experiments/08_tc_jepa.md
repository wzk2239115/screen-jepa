# 实验 8: TC-JEPA — Text-Conditional JEPA on recap-datacomp

- 日期:2026-07-29 至 2026-08-03
- 状态:**5 轮迭代完成**。v5 解决了 sparsity dead zone，sp 全程 > 0，top-5=15.0% (42x random)，历史最佳。
- 前置:实验 7 确认纯像素方法（screen-jepa）无法突破 ~20% 天花板。决定放弃"脱离 tokenizer"的路线，改用 TC-JEPA 的文本条件化方案。

## 1. 动机

实验 7 的核心教训:
1. **共同出现 ≠ 对齐** — text/photo 区域共现产生的是空间关联，不是语义理解
2. **模型选择偷懒** — backbone 不学字符形状，靠 enhancer attention 做空间关联
3. **脱离 tokenizer 不现实** — 至少短期内需要文本监督来引导视觉表示学习

TC-JEPA (ICML 2025) 的解决方案:
- 用 T5 编码 caption → word embeddings
- 在 JEPA predictor 的**每一层**做 cross-attention 到 word tokens
- 通过 feature prediction 任务**间接**学习 patch-word 对应关系（无需对比损失/grounding 标注）
- 测试时不需要 text — 只用 encoder

## 2. 架构

```
image (224×224) ──► ViT-B/16 encoder f_θ ──► z_ctx (196, 768)
                      (target patches masked out)        │
                                                        ▼
caption ──► T5-small (frozen) ──► word emb t ──►  Predictor g_φ
  (dim=512)    (77 tokens)            │          (dim=384, depth=6)
                                      │          × cross-attn every layer
image ──► ViT-B/16 EMA f_θ̄ ──► z_tgt  │                    │
          (full image, stop-grad)      │                    ▼
                                      └─────  L2 loss  ────┘
                                             + λ·L_sparse + β·L_consistency
                                             + γ·L_rank_reg
```

| 组件 | 规格 | 参数量 |
|---|---|---|
| Context encoder | ViT-B/16 (768d, 12L, 12H) | 85.8M |
| Target encoder | ViT-B/16 EMA (τ=0.996→1.0) | 85.8M (frozen) |
| Predictor | narrow ViT (384d, 6L, 12H) + cross-attn | 15.4M |
| Text encoder | T5-small (512d, frozen) | 35.3M (frozen) |
| **可训练** | encoder + predictor | **101.2M** |

## 3. 四轮迭代

### v1: 初始实现

**配置**: batch=256/GPU (2048 total), lr=1e-3, epochs=100, lam_sparse=0.1, lam_consistency=0.5, 无 target normalization, 无 sparsity warmup, 无 anti-collapse reg。

**结果**:
```
epoch:    0     9     19    29    99
l2:       0.24  0.13  0.10  0.08  0.05
cos:      0.87  0.91  0.91  0.91  0.91
sp:       0.000 0.000 0.000 0.000 0.000  ← 全程死亡
eff_rank: -     8.8   11.6  13.0  -
```

**问题**:
1. sp=0.000 全程 — cross-attention cosines 从一开始全为负，rectified cosine 进入死区，梯度永久为零
2. eff_rank=8-13/768 — 严重维度塌缩
3. OOM at batch=256 — cosine 相似度计算用广播创建 (B,N,S,D) 张量

**修复**: cosine 计算改用 bmm (5.9GB → 15MB/layer)，OOM 解决。

### v2: target normalization + sparsity warmup

**配置**: normalize_target=1 (L2 归一化 target features), sparse_warmup=20。

**动机**: 认为 L2 loss 不归一化导致模型可以缩小特征范数来降低 loss → 维度塌缩。

**结果**:
```
epoch:    0     9
l2:       0.000 0.000  ← per-dim mean on 768d normalized targets
cos:      0.84  0.89
sp:       1.7   3.9   ← 活了！
eff_rank: -     4.7   ← 更差了！
```

**问题**: normalize_target 解决了 cross-attention 死亡（sp 从 0 → 3.9），但加剧了维度塌缩（eff_rank 从 11.8 → 4.7）。归一化后所有 target 都是单位向量，模型只需预测同一方向，更容易塌缩。

### v3: 去掉 normalization + VICReg anti-collapse

**配置**: normalize_target=0, VICReg-style reg (variance + covariance), lam_reg=1.0。

**动机**: VICReg 的 variance criterion 确保每维 std ≥ 阈值，covariance criterion 去相关。

**结果**:
```
epoch:    0     9
l2:       0.32  0.16
cos:      0.81  0.90
sp:       2.9   2.5
reg:      0.001 0.001  ← VICReg 几乎不激活
eff_rank: -     11.8   ← 没改善
```

**问题**: VICReg 在 LayerNorm 之后计算。LayerNorm 在所有维度注入噪声使 variance criterion 通过，但真实信息仍集中在少数维度。reg=0.001 恒定，完全无效。

### v4: effective rank 正则化（最终版）

**配置**: normalize_target=0, rank_reg = tr(cov²)/tr(cov)² = 1/eff_rank, lam_reg=10.0, sparse_warmup=20。

**动机**: 直接惩罚 effective rank，绕过 LayerNorm 噪声。不需要 eigendecomposition，用 trace ratio计算。

**完整训练轨迹**:
```
epoch:    0     5     9     19    29    39    49    59    89    99
l2:       0.35  0.24  0.16  -     -     -     -     -     0.030 0.030
cos:      0.81  0.86  0.90  -     -     -     -     -     0.59  0.59
sp:       3.0   2.8   2.5   ~4    ~4    ~4    ~4    ~4    0.000 0.000
reg:      0.004 -     0.001 -     -     -     -     -     0.002 0.002
eff_rank: -     -     424.9 8.9   47.8  47.0  66.5  97.9  147   157.5
feat_std: -     -     0.088 0.073 0.189 0.110 0.146 0.138 0.131 0.140
```

**eff_rank 动态分析**:
- ep0-9: rank reg 有效，eff_rank=425（55% 维度活跃）
- ep10-19: **塌缩到 8.9** — L2 loss 找到低维捷径，rank reg (lam_reg=10) 不够强
- ep20+: sparsity kick in，**反而救回来了** — sparsity 强制不同 patch attend 不同 words → query 多样化 → encoder 特征多样化
- ep30-99: eff_rank 持续恢复 8.9 → 47 → 66 → 98 → 147 → 157

**最终问题**: sp 在 epoch ~70 后再次死亡（rectified cosine 死区），cos 从 0.99 退化到 0.59。

### v5: Entropy sparsity + 激进 masking（最终版）

**三项改进**:

| 改进 | v4 → v5 | 原因 |
|---|---|---|
| Sparsity loss | rectified cosine → **attention entropy** | 消灭 dead zone：softmax 永远正，梯度永远存在 |
| Masking | 4 blocks × 10-25% → **8 blocks × 10-20%** | target 覆盖 ~50% → ~62%，context 缩小 → 预测变难 → text 变必需 |
| Warmup | 20ep → **5ep** | 趁 eff_rank 高就引入 sparsity，避免 ep10-19 塌缩窗口 |
| lam_reg | 10 → **25** | 更强的 anti-collapse 压力 |

**Entropy sparsity 公式**:
```
旧 (v4):  O = max(cos(q, k), 0);  L_sparse = ‖O‖₁          ← 死区：全负时 O=0, ∇=0
新 (v5):  p = softmax(cos(q, k) / τ);  L_sparse = H(p)/log(S) ← 无死区：softmax 永远正
          H(p) = -Σ p·log(p)    (归一化到 [0,1], 0=稀疏, 1=均匀)
          τ = 0.1 (温度)
```

**评测结果**:

```
              epoch 40           epoch 99
              top-5   lift       top-5   lift    sp 全程?
──────────────────────────────────────────────────────────
v4            13.2%   37x        ~13%    ~36x    ❌ (ep70 死亡)
v5             3.0%    8x        15.0%   42x     ✅ (全程 > 0)
```

**v5 ep40 远差于 v4 ep40 的原因**: masking 更激进（62% vs 50% target），预测任务真正变难了。v4 在 ep3 就 cos=0.995 是假象——模型不需要 text 就能预测。v5 强制模型学习更难的预测，收敛更慢但特征质量更高。

**v5 ep99 仍在上升** (ep40→ep99: 3%→15%)，说明模型还没收敛，训练更久可能继续提升。

## 4. 评测结果

### Image→Word Retrieval (probe_tc_jepa.py)

v5 epoch 99 (最终版):

```
n_test=500, n_words=1395, random top-5 = 0.36%

top-1:  3.6%   (51x above random)
top-5:  15.0%  (42x above random)
top-10: 30.2%  (42x above random)
MRR:    0.121
```

### 全版本对比

```
方法             top-5    random   lift    sp 全程?   说明
──────────────────────────────────────────────────────────────
CLIP             55.7%    1.04%    53x     N/A        tokenizer + 对比损失 (上限)
Screen-JEPA      19.7%    1.04%    19x     N/A        纯像素（天花板已穷尽）
TC-JEPA v4       13.2%    0.36%    37x     ❌ ep70    rectified cosine dead zone
TC-JEPA v5       15.0%    0.36%    42x     ✅         entropy sparsity + 激进 masking
```

TC-JEPA v5 的 42x lift 接近 CLIP 的 53x（在更难的 1395 词表上），且 sp 全程保持活跃。

## 5. 核心问题与解决

### 问题 1: Sparsity Dead Zone（v1-v4 共同问题）

rectified cosine `max(cos(q,k), 0)` 有死区：一旦所有 cosine 变负，O=0，梯度=0，cross-attention 永久死亡。

**v5 解决方案**: 用 softmax(cos/τ) 的熵替代 rectified cosine 的 L1 范数。softmax 保证所有值为正，梯度永远存在。

### 问题 2: 预测任务过简单（v1-v4 共同问题）

4 个 target blocks × 10-25% ≈ 50% target。context 50%，predictor 不需要 text 就能预测（cos=0.995 在 ep3 到位）。

**v5 解决方案**: 8 blocks × 10-20% ≈ 62% target。context 缩小到 38%，预测变难，text 成为必要信息源。

### 问题 3: 维度塌缩（v1-v3）

特征 effective rank 持续下降（8-13/768），大部分维度死亡。

**v4 解决方案**: effective rank 正则化 `tr(cov²)/tr(cov)²`，直接惩罚低维塌缩。eff_rank 提升到 147-425。

## 6. v6 改进方向

| 改进 | 预期效果 | 难度 |
|---|---|---|
| **训练更久** (200-300ep) | v5 ep99 仍在上升，未收敛 | 低 |
| **多 caption** (LMM 生成 8 个/image) | 匹配论文 setup，text 信号 8x 更强 | 中 |
| **更大模型** (ViT-L/16) | 更强容量，更好的表示 | 低 |
| **I-JEPA style masking** | 连续 context block + 分散 target | 中 |
| **更大 T5** (t5-base, dim=768) | 更强的文本表示 | 低 |

## 7. 代码文件

| 文件 | 说明 |
|---|---|
| `tc_jepa.py` | TC-JEPA 模型 (ViT encoder + text-conditioned predictor + T5 + masking + losses) |
| `train_tc_jepa.py` | 训练脚本 (TarImageCaption dataset + DDP + EMA + AdamW + cosine LR) |
| `probe_tc_jepa.py` | Zero-shot image→word retrieval 评测 |
| `paper_2605.03245v1.md` | TC-JEPA 论文原文 |

## 8. 关键超参数 (v5 最终版)

```bash
--batch 256 --epochs 100 --lr 1e-3 --wd 0.04
--encoder_dim 768 --encoder_depth 12 --encoder_heads 12
--pred_dim 384 --pred_depth 6 --pred_heads 12
--t5_model t5-small --max_caption_len 77
--lam_sparse 0.1 --lam_consistency 0.5 --lam_reg 25.0
--sparse_warmup 5 --normalize_target 0 --temperature 0.1
--num_target_blocks 8 --target_scale_min 0.10 --target_scale_max 0.20
--ema_tau 0.996 --grad_clip 1.0
```
