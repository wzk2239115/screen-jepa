# 实验 8: TC-JEPA — Text-Conditional JEPA on recap-datacomp

- 日期:2026-07-29 至 2026-07-30
- 状态:4 轮迭代完成，定位核心问题（sparsity dead zone + 预测任务过简单），待 v5 改进
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

## 4. 评测结果

### Image→Word Retrieval (probe_tc_jepa.py)

epoch 40 checkpoint (sp 还活着时):

```
n_test=500, n_words=1395, random top-5 = 0.36%

top-1:  1.6%   (23x above random)
top-5:  13.2%  (37x above random)
top-10: 27.2%  (38x above random)
MRR:    0.104
```

### 对比历史实验

```
方法             top-5    random   lift    说明
──────────────────────────────────────────────────────
CLIP             55.7%    1.04%    53x     tokenizer + 对比损失
Screen-JEPA best 19.7%    1.04%    19x     纯像素（天花板）
TC-JEPA ep40     13.2%    0.36%    37x     lift 介于两者之间
```

注意：TC-JEPA 评测词表更大（1395 vs ~300），random baseline 更低（0.36% vs 1.04%），所以 lift 是更公平的对比指标。TC-JEPA 的 37x lift 已经超过 screen-jepa 的 19x，说明文本条件化确实比纯像素方法更有效。

## 5. 核心问题定位

### 问题 1: Sparsity Dead Zone

rectified cosine `max(cos(q,k), 0)` 有死区：一旦所有 cosine 变负，O=0，梯度=0，cross-attention 永久死亡。

sparsity warmup（前 20 epoch λ=0）延迟了死亡（从 ep0 推到 ep~70），但没有解决根本问题。

### 问题 2: 预测任务过简单

4 个 target blocks × 10-25% scale ≈ 40-60% target 覆盖。context 占 40-60%，predictor 靠 self-attention + 位置信息就能预测（cos=0.995 在 ep3 就到了），text 非必需。

当 text 非必需时：
1. L2 loss 不提供维持 cross-attention 的梯度
2. sparsity loss 的最小代价是把所有 cosine 推负（trivially sparse = 全零）
3. → cross-attention 死亡

### 根因链

```
预测任务简单 → text 非必需 → sparsity 把 cosine 推负 → dead zone → 交叉注意力死亡 → cos 退化
                                                                    ↓
                                        L2 找到低维捷径 → eff_rank 塌缩
```

## 6. v5 改进方向

| 改进 | 具体方案 | 预期效果 |
|---|---|---|
| **更激进 masking** | target 占 60-70%，context 30-40% | 预测变难 → text 变必需 → cross-attention 保活 |
| **去掉 dead zone** | rectified cosine → attention entropy | 永远有梯度 → sparsity 不会死亡 |
| **缩短 warmup** | 20ep → 5ep | 趁 eff_rank 高就引入 sparsity，避免 ep10-19 的塌缩 |
| **多 caption** | 用 LMM 生成 8 个 caption/image | 匹配论文 setup，text 信号更强 |
| **增大 lam_reg** | 10 → 25-50 | 防止 ep10-19 的塌缩 |

## 7. 代码文件

| 文件 | 说明 |
|---|---|
| `tc_jepa.py` | TC-JEPA 模型 (ViT encoder + text-conditioned predictor + T5 + masking + losses) |
| `train_tc_jepa.py` | 训练脚本 (TarImageCaption dataset + DDP + EMA + AdamW + cosine LR) |
| `probe_tc_jepa.py` | Zero-shot image→word retrieval 评测 |
| `paper_2605.03245v1.md` | TC-JEPA 论文原文 |

## 8. 关键超参数 (v4 最终版)

```bash
--batch 256 --epochs 100 --lr 1e-3 --wd 0.04
--encoder_dim 768 --encoder_depth 12 --encoder_heads 12
--pred_dim 384 --pred_depth 6 --pred_heads 12
--t5_model t5-small --max_caption_len 77
--lam_sparse 0.1 --lam_consistency 0.5 --lam_reg 10.0
--sparse_warmup 20 --normalize_target 0
--num_target_blocks 4 --target_scale_min 0.10 --target_scale_max 0.25
--ema_tau 0.996 --grad_clip 1.0
```
