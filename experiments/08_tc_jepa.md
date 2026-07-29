# 实验 8: TC-JEPA — Text-Conditional JEPA on recap-datacomp

- 日期:2026-07-29
- 状态:代码实现完成，待部署训练
- 前置:实验 7 确认纯像素方法（screen-jepa）无法突破 ~20% 天花板，瓶颈是 backbone 无法从 16×16 patch 提取文字形状。决定放弃"脱离 tokenizer"的路线，改用 TC-JEPA 的文本条件化方案。

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

关键差异:我们的 screen-jepa 尝试从**渲染文本像素**学习词义（失败），TC-JEPA 直接用 **tokenizer 编码文本**作为条件信号引导视觉表示学习。

## 2. 方法

### 2.1 架构

```
image (224×224) ──► ViT-B/16 encoder f_θ ──► z_ctx (196, 768)
                      (target patches masked out)        │
                                                        ▼
caption ──► T5-small (frozen) ──► word emb t ──►  Predictor g_φ
  (dim=512)    (77 tokens)            │          (dim=384, depth=6)
                                      │          × cross-attn every layer
image ──► ViT-B/16 EMA f_θ̄ ──► z_tgt  │                    │
          (full image, stop-grad)      │                    ▼
                                      │              ẑ_tgt (predicted)
                                      │                    │
                                      └─────  L2 loss  ────┘
                                             + λ·L_sparse + β·L_consistency
```

### 2.2 核心组件

| 组件 | 规格 | 说明 |
|---|---|---|
| Context encoder | ViT-B/16 (768d, 12L, 12H) | target patches 用 mask_token 替换 |
| Target encoder | ViT-B/16 EMA (τ=0.996→1.0) | 全图编码，stop-grad |
| Predictor | narrow ViT (384d, 6L, 12H) | 每层 self-attn + cross-attn + MLP |
| Text encoder | T5-small (512d, frozen) | 编码 caption → 77 word tokens |
| Cross-attention | W_Q(384→384), W_K/V(512→384) | 每层独立投影 |

### 2.3 损失函数

```
L = L_predict + λ · L_sparse + β · L_consistency

L_predict = (1/|B_y|) Σ MSE(ẑ_yj, sg(z_yj))        λ = 0.1
L_sparse = mean_patches mean_layers ‖max(cos(q, K), 0)‖₁    β = 0.5
L_consistency = mean_patches mean_layers ‖O^(l) - mean_l(O^(l))‖₁
```

- **L_predict**: 预测 masked patch 的 EMA target features
- **L_sparse**: 每个 patch 只 attend 少数 words（L1 正则化 rectified cosine）
- **L_consistency**: 同一 patch 在不同层的 word selection 保持一致

### 2.4 Masking

I-JEPA style 多块掩码:
- 4 个随机矩形 target blocks（scale 10-25%）
- Context = 非 target 区域的补集

### 2.5 与论文的差异

| 方面 | 论文 | 本实现 |
|---|---|---|
| Dataset | IN-1k/21k, YFCC15M, CC12M | recap-datacomp-384-1M (~356k pairs) |
| Captions/image | 8 (ShareGPT4V generated) | 1 (DataComp original) |
| Epochs | 300-1200 | 100 (快速验证) |
| Batch size | 2048 | 1024 (128×8 GPU) |
| Text encoder | T5 (variant unclear) | T5-small (60M, dim=512) |
| Image size | 224 | 224 |
| Encoder mask | I-JEPA (remove target patches) | mask_token 替换（编码器仍处理全序列） |

## 3. 实验计划

### v1: 基线 TC-JEPA

```bash
torchrun --nproc_per_node=8 --rdzv-endpoint 127.0.0.1:29500 \
    train_tc_jepa.py \
    --tar_dir /home/jovyan/h800fast/wangzekai/recap-datacomp-384-1M \
    --num_tars 81 \
    --batch 128 --epochs 100 --lr 1e-3 --wd 0.04 \
    --lam_sparse 0.1 --lam_consistency 0.5 \
    --out outputs/tcjepa_v1
```

### 关注指标

| 指标 | 期望 | 说明 |
|---|---|---|
| train/l2 | ↓ 稳定下降 | 预测误差 |
| train/cos_pt | ↑ 接近 1 | 预测-目标余弦相似度 |
| train/sparse | > 0 | cross-attention 是否有意义（=0 表示退化） |
| eval/feat_std | > 0.1 | 特征是否塌缩 |
| eval/eff_rank | > 100 / 768 | 特征有效维度 |

### 潜在问题及预案

1. **Sparse loss → 0** (smoke test 中观察到): 如果 cross-attention cosines 全变负，sparsity 退化。预案: warmup λ（前 10 epoch λ=0）
2. **特征塌缩**: EMA + stop-grad 应该足够。如果不够，加 VICReg-style variance/invariance 正则
3. **单 caption 信息不足**: N=1 可能不如论文的 N=8。预案: 后续用 LMM 生成多 caption

## 4. 代码文件

- `tc_jepa.py`: TC-JEPA 模型（ViT encoder + text-conditioned predictor + T5 + masking + losses）
- `train_tc_jepa.py`: 训练脚本（TarImageCaption dataset + DDP + EMA + AdamW + cosine LR）

## 5. 本地验证

- 模型 forward + backward: ✅ 通过
- ViT-B/16 batch=32: 16.1GB GPU mem
- ViT-B/16 batch=128: 61.2GB GPU mem → H800 80GB 可用
- Data loading + T5 tokenization: ✅ 通过
- 2 epoch smoke test (tiny model, 1 tar): ✅ 通过
