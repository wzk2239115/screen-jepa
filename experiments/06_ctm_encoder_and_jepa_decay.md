# 实验 6: CTM 增强编码器 + JEPA 权重退火（完整调参）

- 日期:2026-07-19 至 2026-07-28
- 结论:**JEPA 权重退火 (w_mse 0.3→0, ep10→20) 是最优训练方案。15+ 实验穷尽所有调参方向，天花板固定在 peak ~20% / floor ~14%。逐层探测 (实验 7) 定位根因：backbone 几乎不携带语义信息 (gap=0.017)，所有语义能力来自 CTM enhancer 的 attention 机制。瓶颈在 backbone 无法从 16×16 patch 中提取文字形状。**
- 状态:JEPA 调参穷尽，根因定位完成，下一步是结构改进

## 1. 动机

实验 4 的跨模态 JEPA (top-5=13.7%) 在 100 epoch 饱和，无法进一步提升。两个方向改善:

1. **CTM (Continuous Thought Machines)**:用迭代推理增强 encoder feature map，让 word 特征从"深思熟虑"的 feature map 中提取
2. **梯度冲突分析**:理解 JEPA + CLIP 多目标训练的退化机制，找到防退化方案

## 2. CTM 增强编码器设计 (train_ctm_enc_jepa.py)

### 架构
```
合成图 → ConvNeXt → initial feature map (B, 196, 768)
                        ↓
          CTM iterative enhancement
          (8 thought tokens × 50 ticks, attending to 196 cells)
                        ↓
          enhanced feature map (B, 196, 768)
                        ↓
    ┌──── cell pooling → 词/句特征 → CLIP/SigLIP 对齐
    └──── mask + Transformer predictor → JEPA MSE
```

### CTMEnhancer 核心
- **K=8 thought tokens** 初始化为可学习参数
- **T=50 ticks**:每 tick = attention(thoughts←cells) + synapse + NLM (neural lag memory)
- **Broadcast**:最终 cells query thoughts → enhanced feature map
- **Truncated BPTT**:每 15 ticks detach 一次，控制梯度传播
- 参数量:~120M (backbone 100M + enhancer 20M)

### 关键设计决策
- SigLIP loss (sigmoid 替代 softmax，小 batch 更好)
- EMA tau 退火 (0.996→1.0，I-JEPA 标准)
- 句子级对齐 (text global pool vs photo global，比词级更稳定)

## 3. 实验矩阵与结果

### 3.1 总览（20 组实验完整矩阵）

| 实验 | 配置 | Peak top-5 | Peak ep | ep99 | 退化? |
|---|---|---|---|---|---|
| Unified (always) | split_grad=False, w_mse=0.3 | **24%** | 0 | 7% | 严重 |
| Split_grad (word) | split_grad=True, w_mse=0.3 | 10.3% | 20 | 10% | 否 (低) |
| Split_grad (sent, 100ep) | sentence-level | 13.0% | 80 | 7% | 否 (低) |
| Split_grad + enhancer lr=2e-4 | lr_enhancer=2e-4 | 16.3% | 0 | 10% | 是→稳 |
| Two-stage (uni5) | unified_epochs=5 | **20.7%** | 10 | 10.7% | 中度 |
| Two-stage + enhancer lr=1e-4 | lr_enhancer=1e-4 | 15.7% | 15 | 12.0% | 中度 |
| Freeze5 | freeze_backbone_after=5 | 8.3% | 5 | 8.7% | 否 (更低) |
| Freeze10 | freeze_backbone_after=10 | 19.7% | 10 | 10.0% | 是→崩 |
| Pure CLIP+CTM | w_mse=0.0 | 13.7% | 15 | **13.7%** | **否!** |
| **JEPA decay 10→20** | decay w_mse 0.3→0 | **19.7%** | 10 | **13.0%** | **否! ← 最优** |
| Cyclic30 | w_mse_cycle=30 | 19.7% | 10 | 13.0% | 否 (无积累) |
| w_mse=0.5 + decay | 更强 JEPA | 18.0% | 10 | — | 是 (更差) |
| w_mse_min=0.05 | 残留 JEPA | 19.7% | 10 | — | 是 (floor 12%) |
| w_mse_min=0.1 | 残留 JEPA | **28.0%** | 20 | 13.7% | spike→崩 |
| Slow decay 10→40 | 30 epoch 衰减 | 19.7% | 10 | 11.0% | 否 (floor 低) |
| Always unified + decay 5→15 | enhancer 全程收 JEPA | 17.7% | 30 | 13.7% | 否 (波动) |
| Augment + decay 10→20 | 数据增强 | 19.3% | 20 | — | 否 (floor 15%) |
| Augment + decay 20→30 | 增强延迟 decay | 20.0% | 25 | 12.0% | 否 (波动) |
| uni10 + decay 15→25 | 更长 unified | 19.3% | 15 | 12.3% | 否 (floor 低) |
| jepa_decay5_uni | 全程 unified 退火 | 17.7% | 30 | 13.7% | 否 (波动) |

### 3.2 详细 top-5 趋势

#### Split gradient path (始终分离)
```
epoch:    0     5     10    20    30    49
top-5:    6.7%  5.7%  6.3%  10.3% 6.7%  10.0%   ← 稳定但低 (~8%)
```
Enhancer 只从 CLIP 学 (JEPA 不过 enhancer) → 不退化但天花板低。

#### Sentence-level split_grad (100 epoch)
```
epoch:    0     10    20    40    60    80    99
top-5:    9.3%  8.3%  9.3%  11.3% 10.0% 13.0% 7.0%   ← 稳定 ~10%
```
句子级比词级好 (13% vs 10%)，但天花板仍低。

#### Two-stage (unified 5 epoch, then split_grad)
```
epoch:    0     5     10    15    20    30    40    49
top-5:    7.7%  12.3% 20.7% 12.7% 12.3% 11.0% 14.3% 10.7%
```
**Peak 20.7%@ep10** — 训练过程中历史最高！但 ep10 后退化。

#### Split_grad + enhancer 高 lr (2e-4)
```
epoch:    0     5     10    20    49
top-5:    16.3% 10.7% 7.3%  8.7%  10.0%
```
**Epoch 0 = 16.3%** — enhancer 能快速学语义！但 backbone JEPA 梯度腐蚀 → 退化。

#### Freeze backbone (freeze5 / freeze10)
```
freeze5:  8.3%@5 → 6.0%@10 → 7.0%@99    ← 冻结后更差
freeze10: 19.7%@10 → 9.0%@15 → 10.0%@99 ← peak 后直接崩
```
**冻结 backbone 导致 enhancer 过拟合** — backbone feature 演化是隐式数据增强。

#### Pure CLIP + CTM (w_mse=0)
```
epoch:    0     5     10    15    20    30    50    70    99
top-5:    6.7%  6.3%  9.3%  13.7% 9.7%  9.7%  10.7% 13.0% 13.7%
```
**100 epoch 完全不退化**，缓慢上升。纯 CLIP+CTM 天花板 ~14%。

#### JEPA 权重退火 (decay 10→20) — **最优方案**
```
epoch:    0     5     10    15    20    25    30    50    70    99
top-5:    7.7%  8.3%  19.7% 17.0% 13.3% 16.0% 13.3% 12.3% 15.7% 13.0%
w_mse:    0.3   0.3   0.3   ~0.15 0.0   0.0   0.0   0.0   0.0   0.0
```
**Peak 19.7%@ep10 + 稳定 13-16%**。JEPA boost + CLIP stability 的最佳结合。

#### Cyclic JEPA (cycle=30, 3+ 轮)
```
Cycle 0 (ep0-29):  peak=19.7%@10  floor=13.3%
Cycle 1 (ep30-59): peak=17.3%@45  floor=11.3%  ← 更低
Cycle 2 (ep60-89): peak=17.0%@60  floor=10.0%  ← 更低
```
每轮 peak 递减 — 循环不积累，反而引入噪声。

## 4. 梯度冲突分析

### 4.1 问题定义

多目标 loss: `L = w_mse * L_jepa + w_clip * L_clip + lam * L_sigreg`

JEPA 的梯度方向 (学习 world dynamics) 与 CLIP 的梯度方向 (学习 semantic alignment) 可能冲突。

### 4.2 Split gradient path

```python
# split_grad=True: JEPA 用 raw backbone features (不过 enhancer)
ctx_raw = backbone(x)           # → JEPA (梯度只到 backbone)
ctx_enh = enhancer(ctx_raw)     # → CLIP  (梯度到 backbone + enhancer)

# split_grad=False (unified): JEPA 也用 enhanced features
ctx_enh = enhancer(backbone(x)) # → JEPA + CLIP (梯度都到 enhancer)
```

**验证**: split_grad=True 时 enhancer JEPA 梯度为 None (正确阻断)；split_grad=False 时 enhancer JEPA 梯度为 0.0008 (正确流通)。

### 4.3 退化根因

| 假设 | 实验 | 验证结果 |
|---|---|---|
| Enhancer 被 JEPA 腐蚀 | split_grad (阻断) | ❌ 不退化但天花板低 (8%) |
| Backbone 被 JEPA 腐蚀 | freeze backbone | ❌ 冻结后更差 (enhancer 过拟合) |
| Enhancer 学太慢 | enhancer lr=2e-4 | ❌ epoch 0=16% 但仍退化 |
| 纯过拟合 (数据太少) | pure CLIP (w_mse=0) | ✅ 不退化！ |
| JEPA 本身导致退化 | JEPA decay 0.3→0 | ✅ 退火后稳定！ |

**根因**: JEPA 梯度提供了初始表示 boost (到 ~20%)，但在 backbone 收敛后 (ep10+) 继续施加 JEPA 导致 backbone features 漂移，enhancer 追不上 → 退化。

### 4.4 为什么冻结 backbone 不 work

冻结 backbone 后 enhancer 在固定 features 上 4 epoch 就过拟合 (19.7%→9.0%)。**Backbone 的持续训练 (feature 演化) 是隐式数据增强**，防止 enhancer 记忆训练集。

### 4.5 为什么循环 JEPA 不 work

每轮循环的 peak 递减 (19.7→17.3→17.0)。原因:
1. Backbone 已收敛，JEPA boost 效果减弱
2. EMA tau→1.0，JEPA target 几乎不变，loss 变平凡
3. Enhancer 已学到接近最优映射，重新 boost 收益递减

## 5. 最终训练方案: JEPA 权重退火

### 方案
```python
# Phase 1 (ep0-9): JEPA boost
w_mse = 0.3  # JEPA 提供表示学习信号 → peak ~20%

# Phase 2 (ep10-19): 线性退火
w_mse = linear_decay(0.3 → 0.0)  # 平滑过渡

# Phase 3 (ep20+): Pure CLIP stability
w_mse = 0.0  # 纯 CLIP+CTM → 稳定 ~14%
```

### 命令
```bash
torchrun --nproc_per_node=8 train_ctm_enc_jepa.py \
  --align_mode sentence \
  --unified_epochs 5 \
  --w_mse 0.3 --w_mse_decay_start 10 --w_mse_decay_end 20 --w_mse_min 0.0 \
  --w_clip 1.0 --lam 0.1 \
  --lr_enhancer 2e-5 --lr_encoder 2e-5 --lr 2e-4 \
  --loss_type siglip \
  --epochs 100
```

### 效果
- Peak: 19.7% (ep10，JEPA boost)
- Floor: 13-16% (ep20-99，pure CLIP 稳定)
- 不退化，100 epoch 稳定运行
- 比纯 CLIP+CTM (14%) 在 peak 期间高 ~6%
- 比无退火 two-stage (退化到 11%) floor 高 ~3%

## 6. 完整调参穷尽（20 组实验）

### 6.1 最优配置

```bash
--unified_epochs 5          # 5 epoch unified (enhancer 收 JEPA)
--w_mse 0.3                 # JEPA 权重
--w_mse_decay_start 10      # ep10 开始退火
--w_mse_decay_end 20        # ep20 退火到 0
--w_mse_min 0.0             # 完全关闭 JEPA
--lr_enhancer 2e-5          # enhancer 学习率
--lr_encoder 2e-5           # backbone 学习率
--lr 2e-4                   # predictor/其他
--align_mode sentence       # 句子级对齐
--loss_type siglip          # SigLIP loss
--lam 0.1                   # SIGReg 权重
```

### 6.2 调参穷尽结果

| 变量 | 测试值 | 最优值 | 理由 |
|---|---|---|---|
| w_mse (JEPA 权重) | 0.0, **0.3**, 0.5 | **0.3** | 0.5 太强(18%)，0.0 无 boost(14%) |
| w_mse_min (残留 JEPA) | **0.0**, 0.05, 0.1 | **0.0** | 0.05 最差(12%)，0.1 有 spike(28%) 但崩 |
| decay 速度 | **10ep**, 30ep | **10ep** | 30ep floor 更低(12% vs 14%) |
| unified_epochs | **5**, 10 | **5** | 10 延迟 peak 且 floor 更低 |
| enhancer lr | **2e-5**, 1e-4, 2e-4 | **2e-5** | 高 lr 不稳定 |
| cyclic JEPA | cycle=30 | 不用 | 每轮 peak 递减(19.7→17.3→17.0) |
| freeze backbone | ep5, ep10 | 不用 | enhancer 在固定 features 上过拟合 |
| 数据增强 | flip+jitter | 边缘改善 | floor 略好(15% vs 14%)，peak 不变 |

### 6.3 关键发现：28% spike (decay_min01 ep20)

w_mse_min=0.1 实验中，ep20（w_mse 刚降到 0.1）出现 28.0% top-5——历史最高。但 5 epoch 后崩到 12.7%。不可复现于 w_mse_min=0.0 或 0.05。这是 **transition sweet spot**：模型在 JEPA 权重变化的过渡期表现最好。

## 7. 关键经验

1. **JEPA 是双刃剑**:前期提供表示 boost (13.7%→20%)，后期导致退化 (20%→11%)
2. **退火是优雅的解决方案**:利用 JEPA 的 boost，避免其退化
3. **冻结不是答案**:backbone feature 演化是必要的隐式增强
4. **循环不积累**:多轮 boost 效果递减，单次退火更优
5. **Enhancer lr=2e-5 最优**:更高 lr (1e-4, 2e-4) 导致不稳定
6. **句子级 > 词级**:句子级 split_grad 天花板更高 (13% vs 10%)
7. **模型在过渡期表现最好**:peak 都发生在 JEPA 权重/split_grad 模式变化的过渡区
8. **天花板 ~20% 不可突破**:15+ 实验调参全部收敛到同一天花板

## 8. 根因定位（详见实验 7：逐层探测）

逐层探测 (probe_layers.py) 发现：
- **Backbone（所有 4 个 stage）几乎不携带语义信息**（alignment gap = 0.001-0.017）
- **所有语义能力来自 CTM enhancer 的 attention 机制**（tick 1 即从 gap=0.017 跃升到 0.53）
- **瓶颈在 backbone 的 16×16 patch 分辨率**：字符级形状信息在 patchify 时丢失

## 9. 待探索方向（结构改进）

1. **更高分辨率文字处理**:text 区域用 8×8 或 4×4 patch，保留字符形状
2. **文字形状辅助损失**:OCR-like 监督引导 backbone 学习文字结构
3. **多尺度 backbone**:fine resolution 处理文字 + coarse resolution 处理图片
4. **Latent split**:feature dim 分两半，CLIP 用前半，JEPA 用后半
5. **更大数据集**:356k 对自监督训练偏少 (CLIP 用 400M)

## 10. 代码文件

- `train_ctm_enc_jepa.py`:CTM encoder JEPA (主实验，含 split_grad/unified/two-stage/freeze/decay/cyclic/augment)
- `probe_zeroshot.py`:训练分布 zero-shot 词-图检索 (支持 CTMEncoderJepa)
- `probe_collapse.py`:特征塌缩诊断
- `probe_layers.py`:逐层探测 (详见实验 7)
