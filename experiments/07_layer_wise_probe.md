# 实验 7: 逐层探测 + 冻结早期层——语义理解在网络中的分布与瓶颈定位

- 日期:2026-07-28 至 2026-07-29
- 结论:**Backbone（所有 4 个 stage）几乎不携带语义信息（alignment gap=0.001-0.017）。所有语义能力来自 CTM enhancer 的 attention 机制——第一个 tick 就将 gap 从 0.017 跃升到 0.53（31 倍）。冻结早期层（stem+stage0+stage1）反而更差（peak 9%→5.7%），因为早期层尚未学到任何形状信息就被锁定。瓶颈在 backbone 的 16×16 patch 分辨率：字符级形状信息在 patchify 时丢失。所有训练策略已穷尽，天花板 ~20% 是架构限制。**
- 状态:根因定位完成，训练策略穷尽，下一步是结构改进

## 1. 动机

实验 6 穷尽了所有 JEPA 调参方向，天花板固定在 peak ~20% / floor ~14%。需要理解**为什么无法突破**——是哪一层的表示能力不足？

核心问题：**"从视觉中捕获文字形状的能力"是集中在网络的前面、中间还是后面？**

## 2. 方法 (probe_layers.py)

### 2.1 探测点

在训练好的 jepa_decay10 (epoch 10, peak checkpoint) 上，提取以下层的特征：

**Backbone（ConvNeXt 4 个 stage）：**
```
stage0: 56×56, 192d  (2 个 ConvNeXtBlock，浅层边缘/纹理)
stage1: 28×28, 384d  (2 个 block，中层形状)
stage2: 14×14, 768d  (6 个 block，深层语义 ← 模型使用的输出)
  └ s2_blk0 ~ s2_blk5: stage2 内部 6 个 block 逐个探测
stage3: 7×7, 1536d   (2 个 block，最终层 ← 模型未使用)
```

**CTM Enhancer（50 个 tick）：**
```
tick 1, 3, 5, 10, 20, 35, 50
每个 tick 后 broadcast thoughts → cells，得到 enhanced feature map
```

### 2.2 评测指标

对每层的 feature map，pool text 区（上半）和 photo 区（下半）：

- **matched cosine**:同一图像的 text feat 和 photo feat 的余弦相似度
- **unmatched cosine**:不同图像的 text feat 和 photo feat 的余弦相似度
- **gap** = matched - unmatched（越大说明语义对齐越好）
- **top-5**:从 text feat 检索正确 photo 的 top-5 命中率

### 2.3 数据

300 个训练分布样本（caption + photo 合成图），与 probe_zeroshot 相同的采样方式。

## 3. 结果

### 3.1 完整表格

```
layer             matched  unmatch      gap    top-5
----------------------------------------------------
stage0             0.4789   0.4782   0.0007    0.027   ← 纯随机
stage1             0.2304   0.2168   0.0137    0.027   ← 微弱信号
stage2             0.5231   0.5057   0.0173    0.027   ← backbone 最佳
stage3             0.2917   0.2844   0.0073    0.030   ← 比stage2更差
s2_blk0            0.5372   0.5204   0.0168    0.030   ┐
s2_blk1            0.5373   0.5205   0.0168    0.030   │ 6 个 block
s2_blk2            0.5374   0.5207   0.0167    0.030   │ 几乎无变化
s2_blk3            0.5374   0.5207   0.0167    0.030   │ (残差停滞)
s2_blk4            0.5377   0.5210   0.0167    0.030   │
s2_blk5            0.5377   0.5211   0.0167    0.027   ┘
enh_t1             0.9393   0.4071   0.5322    0.990   ← 31 倍跃升！
enh_t3             0.9436   0.4410   0.5026    0.993
enh_t5             0.9419   0.3704   0.5715    0.997
enh_t10            0.9470   0.4164   0.5306    0.997
enh_t20            0.9453   0.2515   0.6938    1.000
enh_t35            0.9467   0.1581   0.7886    1.000
enh_t50            0.9468   0.0129   0.9339    1.000   ← 近乎完美

backbone best: stage2 gap=0.0173 top5=2.7%
enhancer best: enh_t50 gap=0.9339 top5=100%
enhancer adds: +0.9166 gap (+5289.9%)
```

### 3.2 可视化趋势

```
gap
0.94 │                                          ●────● enhancer
    │                                     ●────●
0.69 │                              ●────●
    │                         ●────●
0.53 │                   ●────●────●────●
    │             ●────●
0.02 │  ●────●────●────●                         backbone (所有层)
    │──●─────────────────────────────────────────────────
0.00 │ stem  s0   s1   s2   s3                  t1   t50
    └─────────── backbone ────────────────── enhancer ───
```

## 4. 核心发现

### 4.1 Backbone 是纯纹理检测器

所有 4 个 stage 的 gap 都在 **0.001-0.017** 之间——top-5 仅 2.7%（接近随机）。这意味着 backbone 提取的特征中，同一图像的 text 区和 photo 区**并不比不同图像更相似**。

ConvNeXt 的 16×16 patch 意味着每个 patch 覆盖 256 像素。一个字母可能只占半个 patch，一个词占 2-3 个 patch。字符级的形状信息（曲线、交叉点、闭合区域）在 patchify 时就被抹平了。backbone 看到的只是"这块区域有黑色像素"（文字纹理），不是"这是字母 A"。

### 4.2 Stage2 的 6 个 block 几乎无变化

```
s2_blk0: gap=0.0168
s2_blk5: gap=0.0167  (几乎相同)
```

ConvNeXtBlock 使用 `gamma` 初始化为 1e-6 的残差连接。在 backbone lr=2e-5 下，这些 block 的残差贡献极小——**stage2 的 6 个 block 实际上是"冻结"的**，只传递梯度但不改变特征。

### 4.3 所有语义能力来自 CTM enhancer

第一个 enhancer tick 就将 gap 从 **0.017 跃升到 0.53**（31 倍）。这不是渐进学习——是 **attention 机制瞬间把 text 和 photo 的信息通过 thought tokens 混合在一起**。

thought tokens 同时 attend 到 text cells 和 photo cells，获取了跨模态信息。broadcast 回 cells 时，text cells 和 photo cells 都获得了来自对方的信息，因此同一图像的 text/photo 特征变得高度相似。

### 4.4 更多 tick = 更锐利的区分（不是更好的对齐）

```
tick    matched    unmatched     解读
  1      0.939      0.407       attention 瞬间混合 → matched 已经很高
 20      0.945      0.252       SIGReg + CLIP → unmatched 开始下降
 50      0.947      0.013       近乎正交 → 完美区分
```

matched cosine 从 tick 1 就达到 0.94（attention 的即时效果），后续 tick 主要在**压低 unmatched cosine**——让不同图像的特征变得正交。这是 SIGReg 防塌缩 + CLIP 对比损失的效果。

### 4.5 Top-5=100% 是句子级对齐的假象

enhancer 的 top-5=100% 是因为同一张 composite 的 text 和 photo 天然相关（共享 backbone 通路 + attention 混合）。这只是**空间关联**（text 区和 photo 区来自同一张图），不是**语义理解**（识别具体词义）。

真正的词级检索（probe_zeroshot，需要从 photo 检索具体单词）只有 **19.7%**。

## 5. 根因总结

```
问题：为什么天花板卡在 ~20%？
答案：backbone 无法从 16×16 patch 中提取文字形状。

证据链：
1. Backbone gap=0.017（所有 stage）→ 零语义结构
2. Enhancer gap=0.53（tick 1）→ attention 做空间关联，不做语义
3. 词级 top-5=19.7% → 有一定语义，但远低于 CLIP 55.7%
4. CLIP 用 tokenizer 直接编码词结构 → 不需要从像素提取形状

瓶颈层级：
backbone (patchify 丢失字符形状) > enhancer (attention 只做空间关联) > JEPA/CLIP loss (无法弥补特征不足)
```

## 6. 对实验 6 的解释

逐层探测结果解释了实验 6 的所有现象：

| 现象 | 解释 |
|---|---|
| 天花板固定在 ~20% | Backbone 提供不了足够的文字形状信息 |
| JEPA 调参全部无效 | 问题在特征提取层，不在损失函数 |
| 数据增强无突破 | 增强 photo 不帮助 text 形状提取 |
| 冻结 backbone 更差 | 连微弱的纹理特征也失去了演化 |
| CLIP (tokenizer) 55.7% | Tokenizer 直接保留词级结构 |
| Enhancer 50 tick vs 1 tick 差别小 | Attention 在 tick 1 就完成了空间关联 |

## 7. 冻结早期层实验 (--freeze_early_after)

### 7.1 动机

逐层探测显示 backbone 早期层（stem+stage0+stage1）是纹理/边缘检测器，收敛快。假设：冻结早期层后，stage2+enhancer 在稳定基础上学语义 → 更高的 peak 或 floor。

### 7.2 配置

```bash
--unified_epochs 5          # 与最优配置相同
--w_mse 0.3 --w_mse_decay_start 10 --w_mse_decay_end 20 --w_mse_min 0.0
--freeze_early_after 5       # ep6 冻结 stem+stage0+stage1
```

冻结的模块：`down_layers[0]`, `stages[0]`, `down_layers[1]`, `stages[1]`, `down_layers[2]`
保留可训练：`stages[2]`, `enhancer`, `predictor`, `pos`, `mask_token`

### 7.3 结果

```
epoch:     0     5     10    15    20    30    50    99
──────────────────────────────────────────────────────────
freeze_early5  8.0%  9.0%  5.7%  5.7%  5.7%  7.3%  9.7%  10.0%
jepa_decay10   7.7%  8.3%  19.7% 17.0% 13.3% 13.3% 12.3% 13.0%
```

**冻结后直接崩盘**：ep5=9.0% → ep10=5.7%（冻结后 4 epoch 内暴跌 37%）。之后 95 epoch 缓慢恢复到 10%，远低于不冻结的 13-14%。

### 7.4 失败原因

1. **ep5 时早期层尚未学到任何形状信息**（逐层探测证实 gap=0.017）
2. 冻结把这个"零语义"状态**永久锁定**
3. stage2 和 enhancer 被迫在**永久无语义的基础特征**上工作
4. 这比冻融整个 backbone 更差，因为至少 backbone 整体冻结时 stage2 也不变（稳定的恒等映射），而这里 stage2 还在试图适应一个无意义的冻结基础

### 7.5 与冻融整个 backbone 的对比

| 策略 | 冻结范围 | Peak | ep10 | ep99 |
|---|---|---|---|---|
| 不冻结 (jepa_decay10) | 无 | 19.7% | 19.7% | 13.0% |
| 冻融整个 backbone (ep10) | 全部 backbone | 19.7% | 19.7% | 10.0% |
| 冻融早期层 (ep5) | stem+s0+s1 | 9.0% | 5.7% | 10.0% |

冻融早期层比冻融整个 backbone 更差——因为 backbone 的全部层共同演化形成了一致的特征空间（即使 gap 小），冻结部分层破坏了这种一致性。

## 8. 训练策略彻底穷尽总结

### 8.1 所有策略一览

| 策略 | 变量 | 最优值 | Peak | Floor | 退化? |
|---|---|---|---|---|---|
| JEPA 权重 | w_mse | 0.3 | 19.7% | ~14% | 否 (decay) |
| JEPA 退火 | decay_start/end | 10→20 | 19.7% | ~14% | **否** |
| 残留 JEPA | w_mse_min | 0.0 | 19.7% | ~14% | 否 |
| JEPA 循环 | cycle | 不用 | 19.7% | ~13% | 否 |
| 统一阶段 | unified_epochs | 5 | 19.7% | ~14% | 否 |
| 分离梯度 | split_grad | two-stage | 19.7% | ~14% | 否 |
| Enhancer lr | lr_enhancer | 2e-5 | 19.7% | ~14% | 否 |
| 数据增强 | augment | 边缘改善 | 19.3% | ~15% | 否 |
| 冻融整个 backbone | freeze_backbone | 不用 | 19.7% | ~10% | 是 |
| 冻融早期层 | freeze_early | 不用 | 9.0% | ~10% | 是 |

### 8.2 结论

**所有训练策略都触碰同一个天花板 peak ~20% / floor ~14%。** 无论怎么调整 loss 权重、学习率、冻结策略、数据增强、循环调度，都无法突破。

天花板来自架构限制（16×16 patch 丢失字符形状），不是训练策略问题。需要**结构改进**。

## 9. 结构改进方向

| 方向 | 思路 | 改动量 | 预期效果 |
|---|---|---|---|
| **高分辨率输入** | 448×448 (grid=28) | 小 (改 img_size) | patch 数 196→784，文字区 patch ×4 |
| **Fine patch text** | text 区 8×8 patch | 中 (双分支 stem) | 字符形状保留 |
| **多尺度 backbone** | fine text + coarse photo | 大 (新架构) | 兼顾分辨率和感受野 |
| **文字形状辅助损失** | OCR-like 监督 | 中 (新 loss) | 引导 backbone 学字符 |
| **专门 text stem** | CNN 处理文字行 | 中 (新模块) | 提取笔画特征 |

## 10. 代码文件

- `probe_layers.py`:逐层探测脚本（backbone stages + enhancer ticks）
- `train_ctm_enc_jepa.py`:主训练脚本（含 freeze_early_after）
- 使用的 checkpoint:`outputs/jepa_decay10/epoch10.pt`（peak 模型）
