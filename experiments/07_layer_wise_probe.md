# 实验 7: 逐层探测——语义理解在网络中的分布

- 日期:2026-07-28
- 结论:**Backbone（所有 4 个 stage）几乎不携带语义信息（alignment gap=0.001-0.017）。所有语义能力来自 CTM enhancer 的 attention 机制——第一个 tick 就将 gap 从 0.017 跃升到 0.53（31 倍）。瓶颈在 backbone 的 16×16 patch 分辨率：字符级形状信息在 patchify 时丢失，ConvNeXt 只学到了纹理/边缘特征，没有学到文字形状。**
- 状态:根因定位完成，明确了结构改进方向

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

## 7. 结构改进方向

| 方向 | 思路 | 预期效果 |
|---|---|---|
| **Fine patch for text** | text 区域用 8×8 或 4×4 patch | 保留字符形状 → backbone gap 提升 |
| **多尺度 backbone** | fine (text) + coarse (photo) 并行 | text 保持分辨率，photo 保持感受野 |
| **文字形状辅助损失** | OCR-like 监督 | 直接引导 backbone 学习字符结构 |
| **更高分辨率输入** | 448×448 (grid=28) | 更多 patch 覆盖文字区域 |
| **专门 text stem** | CNN stem 处理文字行 | 提取笔画/字符特征 |

## 8. 代码文件

- `probe_layers.py`:逐层探测脚本（backbone stages + enhancer ticks）
- 使用的 checkpoint:`outputs/jepa_decay10/epoch10.pt`（peak 模型）
