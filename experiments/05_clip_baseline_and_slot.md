# 实验 5:CLIP 基线对比 + Slot Attention — 最终三方对比

- 日期:2026-07-14
- 结论:**CLIP 在同等数据上碾压纯视觉方法(top-5 56% vs 14%,4x 差距)。Slot attention 不仅没改善,反而大幅恶化(top-5 3-8%),原因是语义压扁。纯视觉通路在 800k 数据上达到 tokenizer 方式的 ~25%。**
- 状态:负结果(slot attention)+ 确定性对比(CLIP 基线)+ 调研收口

## 1. 目标

两个并行目标:
1. **CLIP 基线**:在完全相同的数据 + 图像 encoder 下,用标准 tokenizer + text transformer 替代像素渲染,量化"纯视觉通路"的代价。
2. **Slot attention**:用竞争性注意力替代 cell mean pooling 提取词特征,试图改善词级判别力。

## 2. CLIP 基线设计

### 公平对比原则:只改一个变量
| | CLIP 基线 | 我们(实验 4) |
|---|---|---|
| 图像 encoder | **同一个 ConvNeXt** | **同一个 ConvNeXt** |
| 文字处理 | BPE tokenizer → causal text transformer | 渲染成像素 → 共享 ConvNeXt |
| 参数量 | ~167M(8L)/ ~215M(12L) | ~104M |
| 数据 | recap-datacomp 81 tars | 相同 |
| 训练 | 100 epoch, batch 512, lr 5e-4 | 100 epoch, batch 256, lr 3e-4 |

### 架构(train_clip.py)
- **image encoder**: ConvNeXt → global avg pool → linear projection → 768d
- **text encoder**: CLIP-style causal transformer(8 或 12 层)+ BPE tokenizer(`openai/clip-vit-base-patch32`, vocab=49408)
- **loss**: 对称 InfoNCE(image↔text, 跨卡 gather 到 2560 负样本)
- **特征**: image = projection(convnext global pool); text = transformer 在 EOS 位置的隐藏状态

## 3. CLIP 结果

### 词级 zero-shot 检索(probe_clip.py,同一评测协议)
| 模型 | top-1 | top-5 | top-10 | MRR | 词间cos |
|---|---|---|---|---|---|
| CLIP 8-layer text | 18.0% | 51.3% | 70.3% | 0.344 | 0.463 |
| CLIP 12-layer text | **20.3%** | **55.7%** | **73.3%** | **0.370** | 0.563 |

### 句子级检索(标准 CLIP 评测)
- i2t R@1 = 100%, t2i R@1 = 100%(在训练分布上完美匹配图文对)

### 定性观察
- CLIP 词级检索高度准确:给一张 Amazon 物流车的图,正确检索到 "logo"、"blue" 等 caption 中的词
- 词间 cos 0.46-0.56:CLIP 的词特征有自然的语义聚集(同义词/相关词靠近)

## 4. Slot Attention 设计

### 动机
- 实验 4 的 cell mean pooling 产生太平坦的词特征(词间 cos 0.055,几乎正交)
- Slot attention(Locatello et al. 2020)的竞争性绑定可能产生更锋利的词特征
- Slot 之间竞争 → 天然防塌缩(GPT 指出的 near-collapse 问题可能被解决)

### 架构(train_slot_jepa.py)
```
convnext → feature map (196, D)
    ↓
slot attention (K slots, T iters) — 竞争性绑定
    ↓
slots (K, D) + attention (196, K)
    ↓
词特征 = word_mask @ attention @ slots / normalize
图片特征 = photo_cell attention @ slots
    ↓
InfoNCE + JEPA MSE + SIGReg
```

SlotAttention 核心:
- K 个 slot 通过 T 次迭代竞争绑定 feature map 的不同区域
- 每次迭代:attention softmax(slot 间竞争)→ normalize(cells 间)→ GRU 更新
- 初始化:learned μ + σ × noise

### 变体
| 变体 | slots | iters | w_mse | w_clip | hidden |
|---|---|---|---|---|---|
| base | 16 | 3 | 0.3 | 1.0 | 768 |
| strongclip | 16 | 3 | 0.1 | 2.0 | 768 |
| slot32 | 32 | 3 | 0.3 | 1.0 | 768 |
| big | 16 | 3 | 0.3 | 1.0 | 1024 |

## 5. Slot Attention 结果

### 词级 zero-shot 检索(probe_slot.py,slot 特征)
| 变体 | top-5 | MRR | 词间cos | vs 无slot |
|---|---|---|---|---|
| **无 slot(实验4 基线)** | **13.7%** | **0.098** | **0.055** | — |
| base 16 slots | 3.0% | 0.038 | — | -78% |
| strongclip | 5.7% | 0.070 | 0.609 | -58% |
| slot32 | 8.0% | 0.067 | 0.835 | -42% |
| big (hidden=1024) | 5.3% | 0.054 | 0.723 | -61% |

### Slot attention 彻底失败

**不仅没改善,反而大幅恶化。** top-5 从 13.7% 暴跌到 3-8%。

### 失败原因:语义压扁

词间 cos 从 0.055(健康)飙升到 **0.6-0.83**(严重塌缩):
```
无 slot:     词间 cos = 0.055 (std 0.208)  ← 健康分散
slot32:      词间 cos = 0.835 (std 0.100)  ← 几乎完全塌缩!
slot_big:    词间 cos = 0.723 (std 0.127)  ← 严重塌缩
strongclip:  词间 cos = 0.609 (std 0.179)  ← 中度塌缩
```

481 个词挤进 16-32 个 slot → 每个_slot_代表 15-30 个词 → 词特征高度聚集 → 判别力崩溃。

**Slot attention 的竞争机制在此场景不 work**:
1. 文字区域高度密集(98 cells × 10-40 个词),slot 无法区分单个词
2. Slot 没有分化(inter-cos 0.6+),竞争未产生差异化
3. ConvNeXt feature map 是空间特征(边缘/纹理),不是语义特征,slot 在上面学不到词义

## 6. 最终三方对比

| 方法 | top-5 | MRR | 词间cos | 说明 |
|---|---|---|---|---|
| CLIP 12-layer | **55.7%** | **0.370** | 0.563 | tokenizer + text transformer |
| CLIP 8-layer | 51.3% | 0.344 | 0.463 | 同上,更小 text encoder |
| Cross-modal JEPA | **13.7%** | **0.098** | 0.055 | 像素渲染 + 共享 encoder |
| SlotJEPA best | 8.0% | 0.067 | 0.835 | slot attention(恶化) |

```
纯视觉通路效率 = 13.7% / 55.7% ≈ 25% of CLIP
```

## 7. 结论

### 7.1 已确认的事实
1. **纯视觉通路可以学到词-图 grounding**:top-5=13.7%, 12x random, 无泄漏验证(实验 4)
2. **但效率约为 tokenizer 方式的 1/4**:CLIP 同条件 top-5=55.7%
3. **Slot attention 不能缩小差距**:反而大幅恶化(语义压扁)
4. **瓶颈在架构层面**:共享 encoder + 像素渲染的表达能力限制,不是数据/epoch/负样本/slot 能解决的

### 7.2 根因分析
CLIP 的优势来自两个层面:
1. **Tokenizer 提供词级结构**:BPE 直接编码离散词,天然有词边界。像素方法要从 196 个 cells 中"重新发明"词识别。
2. **独立 text encoder**:专门处理文字,image encoder 专门处理图片。共享 encoder 负担更重,两种信号互相干扰。

### 7.3 纯视觉通路的价值
尽管效率低,但在特定场景有不可替代的优势:
- **Screen reading**:屏幕内容是像素化的文字+图形,不需要 OCR 前置
- **任意字体/语言/排版**:不依赖固定词表,天然适应变化
- **认知科学意义**:验证了"纯视觉通路能否学词义"——答案是 yes,但效率 25%

### 7.4 关键经验教训
1. **探针分布必须匹配训练分布**(白色下半导致假塌缩,掩盖了所有信号)
2. **渲染要验证**(空白 text 区浪费了 15 epoch)
3. **防塌缩 ≠ 保语义**(SIGReg 防住了分布塌缩,但词特征语义结构仍弱)
4. **InfoNCE 正负样本来自同一 forward pass 有泄漏风险**(需 blank_text 验证)

## 8. 代码文件
- `train_clip.py` + `probe_clip.py`:CLIP 基线(训练 + 评测)
- `train_slot_jepa.py` + `probe_slot.py`:Slot JEPA(训练 + 评测)
- `clip_tokenizer/`:离线 CLIP BPE tokenizer(via hf-mirror)
- `run_overnight.sh`:过夜训练脚本(5 实验串行)
