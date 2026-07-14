# Screen-JEPA: 纯视觉通路能否从像素学到词义？

> 一项关于"文字渲染成像素后,能否通过联合嵌入预测(JEPA)学到词-图语义对齐"的调研。

## 核心问题

人类阅读时,视觉皮层(V1)同时处理文字和图片——没有独立的"文字通道"。CLIP 等现代视觉-语言模型依赖 tokenizer(BPE)将文字编码为离散 token,绕过了"从像素认字"这一步。

**能否不做 tokenization,把文字渲染成像素,用纯视觉通路(单一 encoder)同时学到文字识别和词-图语义对齐?**

## 一句话结论

**可以,但效率约为 tokenizer 方式的 1/4。** 在 800k 图文对上,纯视觉方法达到 zero-shot 词-图检索 top-5=13.7%(12x random),CLIP 基线(同数据、同图像 encoder)达到 55.7%。瓶颈在架构层面:共享 encoder 的双重负担 + 像素认字的额外开销,非数据量或训练时长可解决。

---

## 实验全景

### 方法演进(5 个阶段)

| 实验 | 方法 | 目标函数 | 词级结果 | 状态 |
|---|---|---|---|---|
| 01 | 全局不变性 JEPA | full vs masked 文本对齐 | syn-rand ≈ 0(噪声) | 负:无语义信号 |
| 02 | +强增强 | 不变性 + 几何/字体增强 | retrieval 0.43 | 负:认字但不理解 |
| 03 | 两阶段预测式 | 冻结 encoder → predictor | predictor 随机水平 | 负:像素无语义信号 |
| **04** | **跨模态 JEPA + 词级 CLIP** | **图文合成 + InfoNCE** | **top-5=13.7%** | **正:首个真实 grounding** |
| 05 | CLIP 基线 + Slot Attention | 对照 + 改进尝试 | CLIP 55.7%, Slot 失败 | 确定性对比 |

### 最终三方对比(zero-shot 词-图检索,481 词表,300 测试图)

| 方法 | top-1 | top-5 | top-10 | MRR | 词间 cos |
|---|---|---|---|---|---|
| **CLIP 12-layer** | **20.3%** | **55.7%** | **73.3%** | **0.370** | 0.563 |
| CLIP 8-layer | 18.0% | 51.3% | 70.3% | 0.344 | 0.463 |
| **Cross-modal JEPA (我们)** | **2.3%** | **13.7%** | **24.7%** | **0.098** | 0.055 |
| SlotJEPA best (32 slots) | 1.0% | 8.0% | 15.7% | 0.067 | 0.835(塌缩) |

> random baseline: top-5=1.0%, MRR=0.002。纯视觉方法 12x random,CLIP 53x random。

---

## 关键发现

### 1. 词级 CLIP 是唯一有效的 grounding 目标

实验 01-03 尝试了三种纯像素目标(不变性、预测、MLM),都学不到语义。唯一产生真实 grounding 信号的是**词级 CLIP 对齐**:把渲染的 caption 和真实 photo 拼在一张合成图里,让每个词的 cell 特征与图片特征做 InfoNCE。

### 2. 两个 bug 掩盖了信号(耗费大量排查时间)

- **渲染 bug**:`render_caption_block` 初始坐标 `y=hh, x=ww`(右下角外),任何字号都 fit 失败 → text 区一直是**空白**。前 15 epoch 完全无效。
- **探针 OOD**:所有探针用"单词渲染 + 白色下半"测试。模型训练时下半总是 photo,从没见过纯白下半 → **探针特征自身塌缩到 cos 0.97+**,产生假象(所有词看起来都一样)。

修复方式:
- 渲染:`y=0, x=6`
- 探针:从**训练分布**(真实合成图)提取词特征,而非 OOD 输入

### 3. 无泄漏验证

测试合成图的上半如果保留 caption,photo 特征会受 text 影响(泄漏)。对比实验:
```
上半=caption(有泄漏):  top-5 = 14.0%
上半=空白(无泄漏):      top-5 = 12.7%   ← 只降 10%
```
确认 grounding 是真实的(不是靠"读 caption 作弊")。

### 4. 方法饱和与过拟合

```
epoch:    20     100     250
top-5:    12.7%  13.7%   10.7%   ← 100 epoch 最优,250 过拟合
```

100 epoch 是最优点。更多 epoch 导致 InfoNCE 过拟合(记住 batch 配对,泛化退化)。数据翻倍(40→81 tars)、负样本 5x(跨卡 gather)均无改善。

### 5. Slot attention 失败

Slot attention(竞争性注意力绑定)替代 cell mean pooling 提取词特征,反而大幅恶化(top-5 从 13.7% 跌到 3-8%)。原因:**语义压扁**——481 个词挤进 16-32 个 slot,词特征高度聚集(词间 cos 0.6-0.83),判别力崩溃。

### 6. CLIP 的三层效率优势

纯视觉通路(13.7%)与 CLIP(55.7%)的 4x 差距来自三层叠加:

| 层面 | CLIP | 纯视觉 | 代价 |
|---|---|---|---|
| 文字识别 | tokenizer 免费 | convnext 从像素学 | ~2x |
| encoder 分工 | 两个独立 encoder | 共享一个 | ~1.5x |
| 词特征质量 | 12 层 transformer EOS | cell mean pooling(3-4 cells) | ~1.3x |

---

## 瓶颈分析

### 不是数据量的问题
CLIP 在同样的 800k 数据上达到 55.7%。差距是架构性的,非数据可弥补。

### 不是训练时长的问题
100→250 epoch 过拟合,top-5 从 13.7% 降到 10.7%。

### 是架构问题
1. **共享 encoder 的双重负担**:一个 convnext 同时做 OCR + 图片理解 + 跨模态对齐。文字(锐利、人工)和图片(柔和、自然)两种信号在同一个 feature map 里互相干扰。
2. **词特征提取效率低**:每个词只覆盖 3-4 个 grid cells,mean pooling 压成一个向量,信息密度远低于 CLIP 的 12 层 transformer。
3. **像素到语义的路径过长**:CLIP 从 token(已经是语义单元)开始;我们从像素(笔画)开始,中间需要"重新发明"文字识别。

---

## 代码结构

### 训练脚本
| 文件 | 说明 |
|---|---|
| `train_crossmodal_jepa.py` | **核心**:跨模态 JEPA + 词级 CLIP(ConvNeXt + InfoNCE + SIGReg) |
| `train_clip.py` | CLIP 基线(同 ConvNeXt + text transformer + BPE tokenizer) |
| `train_slot_jepa.py` | Slot attention 变体(失败实验) |
| `train.py` | DDP 工具函数 + 实验 01-03 的训练器 |
| `train_wordpred.py` | 词预测(MLM)实验 02 |
| `train_pixel_cbow.py` | pixel-CBOW 实验(未完整评估) |

### 评测脚本
| 文件 | 说明 |
|---|---|
| `probe_zeroshot.py` | **主探针**:训练分布 zero-shot 词-图检索(+ `--blank_text` 无泄漏选项) |
| `probe_collapse.py` | 特征塌缩诊断(photo/text/word pairwise cos) |
| `probe_clip.py` | CLIP 词级 + 句子级检索 |
| `probe_slot.py` | Slot 版 zero-shot 检索 |
| `probe.py` | syn/ant 探针(OOD,已弃用) |
| `probe_nouns.py` | 名词聚类(OOD,已弃用) |
| `retrieval.py` | 词身份检索 |

### 核心模块
| 文件 | 说明 |
|---|---|
| `backbones.py` | 8 个 backbone(ConvNeXt / ConvViT / WindowViT / Hiera / PVT / ...) |
| `model.py` | TextJEPA + SIGReg(防塌缩正则) |
| `render.py` | 文字渲染器(auto-fit 字号 + per-word bbox + mask) |
| `pred_model.py` | PredictiveJEPA(EMA / stopgrad / VICReg) |

### 数据与工具
| 文件 | 说明 |
|---|---|
| `dataset.py` | TextImageDataset(parquet 加载 + 增强) |
| `clip_tokenizer/` | 离线 CLIP BPE tokenizer(via hf-mirror) |
| `run_overnight.sh` | 过夜训练脚本(5 实验串行) |
| `show_log.py` | 训练日志查看器 |
| `preview.py` | 渲染预览 |

### 实验记录
| 文件 | 说明 |
|---|---|
| `experiments/01_global_invariance.md` | 全局不变性 JEPA(负结果) |
| `experiments/02_augmentation.md` | +强增强(认字但不理解) |
| `experiments/03_twostage_semantics.md` | 两阶段预测(原理证明:像素无语义信号) |
| `experiments/04_cross_modal_jepa.md` | 跨模态 JEPA(首个真实 grounding) |
| `experiments/05_clip_baseline_and_slot.md` | CLIP 基线 + Slot attention(最终对比) |

---

## 复现

### 环境
```bash
# 开发机
conda create -n omni-jepa python=3.10
pip install torch transformers pillow numpy tqdm

# 计算机检查
python -c "import torch; print(torch.__version__)"  # 需支持 bf16 + SDPA
```

### 数据
```
recap-datacomp-384-1M/
├── data-00000.tar ~ data-00080.tar   # 81 个 tar, 每个 ~4.4k 图文对
└── 每个 tar: {name}.jpg + {name}.json({"caption": "..."})
```

### 训练:跨模态 JEPA(我们的方法)
```bash
torchrun --nproc_per_node=8 --rdzv-endpoint 127.0.0.1:29500 \
  train_crossmodal_jepa.py \
  --tar_dir /path/to/recap-datacomp-384-1M \
  --num_tars 81 --epochs 100 \
  --batch 256 --workers 16 \
  --hidden 768 --layers 12 --heads 12 --pred_depth 4 \
  --w_mse 0.3 --w_clip 1.0 --lam 0.1 --lr 3e-4 \
  --out outputs/xmodal
```

### 训练:CLIP 基线
```bash
torchrun --nproc_per_node=8 --rdzv-endpoint 127.0.0.1:29500 \
  train_clip.py \
  --tar_dir /path/to/recap-datacomp-384-1M \
  --num_tars 81 --epochs 100 \
  --batch 512 --workers 16 \
  --hidden 768 --text_layers 12 --text_heads 12 \
  --lr 5e-4 \
  --out outputs/clip
```

### 评测
```bash
# 我们的(无泄漏)
CUDA_VISIBLE_DEVICES=0 python probe_zeroshot.py \
  --ckpt outputs/xmodal/epoch99.pt \
  --tar_dir /path/to/recap-datacomp-384-1M \
  --blank_text 1

# CLIP
CUDA_VISIBLE_DEVICES=0 python probe_clip.py \
  --ckpt outputs/clip/epoch99.pt \
  --tar_dir /path/to/recap-datacomp-384-1M
```

---

## 纯视觉通路的价值

尽管效率仅为 CLIP 的 25%,纯视觉通路在特定场景有不可替代的优势:

1. **Screen reading**:屏幕内容是像素化的文字 + 图形。纯视觉方法不需要 OCR 前置,直接从屏幕像素理解内容。在网页 / UI / 文档理解场景,省去 OCR + 双系统流水线。

2. **任意字体 / 语言 / 排版**:不依赖固定词表或 tokenizer。天然适应手写文字、艺术字体、多语言混排。

3. **认知科学意义**:验证了"纯视觉通路能否学词义"——答案是 yes(12x random),但效率低于 tokenizer 方式。这为"人类阅读是否依赖专门的文字处理通道"这一认知科学问题提供了计算层面的参照。

---

## 关键经验教训

1. **探针必须匹配训练分布**:OOD 探针(白色下半)导致假塌缩(cos 0.97+),掩盖了所有信号。从训练分布提取特征后才看到真实的健康分布(cos≈0, std=0.17)。
2. **渲染要验证**:初始坐标 bug 让 text 区空白 15 epoch,浪费了大量调试时间。
3. **防塌缩 ≠ 保语义**:SIGReg 成功防止分布塌缩,但词特征的语义结构仍然很弱(词间 cos 0.055,几乎正交)。
4. **InfoNCE 同一 forward 有泄漏风险**:正负样本来自同一 encoder forward pass,photo 特征可能编码了上半 caption 信息。需要 `--blank_text` 验证。
5. **更多 epoch ≠ 更好**:100 epoch 最优,250 过拟合。InfoNCE 的过拟合表现为"记住 batch 配对但泛化退化"。
6. **Slot attention 不是银弹**:在小 slot 数(K=16-32)下导致语义压扁,反不如 cell mean pooling。

---

## 论文角度

### 定位
"纯视觉通路 vs tokenizer 方式的量化对比"——在同等数据 + 同等图像 encoder 下,量化"不做 tokenization"的代价。

### 核心贡献
1. **量化结论**:纯视觉通路效率 = tokenizer 的 ~25%(top-5 13.7% vs 55.7%)
2. **方法验证**:词级 CLIP grounding 可以从像素学到真实语义(12x random,无泄漏验证)
3. **诊断工具**:训练分布探针(probe_collapse / probe_zeroshot)——避免 OOD 探针的假塌缩
4. **瓶颈分析**:三层效率损失(OCR 负担 / encoder 分工 / 词特征质量)

### 适用场景
- **Screen understanding**(网页 / UI / 文档):不需要 OCR 前置
- **多语言 / 跨字体**:不依赖特定语言的 tokenizer
- **认知建模**:模拟人类视觉通路的文字处理
