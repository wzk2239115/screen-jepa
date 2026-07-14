# 实验 4:跨模态 JEPA — 真实图文对的词级 CLIP grounding

- 日期:2026-07-13 ~ 2026-07-14
- 结论:**纯视觉通路学到了真实的词-图 grounding(top-5=13.7%, 12x random,无泄漏验证)。但存在两个关键 bug(渲染空白 + 探针 OOD),修复后才显现信号。词级 CLIP 是唯一产生真实 grounding 信号的目标函数。**
- 状态:正结果(grounding 涌现)+ 两个 bug 发现(渲染 + 探针)

## 1. 目标与假设

- **假设**:把渲染的 caption(上半)和真实 photo(下半)拼在一张 224×224 合成图里,过 convnext 得到 14×14 feature map。让每个词的 cell 特征(从 feature map 按 bbox pooling)与图片特征(下半 cells)做 InfoNCE 对齐 → 模型从"词→图片分布"的映射中学到词义。
- **与实验 1-3 的区别**:之前用纯文本(无图片)做不变性/预测/MLM,没有任何外部语义锚点。本实验引入**真实图片作为语义锚点**——词义通过"这个词出现在什么图片里"来定义。
- **与 CLIP 的区别**:不用 tokenizer,文字渲染成像素,和图片共享同一个 encoder。

## 2. 数据

- **recap-datacomp-384-1M**:81 个 tar × ~4.4k 图文对(共 ~356k 对)。384px 图像缩放到 224×224。caption 为 DataComp-recap(LLM 重写的描述性 caption,平均 18-74 词)。
- 开发环境:`/home/wzk/datasets/recap-datacomp-384-1M`
- 计算环境:`/home/jovyan/h800fast/wangzekai/recap-datacomp-384-1M`

## 3. 架构与目标函数

### 合成图设计
```
┌──────────────────┐
│  渲染的 caption   │ 112px (text 区, grid rows 0-6)
│  (auto-fit字号)   │
├──────────────────┤
│  真实 photo      │ 112px (photo 区, grid rows 7-13)
│  (resize to 224)  │
└──────────────────┘
         224px
```

### CrossModalJEPA 模型
- **encoder**: ConvNeXt(base_dim=192, ~100M params),输出 14×14 feature map
- **predictor**: 4 层 Transformer(JEPA masked latent prediction)
- **target encoder**: EMA(τ=0.996)
- **SIGReg**: 防塌缩正则

### 损失函数
```
L = w_mse · (MSE_pred_text + MSE_pred_photo)    ← JEPA 双向 latent 预测
  + w_clip · InfoNCE(word_feats, photo_feats)    ← 词级 CLIP 对齐
  + lam · SIGReg(ctx)                             ← 防塌缩
```

词特征 = 每个词 bbox 覆盖的 cells 做 masked-mean pooling
图片特征 = photo 区 cells 的 mean pooling
InfoNCE:正样本 = 同一合成图的词-图对;负样本 = batch 内其他图

## 4. 关键 bug 发现与修复

### Bug 1:渲染空白(根因级 bug)
- `render_caption_block` 初始坐标 `y=hh, x=ww`(起点在右下角外),任何字号都 fit 失败 → **text 区一直是空白!**
- **影响**:实验 4a(纯 JEPA,无词级 CLIP)的 15 epoch 完全无效——模型从没见过文字渲染。
- **修复**:`y=0, x=6`。修复后确认渲染正确(dark pixel frac ~9.4%)。

### Bug 2:探针 OOD(测量假象)
- 所有探针(syn/ant、probe_nouns)用"单词渲染 + **白色下半**"测试。
- 模型训练时下半总是 photo,从没见过纯白下半 → **探针特征自身塌缩到 cos 0.97-0.98**。
- **影响**:syn-rand 全程为 0(不是模型没语义,是探针在 OOD 输入上退化了)。
- **诊断**:`probe_collapse.py` 从训练分布(真实合成图)提取特征,发现 photo/text 全局特征完全健康(cos mean≈0, std=0.177)。

## 5. 实验序列与结果

### 5a. 纯 JEPA(无词级 CLIP)— 渲染 bug 未修
| 配置 | epochs | syn-rand | ant-rand |
|---|---|---|---|
| 40 tars, 单卡 | 15 | ±0.02(噪声) | ±0.01(噪声) |

text 区空白,模型只看到 photo。无任何语义信号。

### 5b. 渲染修复 + 词级 CLIP(w_mse=0.3, w_clip=1.0)
| 配置 | epochs | clip loss | syn-rand | ant-rand |
|---|---|---|---|---|
| 40 tars, 单卡 | 40 | 3.7→0.012 | -0.01(噪声) | **+0.03~0.07** |
| 81 tars, 跨卡gather | 20 | →0.088 | -0.01 | +0.03 |
| 81 tars, 跨卡gather | 100 | →0.012 | -0.01 | +0.03 |

- **ant-rand 持续正值(+0.03~0.07)**:第一个真实语义信号!反义词(hot/cold)共享图像语境,grounding 后特征接近。
- **syn-rand 全程为 0**:同义词(big/large)的图片分布分散,grounding 不会让它们接近。
- clip loss 饱和在 0.012(98.8% 正确匹配),但更多 epoch/数据/负样本不改善 ant-rand。

### 5c. 探针诊断与修复

#### probe_collapse.py:训练分布特征诊断
```
[1] PHOTO pairwise cos:  mean=0.0008  std=0.177   ← 完美均匀,无塌缩!
[2] TEXT-GLOBAL cos:     mean=0.0007  std=0.176   ← 也健康
[3] intra-word cos:      0.065  inter-word cos: 0.036  margin: +0.029
[5] word vs photo SAME image: 0.156
    word vs photo DIFF image: -0.010
    gap: +0.166  ← 真实对齐信号!
```

**模型完全没有塌缩!** 特征均匀分布,gap=+0.166 是真实的词-图对齐。之前的探针(白色下半)全在 OOD 上测量,产生假塌缩。

#### probe_zeroshot.py:训练分布 zero-shot 检索
从真实合成图(caption+photo)提取词原型,在 held-out 图上做检索。

| 配置 | top-1 | top-5 | top-10 | MRR |
|---|---|---|---|---|
| 上半=caption(有泄漏) | 2.3% | 14.0% | 26.7% | 0.109 |
| **上半=空白(无泄漏)** | **2.0%** | **12.7%** | **24.3%** | **0.096** |

**无泄漏验证**:去掉 caption 后只降 ~10%。模型不是靠"读上半文字"作弊,是真实的图片理解。top-5 = 12.7%(vocab=481, random=1.04%)→ **12x random**。

### 5d. 饱和确认
| 变量 | 变化 | top-5 |
|---|---|---|
| epoch | 20→100 | 12.7%→13.7%(+1%) |
| 数据 | 40→81 tars | 不变 |
| 负样本 | 256→1280(跨卡gather) | 不变 |

**方法彻底饱和,top-5 卡在 ~14%。** 瓶颈在架构层面。

## 6. 结论

1. **词级 CLIP 是唯一产生真实 grounding 信号的目标**(ant-rand +0.03~0.07,zero-shot 12x random)。之前所有目标(不变性、预测、MLM)都是噪声。
2. **两个 bug 掩盖了信号**:
   - 渲染 bug:text 区空白(15 epoch 完全无效)
   - 探针 OOD:白色下半导致测量假塌缩(syn-rand 全程为 0)
3. **纯视觉通路可行但效率低**:top-5=14% 是真实的(无泄漏验证),但远低于 tokenizer 方式(CLIP=51-56%)。
4. **瓶颈在架构**:共享 encoder + 像素渲染的表达能力,不是数据/epoch/负样本能解决的。

## 7. 代码文件
- `train_crossmodal_jepa.py`:CrossModalJEPA 模型 + 词级 CLIP + 跨卡 gather + 渲染
- `probe_collapse.py`:训练分布特征塌缩诊断
- `probe_zeroshot.py`:训练分布 zero-shot 检索(+ `--blank_text` 无泄漏选项)
- `probe_nouns.py`:名词聚类 + zero-shot 分类(OOD 探针,已弃用)
