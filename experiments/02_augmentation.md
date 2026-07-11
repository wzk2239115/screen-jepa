# 实验 2:强增强下的视觉词识别(认字)—— 0.152 → 0.431

- 日期:2026-07-10
- 结论:**背景/字体/缩放增强把"靠背景与字形偷懒"的路堵死,模型被迫真正认字,词身份检索从 0.152 飙到 0.431(64× 随机)。但语义(syn/ant)仍不出现。**
- 状态:正结果(认字)+ 负结果(语义)

## 1. 动机
实验 1 的不变性 convnext 稳定但词身份弱(retrieval ≈ 0.15)、语义无。假设:模型在靠**背景色 / 特定字形 / 精确布局**偷懒。若强制这些因素随机变化,模型无法依赖它们,只能抓更抽象的字符特征 → 认字能力应提升。

## 2. 增强(逐步叠加,全部作用于增强视角)
- **色块拼贴背景** `--bg_block`:2×2 / 4×4 随机浅色块(替代单色背景)。
- **随机字体** `--font_augment`:DejaVu(无衬线/衬线/等宽/粗)+ Liberation 三族共 7 种。
- **缩放(zoom-out only)** `--geom_strength 1`:0.92–1.0 缩小(只缩小不放大,不旋转、不倾斜,避免文字甩出画布)。
- mask(遮 20% 词)延续。

共享:invariance 目标(online encoder + SIGReg + BN,lewm 那套稳定配方)、convnext 768/12、224×224、密排渲染 font72、English common_corpus subset_100_[1-9](174 万句)、5×H800。

## 3. 结果
| 配置 | retrieval top-1 | 备注 |
|---|---|---|
| 无增强 / 单色背景(exp1) | ≈ 0.15 (flatten) | 基线 |
| 仅背景增强(bg_convnext) | 0.152 (mean) | 微弱 |
| **bg_block + font + geom(aug_L1)** | **0.431 (mean)** | **3× 跃升,64× 随机** |

- 训练全程稳定:`std≈1.0`、`dead=0`、val margin 0.32→**0.975**,无塌方/无爆炸。
- retrieval mean 的 same/diff gap 仅 +0.006、std 0.005 → embedding 高度聚拢,但聚拢内的**微小差异**已足以让 NN 检索到 43% top-1。

### 语义探针(aug_L1, epoch49)—— 仍无
```
mean:    syn=0.992  ant=0.993  random=0.993   syn-random=-0.001
flatten: syn=0.959  ant=0.964  random=0.963   syn-random=-0.004
```
mean 与 flatten 两种聚合下,近/反/随机都无序。

## 4. 结论
- **正**:增强假设被验证 —— 强增强让认字能力 3 倍提升(0.15→0.43)。模型**确实从像素学会了区分词**(150 词 43% top-1)。
- **负**:语义在 mean 与 flatten 下都不出现。增强只逼出"更抽象的字符特征",但**字符特征 ≠ 语义**;目标函数里始终没有"把近义词拉近"的信号。
- 启示:认字(视觉词身份)可由"像素 + 强增强"获得,这正是 screen-jepa 看屏/OCR 方向的真价值。

## 5. 溯源
- run: `outputs/aug_L1/`(log.csv、val_log.csv、epoch49.pt、datasource.txt)
- 代码:`render.py`(patchwork_bg/geom_augment)、`dataset.py`(bg_block/font_augment/geom_strength)、`retrieval.py`(多聚合检索)、`probe.py`(--mode flatten)
