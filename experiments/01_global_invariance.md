# 实验 1:全局不变性 JEPA — 直接训练能否学到有效表征

- 日期:2026-07-09
- 结论:**这种"直接训练"几乎不可能学到有效(词义)表征**。能学到表层布局/长度/字符统计的判别,但学不到词级语义。
- 状态:负结果(对"学含义"目标)/ 正结果(对"骨架选型"与"屏幕布局表征"目标)

## 1. 目标与假设
- 假设:把英文渲染成像素图,对随机词做 mask,用 **全局不变性 JEPA**(整句 full vs masked 的 CLS 表征对齐)训练,验证能否从纯 CV 预训练"看懂"语义。
- 直觉来源:人跳读会跳过几个词但不影响理解 → mask 前后表征应对齐。

## 2. 设置
- 数据:common_corpus 的 English 子集,钉定 `subset_100_[1-9].parquet`(固定 snapshot hash),约 50 万句。
- 目标函数:VICReg 式 = `MSE(z_full, z_masked) + λ·SIGReg(z)`,`λ=0.1`。无 predictor / 无 EMA / 无 stop-grad(靠 SIGReg + BatchNorm 防塌方)。
- 共享超参:`lr=3e-4`, AdamW, `mask_ratio=0.2`, `font_size=72`(密排填满 224×224), `epochs=30`, bf16, 5×H800 DDP。
- 骨架对照(均 ~70-100M):
  - convnext(76M,纯卷积层级)→ bs 256/GPU
  - convvit(92M,conv 茎 + Transformer)→ bs 192/GPU
  - windowvit(86M,SAM 式窗口注意力,patch8)→ bs 128/GPU
- 评测:`show_log`(val margin = cos_same − cos_diff)+ `probe`(近/反义词图像 cosine)。

## 3. 结果

| arch | 训练稳定性 | train loss | val margin(末) | 探针 syn/ant/rand |
|---|---|---|---|---|
| **convnext** | ✅ 丝滑,全程无抖动 | 1.53 → 0.146 | **0.98**(cos_same 0.984 / cos_diff 0.001) | 0.548 / 0.592 / **0.557** |
| convvit | ⚠️ e10 起抖动,e29 爆炸 | 0.28 → 0.98 | 蹦动剧烈,末态 0.60 但 cos_same 掉到 0.60 | 0.902 / 0.917 / 0.911(词级塌方) |
| windowvit | ❌ e8 即发散 | 0.24 → 1.26 | 未收敛 | — |

### convnext 训练曲线(代表)
- loss 30 epoch 单调下降,`std≈1.0`、`dead_dim=0` 全程 → 防塌方成功。
- val margin:0.65(e0)→ 0.98(e15 后稳定)。判别力很强。

### 探针(convnext, epoch29)
```
synonyms=0.548  antonyms=0.592  random=0.557
syn-random=-0.009   ant-random=+0.035   ≈ 0
```
近义词、反义词、随机词对的 cosine 完全无序(random 甚至 ≥ syn)。无任何语义结构。

## 4. 结论与原因

### 4.1 骨架选型(正结果,Fig.1 候选)
- **convnext 完胜**:最稳、判别最强、不塌方。
- convvit / windowvit(裸 from-scratch Transformer + 当前目标)训练不稳、中途爆炸。
- 启示:文本/屏幕这种高频局部信号,**卷积前端远比裸 Transformer 稳**。

### 4.2 学不到语义(负结果,核心)
全局不变性目标 **逻辑上不可能** 涌现词级语义:
1. 它奖励模型对被 mask 的词**不敏感** —— 与"编码词身份/含义"正相反;
2. SIGReg 把不同句子推开,但靠的是**布局/长度/字符统计**等表层视觉特征(margin 能到 0.98,好看但无语义);
3. 全局 CLS pooling 抹掉词级信息;
4. 探针对单词图(OOD)进一步放大了上述缺陷。

→ "人跳读"的直觉对**布局/语用稳定性**成立,对**词义学习**不成立。

## 5. 可复现 / 溯源
- 骨架实现:`backbones.py`;模型:`model.py`;训练:`train.py`;探针:`probe.py`;日志查看:`show_log.py`。
- 每条 run 实际读取的 parquet 写在 `outputs/cmp2_<arch>/datasource.txt`。
- 原始日志:`outputs/cmp2_<arch>/{log.csv,val_log.csv}`。

## 6. 下一步方向
- 若目标=**证明从像素学语义**:换**预测式 JEPA**(用上下文预测被 mask 词/区域的 embedding,强制编码词身份),骨架沿用 convnext。这是唯一可行路径。
- 若目标=**屏幕布局/稳定表征**:当前不变性 + convnext 已成立,推进到屏幕分辨率;语义探针改为布局/OCR 一致性探针。
- convvit/windowvit 若要救:需 EMA target + stop-grad + 更低 lr,或换为 conv 茎前置。
