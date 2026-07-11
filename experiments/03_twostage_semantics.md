# 实验 3:两阶段预测式 —— 尝试从"认字"boot 出"语义",失败 + 原理证明

- 日期:2026-07-10
- 结论:**两阶段(冻结 Stage1 认字 encoder → 训 predictor 从上下文预测被遮词)学不出来;predictor 始终在随机水平。根因是"语义盲的上下文"——纯像素表征无法 boot 出语义。**
- 状态:负结果(带原理证明)

## 1. 思路(两阶段)
- **Stage 1**(已完成,实验 2 的 aug_L1):学"每个词长什么样" → 冻结的、词区分的 encoder(retrieval 0.43)。
- **Stage 2**:冻结 Stage1 encoder 作 target,只训 predictor:遮一个词 → 从上下文预测该词的 Stage1-latent。
- 期望:target 稳定且词区分(不再联合训练、不再 EMA 滞后),predictor 能学;"从上下文猜是哪个词"即分布/语义任务 → 涌现语义。

## 2. 联合训练预测式先全部失败(背景)
预测式 JEPA(encoder + predictor 联合训)跨所有配置都训不起来:
| 配置 | pred loss | encoder |
|---|---|---|
| EMA / 无增强 | 卡死 | 塌 |
| stopgrad / 无增强 | 不降 | 塌 |
| EMA + 增强 | 涨 | 不塌但 pred 不学 |
| stopgrad + 增强 + VICReg | 涨 | 不塌但 pred 不学 |
| stopgrad + 增强 + lam0 | 平 | 塌 |

根因:encoder 联合训练 → target 漂移 / 塌方。

## 3. Stage 2(冻结 encoder)— 机制通了,但仍学不出语义

### 3.1 MSE 回归版
- pred loss 这次会降了(冻结 target 拔掉了漂移根)。
- **但 fill-in 无信号**:correct≈syn≈ant≈random≈0.52。
- **诊断**:predictor 输出**全局均值**。MSE 在"目标预测不准"时的最优解就是条件均值 = 常量,合法偷懒。

### 3.2 对比式 InfoNCE 版(逼 predictor commit)
- 损失换成 InfoNCE:predictor 的 query 必须比 batch 内其他词的 latent 更接近"正确词"的 Stage1-latent。
- **结果**:`ncr`(选对比例)20 epoch 仅 **0.009**,chance≈1/150≈**0.0067** —— 几乎贴着随机。
- predictor 学不出"从上下文猜词"。

## 4. 原理证明:为什么 boot 不起来
Stage 2 要"从上下文猜被遮的词",需要**语义上下文**。但 Stage 1 的 encoder 是**纯视觉**的(实验 1/2 反复验证:认字强、无语义)。于是:

> 上下文特征里**没有语义信息** → predictor 无法判断"这个位置该是哪个词" → 只能瞎猜 → ncr 卡在 chance。

**鸡生蛋的精确断点**:要预测被遮词必须先有语义;要学语义必须靠预测被遮词。纯像素的 Stage 1 给不出语义,Stage 2 就 boot 不起来。**scaling 救不了**——这是目标/表征的固有问题,不是算力问题(对比式是"必 commit"的硬约束,仍学不出,说明不是 MSE 偷懒,而是上下文真的不含可用信息)。

## 5. 最终结论(整个选型+上限探索阶段的收口)
- ✅ 纯像素 JEPA + 强增强 → **鲁棒视觉词身份**(retrieval 0.43),对接看屏/OCR。
- ❌ **学不到词义**,现已原理证明:
  1. 不变性目标无"近义词拉近"信号(实验 1/2);
  2. 预测式无法 boot(联合训塌、两阶段上下文语义盲 → ncr 卡 chance)(实验 3)。
- **唯一出路**:从外部注入语义信号(token 标签 / 文本 encoder 当 teacher 做蒸馏)。那不再是"纯像素自监督",是另一篇(像素 encoder 从文本 teacher 吸收语义的多模态蒸馏)。

## 6. 溯源
- runs:`outputs/stage2/`(MSE)、`outputs/stage2_ctr/`(对比)
- 代码:`pred_model.py`(PredictiveJEPA、init_from/freeze_encoder、stopgrad/EMA、MSE/contrastive、vicreg)、`fill_in.py`(cloze 探针)、`train.py`(--objective/--target_mode/--loss_mode/--init_from/--freeze_encoder)
