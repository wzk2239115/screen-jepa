# 实验 9: VLA 方向规划 — 从 TC-JEPA 到屏幕智能体

- 日期:2026-08-05
- 状态:规划中，multi-caption 生成进行中
- 前置:实验 8 TC-JEPA v5 (top-5=15.0%, 42x random)，确认文本条件化路线可行

## 1. 背景与动机

### 1.1 三篇关键工作

**TC-JEPA (我们的实验 8)**
- 用 T5 tokenizer + cross-attention predictor 学 patch-level 视觉特征
- v5 达到 42x lift（接近 CLIP 的 53x），sp 全程保持活跃
- 结论：文本条件化 JEPA 是有效的视觉表示学习方法

**Mage-VL (Microsoft)**
- codec-native 视觉处理：选择性编码动态区域（16×16 patch 级别）
- 双系统架构：System 1 快速事件门 + System 2 完整 VLM
- 在 100M 无标注数据上从头训练，匹配千亿级模型性能
- 启发：高效 streaming + 选择性 patch 处理

**D2E (Desktop to Embodied AI, ICLR 2026)**
- 桌面交互数据（screen + keyboard/mouse）作为具身 AI 预训练基底
- 273 小时同步录制（29 个 PC 游戏）+ YouTube 伪标注 → 1.3K 小时
- Generalist-IDM：从观测转换预测动作，支持 OOD 泛化
- VAPT：桌面预训练表征迁移到机器人操作（LIBERO 96.6%, CANVAS 83.3%）
- 启发：桌面数据是廉价、高多样性、可规模化的动作数据源

**DeepSeek "Thinking with Visual Primitives"（已下架）**
- 核心发现：坐标比文字描述更适合做视觉指代
- 视觉原语 = 边界框（指代物体）+ 坐标点（追踪轨迹）
- 坐标作为推理过程的中间步骤（视觉链式推理），不只是最终输出
- 7056 倍压缩：571k 像素 → 81 个视觉 KV 条目
- 拓扑推理大幅领先：迷宫导航 66.9%（GPT-5 ~50%），路径追踪 56.7%（GPT-5 46.5%）
- 启发：精确引用比高分辨率更重要

### 1.2 我们的位置

```
实验 1-7:  Screen-JEPA（纯像素学词义）→ 19x lift，已穷尽
实验 8:    TC-JEPA（文本条件化）→ 42x lift，验证成功
实验 9:    VLA（视觉-语言-动作）→ 从表示学习走向智能体
```

## 2. 三阶段规划

### Phase 1: TC-JEPA multi-caption 预训练（进行中）

**目标**: 用 Qwen3-VL-8B 生成高质量 multi-caption，提升 TC-JEPA 特征质量

**进展**:
- 用 Qwen3-VL-8B-Instruct 对 recap-datacomp 生成 2 caption/image（1 short + 1 detailed）
- batch=8 + flash_attention_2，11.6 img/s，约 5 小时完成全部 807k 图
- 输出：`/home/jovyan/h800fast/wangzekai/recap-multicap/*.captions.json`

**下一步**:
- [ ] 修改 `train_tc_jepa.py` 支持 multi-caption 训练（每图采样 N=2 caption，独立条件化，MaxPool 融合）
- [ ] 修改 `tc_jepa.py` forward 支持多 caption 输入
- [ ] 用 multi-caption 数据训练 v6，对比 v5 的 42x lift
- [ ] 同时尝试 T5-base 替代 T5-small
- [ ] 训练 200+ epoch（v5 在 100ep 仍在上升）

### Phase 2: D2E 式 action 预训练

**目标**: 在 TC-JEPA encoder 基础上加 action prediction

**数据方案**:
- [ ] 调研 D2E 数据集格式（OWA Toolkit，screen + mouse/keyboard events）
- [ ] 下载 D2E-Original（273h，29 游戏）或自己采集桌面数据
- [ ] 统一数据格式：(screen_frame, action) pairs

**模型方案**:
```
Screen frame → TC-JEPA encoder (frozen/finetuned) → visual features (196, 768)
                                                          │
Frame_t+Δ ──→ TC-JEPA encoder ──→ features_t+Δ ──────────┤
                                                          ▼
                                                Action Head (mouse xy + keyboard)
```

**训练方案**:
- [ ] Generalist Inverse Dynamics Model：给定 (frame_t, frame_t+Δ)，预测 action_t
- [ ] NEP-τ：用 τ=100ms 的未来帧作为额外上下文
- [ ] 冻结 encoder vs 全量微调，对比效果

**关键设计**:
- [ ] mouse action 用连续坐标 表示，不用离散化
- [ ] keyboard action 用 multi-label 分类
- [ ] 考虑加 coordinate prediction 辅助任务（DeepSeek 启发）

### Phase 3: Mage-VL 式 VLA 智能体

**目标**: 构建可交互的屏幕智能体

**架构**:
```
                    ┌─── System 1 (快速门) ────┐
Screen stream ──────┤  检测"有变化需要动作"     ├── 无变化 → 保持静默
                    │  轻量 encoder + MLP      │
                    └──────────────────────────┘
                               │ 有变化
                               ▼
                    ┌─── System 2 (完整推理) ──┐
Task instruction ──┤  TC-JEPA features         │
                    │  + LLM decoder            ├── 输出 action
                    │  + coordinate reasoning   │   (mouse xy / keyboard)
                    │  + action head            │
                    └──────────────────────────┘
```

**DeepSeek 视觉原语集成**:
- [ ] 在推理过程中输出中间坐标（"我关注 (x1,y1)-(x2,y2) 区域"）
- [ ] 基于坐标区域做下一步推理（visual chain-of-thought）
- [ ] 坐标不只是 action 输出，而是推理的中间锚点

**Mage-VL 效率优化**:
- [ ] Codec-native patch 选择：只处理屏幕变化的区域（鼠标移动、弹窗、动画）
- [ ] I-frame（完整编码）+ P-frame（只编码变化 patch）
- [ ] 减少 visual token 75%+，实现实时推理

**训练流程**:
- [ ] Stage 1: TC-JEPA 预训练（Phase 1 完成）
- [ ] Stage 2: Action 预训练（Phase 2 完成）
- [ ] Stage 3: 多任务 SFT（任务理解 + 动作生成）
- [ ] Stage 4: RL 微调（环境反馈）

## 3. 评测指标

| 阶段 | 指标 | 目标 |
|---|---|---|
| Phase 1 | Image→Word retrieval lift | >50x (接近 CLIP) |
| Phase 2 | Action prediction accuracy | mouse Pearson >0.7, keyboard acc >60% |
| Phase 2 | LIBERO/CANVAS 迁移 | 对标 D2E (96.6% / 83.3%) |
| Phase 3 | 屏幕任务完成率 | TBD（需定义具体任务） |
| Phase 3 | 推理延迟 | <500ms/frame (System 1) |

## 4. 关键设计决策

### 4.1 坐标作为视觉原语（DeepSeek 启发）

传统方法：`pixel → "我看到了一个按钮" → click(button_name)`

DeepSeek 方法：`pixel → "坐标 (120,340) 处有可点击元素" → click(120, 340)`

**为什么坐标更好**:
1. 无歧义：(120, 340) 只指向一个位置
2. 可组合：多个坐标可以描述区域、路径
3. 可推理：坐标间的距离、方向可以计算
4. 天然连接 action：mouse 事件本身就是坐标

**训练中如何实现**:
- 加 coordinate regression head：encoder features → predict key point coordinates
- 加 IoU loss：predict bounding box → compare with ground truth
- 在 LLM decoder 中允许输出坐标 token

### 4.2 桌面数据 vs 机器人数据（D2E 启发）

D2E 证明桌面数据可以迁移到机器人任务。对我们的优势：
1. 桌面数据极廉价（录屏即可）
2. 高多样性（不同应用、不同 UI）
3. 天然包含文本/OCR 场景（适合 TC-JEPA 特征）
4. 可规模化（YouTube 游戏视频 = 无限数据）

### 4.3 效率优先（Mage-VL 启发）

屏幕智能体需要实时响应。Mage-VL 的 codec-native 方法启发：
1. 只处理变化的 patch（屏幕大部分区域是静态的）
2. 双系统避免不必要的完整推理
3. 在 16×16 patch 级别工作（与 TC-JEPA encoder 匹配）

## 5. TODO 清单

### 立即（Phase 1）
- [x] Qwen3-VL caption 生成脚本（batch + flash attention）
- [ ] multi-caption 数据生成完成（~5h）
- [ ] 修改 train_tc_jepa.py 支持 multi-caption
- [ ] 修改 tc_jepa.py forward 支持多 caption MaxPool 融合
- [ ] TC-JEPA v6 训练（multi-caption + T5-base + 200ep）

### 短期（Phase 2）
- [ ] 调研 D2E / OWA Toolkit 数据格式
- [ ] 下载或采集桌面交互数据
- [ ] 实现 action prediction head
- [ ] Generalist-IDM 训练
- [ ] LIBERO/CANVAS 评测

### 中期（Phase 3）
- [ ] LLM decoder 集成
- [ ] coordinate reasoning 实现
- [ ] System 1 / System 2 双系统
- [ ] 屏幕任务 benchmark 定义
- [ ] RL 微调

## 6. 相关文献

| 论文 | 关键贡献 | 与我们的关系 |
|---|---|---|
| TC-JEPA (ICML 2025) | Text-conditioned JEPA | 我们的预训练方法 |
| D2E (ICLR 2026) | 桌面数据→具身 AI 迁移 | 数据和训练范式 |
| Mage-VL (Microsoft) | Codec-native streaming VLM | 效率优化 |
| DeepSeek Visual Primitives | 坐标作为推理原语 | action 表示和推理方式 |
| I-JEPA (Meta) | Masked predictive learning | JEPA 基础框架 |
| ShareGPT4V | 多 caption 数据生成 | caption 质量参考 |
