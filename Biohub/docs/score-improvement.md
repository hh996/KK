# Biohub 提分复盘与行动计划

> 日期：2026-07-10（已按实测修正）  
> **当前实测 best：public 0.889**（`bio-v2` min6 / `biohub-1` min7）  
> **尚未复现公开 0.897**；行数与权重 sha 均与本地好跑不一致  
> 公开分平台：大量 fork 卡在 **0.897**；LB 头部 **0.910 / 0.906**（方法未完整公开）

## 提交文件对照表

| 文件 | 阶段 | 何时交 |
|------|------|--------|
| `notebooks/biohub-competition-solution.ipynb` | **P0** | **立刻** |
| `notebooks/biohub-s1-min6-deepcenter.ipynb` | P1-1 | P0 ≥0.897 后 |
| `notebooks/biohub-p1-2-min6-gap2.ipynb` | P1-2 | P1-1 后 |

---

## 0. 先读结论（2026-07-10）

| 提交 | 配置 | Rows | public |
|------|------|------|--------|
| **`bio-v2`**（推荐基准） | min6 `057_lb897_min6_restore` | **243,058** | **0.889** |
| `biohub-1` | min7 | ~248,558（历史） | **0.889** |
| `bio-s1 - v2` | min6 + DeepCenter | — | **0.881** ✗ |

**当前基准：勾选 `bio-v2`（min6）。** min6/min7 同分，但 min6 是公开 0.897 源设定，后续 A/B 只在它上面叠。

### 为何 0.889 ≠ 公开 0.897（关键线索 = 行数 + 权重）

| 对照项 | 你的 bio-v2 | 本地好跑 / 公开 0.897 源 |
|--------|-------------|-------------------------|
| Rows | **243,058** | min7≈**248,558**；min6≈**251,611** |
| short_track_nodes_removed | 6,102 | min7≈7,465；min6≈5,795 |
| Weight sha256 | **`12f6881e…`** | **`12b5d32a…`** |
| Artifact name（manifest） | **`…400ep-snapshot-v1`** | **`…300ep-snapshot-v1`** |
| Dataset slug | `…50ep-v1`（同名） | `…50ep-v1`（同名） |

**同 slug `50ep-v1` 下挂了不同权重内容。** 你跑到的是 400ep-snapshot（sha `12f6881e`），本地能打出 248k–251k 的是 300ep-snapshot（sha `12b5d32a`）。行数少 ~8.5k（相对公开 min6）→ recall 不足 → public 停在 0.889。  
`44b6_0b24845f` 单集 short_track 删 3653 nodes 是症状，根因在**上游检测/图节点偏少**（错权重），不是再调 min6/min7。

**硬规则：先对齐权重与行数到公开 0.897 平台，再叠任何后处理。不要在 243k / 错 sha 上继续叠 DeepCenter / gap2。**

---

## 1. 历史复盘

### 1.1 曾以为「min7 → 0.889」

早期判断：0.889 主因是全局 min7。公开对照：calib **0.889**、Min8 **0.890**、Dataset Mintrack **0.892**。

### 1.2 实测修正（2026-07-10）

- min6（`bio-v2`）与 min7（`biohub-1`）**同分 0.889**  
- min6 行数 **243k < min7 的 248k**（反常；正常 min6 应更多）  
- DeepCenter S1 在错误底座上 → **0.881**  
- Artifact 展示名 `400ep` / `300ep` **不是**「只是展示名无害」——**sha 不同 = 权重不同**

---

## 2. 本地 notebook 对照（相对公开 0.897）

共同期望底座：`det=0.99`，motion relink，gap1=6μm，safe_div，linefit，**gap2=关**，artifact=`50ep-v1` 且 **sha=`12b5d32a…`**。

| Notebook | 差异 | 本地输出行数 | 公开/预期 | 现在是否值得交 |
|----------|------|--------------|-----------|----------------|
| **competition-solution min6** | 源设定 | 本地缓存仍是旧 min7 输出 248,558；代码已改 min6 | 对齐权重后应 ~251k / **0.897** | ★★★★★ **下一刀必跑** |
| `yusuke-lb893-fork` | min6 + 略收紧 safe_div | **251,611**；sha=`12b5d32a` | ~0.893 | ★★★ 可作「对齐权重」对照（行数目标） |
| `blend-preprocessings` | DeepCenter + **默认 min7** | 248,569；sha=`12b5d32a` | 页标 0.897 | 权重对齐前勿交；对齐后才改 min6 |
| `lb897-baseline` 059 | 全局 min7 + 单集 min6 | 248,723 | Exp043 **0.892** | ✗ 相对 0.897 掉分 |
| `biohub-s1-min6-deepcenter` | min6+DeepCenter | 无本地全量输出 | 实测 **0.881** | ✗ **停做** |
| `learned-graph` | det=0.97 + refine | 284,162；另一套 sha | 页标误导 | ✗ 勿交 |

---

## 3. 可执行行动清单（按优先级）

### P0.【立刻】对齐公开 0.897 权重，复现 ~251k 行

**目标：** public **≥0.895，理想 0.897**；行数进入 **248k–252k**。

**做法（二选一，优先 A）：**

**A. Fork 已出分 0.897 的公开 notebook 原样跑**

1. Kaggle Code 页 Fork 任一标分 **0.897** 的 `competition-solution` / Blend / Yusuke 同源 notebook  
2. Add Data：确认挂的是 `pilkwang/biohub-tracking-support-pack-50ep-v1`  
3. Run All 后**必须**核对日志：
   - `Weight sha256:` 以 **`12b5d32a`** 开头（与本地好跑一致）  
   - `Artifact name:` 优先见 **`300ep-snapshot`**（若仍是 `400ep` 且 sha=`12f6881e` → 数据集版本错了）  
   - `Wrote … submission.csv with` **≥248,000**（min6 应接近 **251k**）  
4. 达标再 Submit  

**B. 用本地 `biohub-competition-solution.ipynb`（已是 min6 preset）重新挂数据**

1. 上传/同步本地 `notebooks/biohub-competition-solution.ipynb`  
2. Settings → Data：删掉旧 support pack，重新 Add `pilkwang/biohub-tracking-support-pack-50ep-v1`（若 Kaggle 有多版本，选能打出 sha `12b5d32a` 的版本；或从公开 0.897 notebook 的 Input 复制同一 dataset 版本）  
3. GPU T4×2、Internet OFF、Dependencies 空  
4. 自检同 A；`output_min_track_len: 6`；忽略 summary 里残留的 `056_…min7` 打印文案  

**成功判据：**

- [ ] sha = `12b5d32a…`（不是 `12f6881e…`）  
- [ ] Rows ≈ **251k**（min6）或至少 **≥248k**  
- [ ] public **→ ~0.897**（至少明显高于 0.889）  

**失败则：** 不要改后处理；继续查 dataset 版本 / 是否挂错 artifact / 是否 `ALLOW_ARTIFACT_FALLBACK` 吃到旧包。

### P1.【仅当 P0 成功】单模块提分

顺序不变，**底座必须是已验证 0.897 / ~251k**：

1. **min6 + DeepCenter**（改 Blend 默认 7→6，或用修好权重后的 S1）— 预期 0.897–0.899  
2. **纯 min6 + gap2**（无 DeepCenter）— 不确定；`<0.897` 立刻关  
3. 社区已出分 PathFix / refine 接到 min6  

### P2. 停做清单

| 停做 | 原因 |
|------|------|
| DeepCenter S1（当前权重） | 已实测 **0.881** |
| 盲目再交全局 min7 / min8 | 与 min6 同分或更差；不解决 243k 缺口 |
| 在 243k / sha=`12f6881e` 上叠 gap2 / PathFix | 错误底座 |
| 原样交 059 / yusuke 当「涨过 0.897」 | 公开分别约 0.892 / 0.893 |
| 一次改 3 个开关 | 无法归因 |
| 指望公开 notebook 直接到 0.91 | 头部未开源 |

---

## 4. Kaggle 信号（2026-07-09，仍有效）

- 头部：**0.910** / **0.906** — 无完整公开解法  
- **0.898–0.900**：可小幅突破 0.897 平台  
- **0.897**：大量公开 fork 挤在同一分  

Code 页优先：Blend（移植 DeepCenter）、Exp048 gap、Exp046c PathFix；已验证掉分：Exp043 **0.892**、Min8 **0.890**、calib **0.889**。

---

## 5. 每次提交自检

- [ ] Data：`50ep-v1`；**打印 sha 必须是 `12b5d32a…`**  
- [ ] Internet OFF；Dependencies 空  
- [ ] `output_min_track_len=6`（除非刻意测过滤）  
- [ ] 行数：min6≈**251k**；min7≈**248k**；若仍 **~243k** → 权重未对齐，勿交后处理变体  
- [ ] DeepCenter 实验另挂 center-prior，且仅在 P0 成功后  
- [ ] 只改一个主轴；写清 `experiment_tag`

---

## 6. 与 `baseline-guide.md` 的关系

- `baseline-guide.md`：首次上手与资源挂载（其中「min7≈0.897」已过时，以本文为准）  
- 本文：**实测 0.889 / 243k** 之后的纠偏与提分；**先对齐权重复现 0.897，再叠模块**

---

## 7. 其他路线（除「对齐权重 → 0.897」以外）

> 更新：2026-07-10。前提仍是：**P0 对齐 sha=`12b5d32a` / 行数≈251k 优先于一切后处理。**  
> 公开信号（Code/Discussion，2026-07）：大量 fork 卡在 **0.897**；头部 **0.910**（Rahul，方法未开源）；Code 活跃变体含 Blend/DeepCenter、Exp046c PathFix、Exp048 gap、Exp043 dataset-mintrack（**0.892**）、另有 ~**0.894** 未命名 fork；Discussion 有 **rule-based 进金区**（ISAKA）、**Affinity Field**（hengck23）、**LAP 求助帖**（Harshul，本地 CV~0.59）。

### 7.1 在当前 0.889 / 错权重（243k / `12f6881e`）上做 = 浪费提交

| 动作 | 为何浪费 |
|------|----------|
| 再交全局 min7 / min8 / calib | 你已 min6=min7=**0.889**；过滤阈值不是瓶颈 |
| 再交 DeepCenter / S1 | 错底座已实测 **0.881** |
| 在 243k 上开 gap2 / 调 safe_div / 关 motion relink | 上游少 ~8.5k 节点；后处理修不回 recall |
| 原样交 059 / yusuke / learned-graph(det0.97) | 公开分别约 **0.892 / 0.893** / 高 FP；相对 0.897 是掉分或假涨行数 |
| 一次改 3 个开关 | 无法归因；错底座上更无意义 |
| 指望公开 notebook 直接到 0.91 | 头部未开源 |

**唯一值得在「未对齐」阶段花提交的：** Fork 标分 0.897 的公开 notebook **原样**跑，只验证 sha/行数/分数（这是 P0，不是「其他路线」）。

### 7.2 后处理 A/B（必须先到 0.897 / ~251k）

| 优先级 | 方案 | 预期收益 | 风险 | 前置 | 操作成本 |
|--------|------|----------|------|------|----------|
| **A1** | **min6 + DeepCenter**（Blend 改 7→6；挂 `center-prior-v1`） | **+0.000～+0.002**（冲 0.898–0.899） | 中：错权重会再掉（你已见 0.881）；漏挂 prior 会挂 | **必须 0.897** | 低：改 1 行 + 挂 dataset |
| **A2** | **纯 min6 + gap2=1**（默认 cap，无 DC） | **−0.002～+0.002** | 中高：假连接易掉；`<0.897` 立刻关 | **必须 0.897** | 低：翻 1 个开关 |
| **A3** | 移植 **Exp046c PathFix**（仅当 Code 页该本已出分 ≥0.897） | **+0.001～+0.003**（冲 0.898–0.900） | 中高：未出分/掉分版勿跟 | **必须 0.897**；且源 notebook 有分 | 中：diff 移植 |
| **A4** | **dataset-specific mintrack**（059：全局 7 + `6bba_05b6850b`→6） | 相对 0.897 多为 **−0.005**（公开 **0.892**） | 高：已验证掉分 | 勿作涨分刀；仅作对照 | 低 |
| **A5** | **safe_div 微收紧**（yusuke：4.66/7.05/7.65…） | 公开约 **0.893**，相对 0.897 **掉** | 中：少假分裂也少真分裂 | 勿优先；仅当 TRA 分裂项诊断明确 | 低 |
| **A6** | 动 **motion relink / gap1 / linefit**（默认已开） | 期望 ≈0；关了多半掉 | 高：这些是 0.897 骨架的一部分 | 不要关；微调半径仅在 0.897 后且有诊断 | 低～中 |
| **A7** | **det=0.97 + node refine**（learned-graph） | 行数冲到 **284k**，分数多半掉（FP） | 很高 | **禁止**当涨分提交 | 低但有害 |

**推荐顺序（P0 成功后）：** A1 →（若 A1≥0.897）A2 或 A3 二选一 → 再叠；**不要** A4/A5/A7。

### 7.3 检测 / 融合：DeepCenter 为何失败、何时再试

**失败原因（你的 S1=0.881）：** 不是「DeepCenter 无效」，而是 **backbone 权重错**（400ep/`12f6881e` → 节点偏少）时再灌 center-prior，等于在稀疏错误图上加噪声点，short-track/连边更乱。公开 Blend 在 **正确 300ep** 上 min7 仍标 **0.897**（本地仅多 ~11 行），说明 DC 是**边际补点**，救不了错权重。

**再试条件（全部满足才交）：**

1. 纯 min6 已 **public ≥0.895** 且行数 **≈251k**、sha=`12b5d32a`  
2. 日志 `full_frame_added_nodes > 0`、DeepCenter 路径解析成功  
3. 相对纯 min6 **只开 DC**，gap2 保持关  

不满足 → **停做 DC**。

### 7.4 换 artifact / 自训 / TTA / ensemble

| 方案 | 预期收益 | 风险 | 前置 | 操作成本 |
|------|----------|------|------|----------|
| **换/钉死正确 support pack 版本**（同 slug 不同内容） | **0.889→0.897**（这是 P0） | 低 | 无 | 低：重挂数据 |
| **自训 UNet+edge / 新 snapshot** | 冲 **0.90+** 的主路径 | 高：算力、过拟合 public、12h 限时 | 本地 CV/TRA 先过公开包 | **很高**（天～周） |
| **多 checkpoint ensemble**（节点/边概率融合） | **+0.002～+0.010**（头部可能在此） | 中高：显存/时长；融合差则掉 | 先有 ≥2 个接近 0.897 的权重 | 高 |
| **TTA**（翻转/强度） | 检测类 **+0～+0.003**；跟踪图未必涨 | 中：×N 推理易超时 | 0.897 后且单次推理有余量 | 中 |
| **ILP 权重微调**（appear/disappear/division） | 小 | 中：易伤谱系 | 0.897 后 + 有验证指标 | 中 |

公开 **0.910** 无 writeup：默认假设是 **私有检测/跟踪改进或未公开后处理**，不是再拧 min_track。

### 7.5 替代路线：rule-based / LAP / Affinity / Ultrack

| 方案 | 预期收益 | 风险 | 前置 | 操作成本 |
|------|----------|------|------|----------|
| **Rule-based（Discussion：ISAKA，曾金区、号称无学习）** | 可到 **金区**（与 0.897 路线正交） | 高：细节未开源；从 0 复现难 | **不依赖** 0.897；可并行调研 | 高（读帖+自研） |
| **hengck23 Affinity Field** | 概念向：用亲和场指导连边，冲头部潜力 | 高：帖新、无完整可跑代码 | 不依赖 0.897；作中长期 | 高 |
| **LAP / 帧间欧氏分配**（Harshul 帖，CV~0.59） | 入门级；**远低于** 0.889 | 当作主提交 = 浪费 | 仅学习用 | 低 |
| **官方 NN / Ultrack 入门** | 远低于 support-pack 线 | 低 | 勿当冲分 | 低 |
| **train/test 重叠疑虑帖**（LeeWhieldon） | 不影响你选法；勿赌泄露 | — | 忽略作提分手段 | — |

**策略：** 主线仍是 **support-pack 0.897 → 单模块**；rule-based/Affinity 仅在 P0 卡住或要冲 0.90+ 时开第二条线，**不要**用 LAP 消耗每日提交。

### 7.6 可执行下一刀（摘要）

```text
现在（0.889）     → 只做 P0：对齐权重，拿回 ~0.897 / 251k
P0 成功后第 1 刀 → A1 min6+DeepCenter
第 2 刀          → A2 gap2 或 A3 已出分 PathFix（只改一个）
并行调研（不占提交）→ ISAKA rule-based 帖、hengck23 Affinity 帖
冲 0.90+         → 自训 / 多权重 ensemble（公开后处理基本到顶）
```

---

## 8. 论文与社区调研（2026-07-10 实测检索）

> 检索源：Nature Methods / CTC 官网、Zebrahub、Kaggle Code 页（`sortBy=scoreDescending`）、Discussion 页 snippet、WebSearch。  
> **前提不变：** 当前 public **0.889**（243k / sha `12f6881e`）→ 任何后处理 A/B 均应先完成 P0 权重对齐。

### 8.1 论文与方法

| 主题 | 要点 | 与当前 0.889 路线关系 |
|------|------|------------------------|
| **Ultrack**（Royer lab, [Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0)） | 多分割假设 + 时序一致性选段；CTC 密集 3D 胚胎 SOTA；Zebrahub 全胚追踪工具；Python/napari/SLURM | 比赛官方背景方法，**不是** support-pack 0.897 管线（UNet+Transformer+ILP）。12h 限时内难完整复现 TB 级 Ultrack 流程 → **中长期参考**，非立即可 fork |
| **CTC TRA / SEG** | `OPCTB = 0.5·(SEG+TRA)`；TRA = 1−AOGM/AOGM₀；边 FN 权重 1.5× | Biohub 公开分与 CTC 同族图匹配思想。提分 = 少 FP 节点 + 少错边/断轨。**243k 行偏 recall 不足** → 先补检测节点再谈 TRA 边优化 |
| **Affinity Field**（hengck23 帖；文献：MPM CVPR'20、GNN tracking arXiv'22） | 用向量场/亲和图联合检测+连边，替代帧间欧氏 LAP | 与现有 **learned edge + ILP** 同方向但可学更强边代价。帖新、**无可跑公开代码** → 概念向，冲 0.90+ |
| **斑马鱼 3D 近期** | UCM/Ultrack (ECCV'24)；GNN EP-MPNN (CTC BGU-IL)；Sugawara incremental DL (eLife) | 头部 Rahul **0.910** 可能在此类检测/边学习 + ensemble，公开后处理平台 **0.897** |
| **Zebrahub**（[Cell 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01147-4)） | 10 阶段 scRNA + 7 条光片 timelapse；Ultrack 谱系；CC0 | 与比赛同 Royer 组成像栈；理解数据密度/分裂模式 → 调 safe_div、gap 阈值 |

### 8.2 Kaggle Code 区（按 Public Score 降序，2026-07-10）

| Public | Notebook（作者 / 关键词） | 备注 |
|--------|---------------------------|------|
| **0.900** | **neilan — `solution`**（+4 datasets，几乎无公开说明） | **Code 区最高公开分**；方法未披露；**优先 Fork 侦察**，勿盲目当稳定底座 |
| 0.897 | Yaroslav — V4 UNet ILP Reproduction | 本地已有 `v4-unet-ilp-reproduction` |
| 0.897 | Yusuke — LB897 BaseLine；khj1222 — yusuke-lb893-fork | 后者 Kaggle 现标 0.897（本地历史对照 ~0.893） |
| 0.897 | Pilkwang — Blend Preprocessings（DeepCenter） | 本地已有；默认 **min7** |
| 0.897 | 暗黑AGI — LB0.897 Visual Pipeline；KAIWALYA — Competition Solution | 0.897 fork 链；**P0 对齐权重首选 Fork 对象** |
| 0.897 | Tamerlan / Yaroslav — Min7 Short-Track Filter | min7 平台对照 |
| 0.896 | Kun Zhang — Exp039 Min8 Short Track | min8 **略低于** 0.897 平台 |
| **0.894** | Pilkwang — Learned Graph w Gap Recovery；Victor — context-gap / learned-flow / relink-slot；OzanM；Devin Exp041 | **gap/relink 小幅突破 0.897 的公开信号**（仍 <0.90） |
| 0.893 | Nikita Biohub 00；Kun Zhang Exp033 Yusuke Score Push | safe_div 类微调 |
| **0.892** | Kun Zhang — **Exp048 Tamerlan Gap Recovery** | **已验证掉分**（相对 0.897） |
| 0.889 | khj vel0-7-09；Arthurs — lb897-calib；Exp049 Vel0 Min6 | 与当前 **同分平台**；calib/vel 微调非主因 |
| 0.887 | Sireesh — Appearance Crossover；Krizsó — CellTrack Panther | 替代 ILP/检测栈，低于平台 |
| 0.885 | Victor — gauss-refine / dimquality 系列 | refine 方向多数 **掉分** |

**本地 8 本 vs 社区缺口（建议下载/Fork）：**

| 缺口 | 原因 |
|------|------|
| **neilan `solution` (0.900)** | 唯一公开 >0.897 的 notebook |
| **Victor 0.894 三连**（context-gap-close / learned-flow-seed / relink-division-slot） | 0.897 后最可信的 +0.001~+0.003 公开变体 |
| **Kun Zhang Exp037–051 系列** | 系统化 A/B 日志（含 Exp048 负例） |
| **暗黑AGI LB0.897 Visual Pipeline** | P0 Fork 对齐 sha/行数 |
| **Yusuke LB897 BaseLine（Kaggle 原版）** | 与本地 v4/059 对照 |

**未在 Code 排序首页见到、文档曾提及：** Exp046c PathFix — 可能未出分 ≥0.897 或命名不同；**移植前必须在 Code 页确认 public score**。

### 8.3 Discussion 区可操作线索

| 帖子 / 作者 | 内容摘要 | 可操作？ |
|-------------|----------|----------|
| **ISAKA — Rule-based is surprisingly strong?** | 号称 **无学习**、曾 **金区 ~7th**；DoG 尺度 + 微米级连边 + gap closing | **并行调研**；与 0.897 线正交；复现成本高 |
| **hengck23 — Your Affinity Field Tells Your Fate** | 亲和场决定连边/命运；Rahul 有互动 | 读帖 + 对照 MPM/GNN 论文；**无现成 notebook** |
| **Harshul — LAP 求助** | 帧间欧氏 LAP，本地 CV ~**0.59** | **勿作主提交**；仅理解 baseline 差距 |
| **LeeWhieldon — train/test 重叠疑虑** | 未检索到正文细节 | **勿赌泄露**；不纳入提分策略 |
| **Rahul（LB #1 0.910）** | 在 hengck23 可视化帖下活跃；**方法未开源** | 观察即可；头部差距 ≈ **+0.021** vs 0.889 |

### 8.4 分阶段方案矩阵（含是否依赖 0.897）

| 阶段 | 方案 | 来源 | 预期收益 | 风险 | 依赖 0.897？ |
|------|------|------|----------|------|--------------|
| **立即可试** | P0：Fork 0.897 notebook，钉死 sha `12b5d32a` / 行数 ~251k | Code 0.897 集群 | **0.889→~0.897** | 低 | **否**（这是前提） |
| **立即可试** | Fork **neilan solution** 对照 sha/行数/分数 | Code **0.900** | 若可复现：**+0.003~+0.010** | 高：可能私有 data / 不可复现 | 否 |
| **立即可试** | 读 **ISAKA / hengck23** 讨论帖（不占提交） | Discussion | 长期路线情报 | 低 | 否 |
| **中期** | min6 + **DeepCenter**（Blend 改 min7→6） | Pilkwang Blend 0.897 | +0~+0.002 | 错权重已见 **0.881** | **是** |
| **中期** | 移植 **Victor 0.894 relink/gap** 单模块 | Code 0.894 系列 | +0.001~+0.003 | 假连接；Exp048 已证 gap 可掉 | **是** |
| **中期** | **learned-graph + gap**（Pilkwang 0.894） | Code | +0~+0.002 vs 0.897 | det=0.97 易 FP（本地 284k） | **是** |
| **中期** | 本地 **py-ctcmetrics TRA/SEG** 诊断 train | CTC 工具链 | 指导 safe_div/gap 阈值 | 需标注格式转换 | 建议 P0 后 |
| **长期** | **自训 UNet+edge / 新 snapshot** | 论文 + 自研 | 冲 **0.90+** | 算力、12h、过拟合 public | 可并行 |
| **长期** | **多 checkpoint ensemble** | 头部推测 | +0.002~+0.010 | 时长/融合 | 需 ≥2 个强权重 |
| **长期** | **Affinity field 边模型** | hengck23 + MPM/GNN | 冲 0.90+ | 研发量大 | 否 |
| **长期** | **Ultrack 多假设分割** | Nature Methods 2025 | 科研向 | Kaggle 限时/算力 | 否 |
| **停做** | Exp048 gap / 059 dataset-mintrack / calib / min8 / gauss-refine / det0.97 | Code 实测 | 相对 0.897 **≤0** | 已验证 | — |

### 8.5 推荐下一步 5 个动作（2026-07-10）

1. **【提交】P0 only**：Fork **KAIWALYA `Biohub Competition Solution`** 或 **暗黑AGI LB0.897** → Run All → 核对 `Weight sha256: 12b5d32a…`、**≥248k 行**、public **>0.889**。不叠任何后处理。
2. **【侦察，不提交】Fork `neilan/solution`**：对比 Input datasets（+4）、权重来源、行数；判断 0.900 是否可复现到自己账号。
3. **【下载】Victor 三支 0.894 notebook** + **Kun Zhang Exp048**（负例对照）→ diff 相对 min6 底座，只挑 **relink-division-slot** 或 **context-gap-close** 一个模块。
4. **【阅读，不占 GPU】** [Ultrack Nat Methods 2025](https://www.nature.com/articles/s41592-025-02778-0) §方法图（多假设+ILP 选型）+ **ISAKA / hengck23** Discussion 全文 → 决定是否开 rule-based / affinity 第二条线。
5. **【P0 成功后第 1 刀】** min6 + DeepCenter（Blend 一行改 min7→6）；**不要**在 243k 底座上再试。
