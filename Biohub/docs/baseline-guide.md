# Biohub Cell Tracking — 首次提交 Baseline 指南

> 基于本地 6 个 Kaggle notebook 的实际代码整理。比赛：`biohub-cell-tracking-during-development`

---

## 1. 推荐首次提交的 Notebook

**首选：`biohub-competition-solution.ipynb`**

| 维度 | 说明 |
|------|------|
| 可读性 | 仅 10 个 cell，结构最简 |
| 稳定性 | 与 LB0.897 同 lineage，preset 明确 |
| 分数 | min7 短轨迹过滤，248,558 行，edges/node ≈ 0.967 |
| 复杂度 | 单 artifact，无额外 DeepCenter 依赖 |

**次选（带安全补丁）：`biohub-cell-tracking-v4-unet-ilp-reproduction.ipynb`**

- Fork 自 `yusuketogashi/lb897-baseline`
- 额外在写 CSV 时将坐标 clamp 为非负整数
- 输出与 competition-solution 相同（248,558 行）

**冲分备选：`lb897-baseline.ipynb`（059 preset）**

- 全局 min7 + 对 `6bba_05b6850b` 恢复 min6
- 248,723 行，略多 165 行 recall restore
- 适合第二次 A/B 提交，不建议第一次就用

---

## 2. 不推荐首次使用

| Notebook | 原因 |
|----------|------|
| `biohub-cell-tracking-learned-graph-w-gap-recovery.ipynb` | 名含 gap-recovery 但 `OUTPUT_GAP2_RECOVERY=0`；实际改的是 `det_threshold=0.97` + node refine，输出 284k 行，FP 风险高 |
| `biohub-cell-tracking-blend-preprocessings.ipynb` | 需第二个 dataset（DeepCenter），多 ~400 行融合逻辑，输出仅多 11 行 |
| `biohub-yusuke-lb893-fork.ipynb` | min6 基线 + safe_div 微调，251k 行，分数低于 min7 路线 |

---

## 3. 必需 Kaggle 资源

### 比赛数据（自动挂载）
```
/kaggle/input/competitions/biohub-cell-tracking-during-development/test
```

### 主 Artifact（必须 Add Data）
```
Dataset: pilkwang/biohub-tracking-support-pack-50ep-v1
Manifest: /kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json
```

内容：
- `repo/` 或 `repo.zip` — 推理代码（含 `scripts/predict_unet_transformer.py`）
- `weights/unet_transformer/split_0/edge_predictor_best.pth`
- `wheels/` — 离线 pip 包（tracksdata, zarr, pyscipopt, geff, ilpy 等）

### 可选
```
pilkwang/pilkwang-public-dataset-for-notebooks-figures  # 封面图
biohub-deepcenter-unet3d-center-prior-v1                # 仅 blend notebook 需要
```

---

## 4. Kaggle 提交 Checklist

### Step 0：准备
- [ ] 加入比赛 [biohub-cell-tracking-during-development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- [ ] Add Data → `pilkwang/biohub-tracking-support-pack-50ep-v1`
- [ ] 确认 Notebook Settings：**Internet OFF**、**GPU ON**（T4 即可）

### Step 1：Fork & Run
- [ ] Fork `biohub-competition-solution`（或上传本地副本）
- [ ] 检查 Settings → Data：competition data + support pack 均已挂载
- [ ] **不要**在 Notebook Dependencies 里填 pip 包（用 artifact 离线 wheels）
- [ ] Run All（预计 ~10–11 分钟）

### Step 2：验证输出
- [ ] 日志出现 `Required graph/Zarr/ILP packages import successfully.`
- [ ] 推理：`Prediction completed in ~6.7 minutes`，`device=cuda`
- [ ] 输出：`Wrote /kaggle/working/submission.csv with ~248,558 rows`
- [ ] Run Summary：`Edges per node: ~0.967`，`short_track_nodes_removed: ~7,465`

### Step 3：Submit
- [ ] Save Version → Save & Run All（Commit）
- [ ] 等运行完成后 Submit to Competition
- [ ] 记录 `experiment_tag`（如 `056_lb897_high_upside_min7_short_track`）便于对比

### Step 4：后续迭代（可选）
- [ ] 改 preset cell 顶部 `os.environ[...]` 做 A/B
- [ ] 或换 `lb897-baseline.ipynb` 试 dataset-specific min track recall

---

## 5. 本地无法跑通 → 直接在 Kaggle Run

本地缺 GPU / wheels / zarr 依赖时，**直接在 Kaggle 跑是最短路径**：

1. 上传 notebook 到 Kaggle（或 Fork 原 public notebook）
2. 只挂载 support pack dataset，不改代码
3. 首次提交**不要改任何超参**，先拿 baseline 分数
4. 本地仅用于阅读 preset cell 和 post-processing 逻辑

本地 smoke test（可选）：
```bash
export BIOHUB_SLICE=":1"                          # 只跑 1 个 dataset
export BIOHUB_ALLOW_ARTIFACT_FALLBACK=1           # 允许旧 artifact
export BIOHUB_ALLOW_PIP_INSTALL=1                 # 允许联网装包
export BIOHUB_MODEL_ARTIFACTS=/path/to/support-pack
```

---

## 6. Pipeline 速查

```
Zarr 3D 体数据
  → UNet 检测场 p_t(r) = σ(h_θ(V_{t:t+1}))
  → 局部极大值 + det_threshold=0.99
  → Node-Transformer 边打分 s_ij
  → ILP 选图（edge=-1, appear/disappear=0.1, division=1.0）
  → 后处理链：
      motion relink (6/10 μm)
    → gap close 1-frame (6 μm)
    → safe divisions
    → short track filter (min7)
    → linefit smooth (w=0.8, win=2)
  → submission.csv
```

物理距离公式（所有后处理共用）：
```
d_μm = √((1.625·Δz)² + (0.40625·Δy)² + (0.40625·Δx)²)
```

---

## 7. 六个 Notebook 差异一览

| Notebook | Preset | 核心 tweak | 输出行数 |
|----------|--------|-----------|---------|
| biohub-competition-solution | min7 short track | 基线 min7 过滤 | 248,558 |
| lb897-baseline | dataset_mintrack_recall | min7 + 6bba 用 min6 | 248,723 |
| v4-unet-ilp-reproduction | min7 + coord clamp | CSV 坐标非负 | 248,558 |
| yusuke-lb893-fork | safe_div_micro | min6 + 收紧 safe_div | 251,611 |
| learned-graph-w-gap-recovery | det 0.97 + node refine | 低阈值检测 | 284,162 |
| blend-preprocessings | DeepCenter fusion | 中心先验补点 | 248,569 |

---

## 8. 常见问题

**Q: Import 失败 / 找不到 artifact？**
- 确认已 Add `biohub-tracking-support-pack-50ep-v1`
- Notebook Dependencies 留空，依赖 artifact 内 wheels

**Q: Internet off 下 pip 报错？**
- 默认 `ALLOW_PIP_INSTALL=false`，不应联网
- 检查 artifact 的 `wheels/` 目录是否存在

**Q: 运行超时？**
- 全量 test 约 10 min（GPU T4）
- 不要开 `RUN_VISUAL_EDA=0` 可略省时间，但影响诊断

**Q: 第一次改什么最可能涨分？**
- 按 lineage：min7 → `lb897-baseline` 的 dataset-specific recall
- 不建议先动 det_threshold 或开 gap2
