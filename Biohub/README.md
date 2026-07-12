# Biohub — Cell Tracking During Development

Kaggle 比赛：[biohub-cell-tracking-during-development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)

## 任务概要

在斑马鱼胚胎 3D+时间 光片显微镜影像中，完成细胞检测、跨帧跟踪与谱系重建（含细胞分裂）。

## 目录结构

```
Biohub/
├── data/          # 竞赛数据（train.csv / test.csv / 影像，勿提交大文件）
├── docs/          # 比赛笔记、指标说明、实验记录
└── notebooks/     # 探索与 baseline notebook
```

## 快速开始

1. **加入比赛并配置 Kaggle CLI**
   ```bash
   kaggle auth login
   kaggle competitions join -c biohub-cell-tracking-during-development
   ```

2. **下载数据**（约 88 GB，CC0 许可）
   ```bash
   kaggle competitions download -c biohub-cell-tracking-during-development -p data/
   ```

3. **官方 baseline**（Kaggle Code 页）
   - `Cell Tracking Getting Started w/ Nearest Neighbor`（最近邻跟踪入门，公开分约 0.14）

4. **推荐工具链**
   - [Ultrack](https://royerlab.github.io/ultrack/) — Biohub 官方跟踪管线
   - [tracksdata](https://royerlab.github.io/tracksdata/) — 图优化跟踪与 CTC 指标评估
   - `py-ctcmetrics` — 本地验证 TRA / SEG / DET 等

## 提交约束（Code Competition）

- 通过 Notebook 提交，运行时间 ≤ 12 小时
- 提交时关闭外网（internet off）
- 输出格式以 `sample_submission.csv` 为准

## 时间节点

| 事项 | 日期 |
|------|------|
| 比赛上线 | 2026-06-29 |
| 报名截止 | 2026-09-22 |
| 最终提交 | 2026-09-29 |

奖金池 $60,000，共 7 名获奖者（冠军 $18,000）。

## 与 ROGII 的关系

`ROGII/` 为另一场 Kaggle 比赛（井筒地质预测，表格/树模型），与本项目任务类型不同。可复用的是 Kaggle 环境检测、路径约定、smoke test 等工程模式，而非建模代码。
