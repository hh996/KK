## 比赛标题和任务目标：
全称：Predicting Irrigation Need（预测灌溉需求）

任务：使用表格数据（tabular data），预测每个样本的 Irrigation_Need 属于哪一类：Low（低）、Medium（中）、High（高）。

这是一个三分类问题（multi-class classification）。

## Evaluation Metric（评估指标） —— 必须重点看清楚：
Balanced Accuracy（平衡准确率）
它会分别计算每个类别的 Recall（召回率），然后取平均值。
优点：能有效处理类别不平衡的问题，不会让模型只顾着预测多数类。

## 项目目录（Predicting_Irrigation_Need）

- `data/`：竞赛 CSV（本地放置 `train.csv`、`test.csv`、`sample_submission.csv`）
- `scripts/`：训练与生成提交的 Python 脚本（从仓库根或本目录运行均可；默认读 `data/`、写 `outputs/submissions/`）
- `notebooks/`：EDA 与分析笔记本（如 `pin.ipynb`）
- `outputs/submissions/`：脚本默认提交文件输出目录
- `outputs/reports/`：EDA 等 HTML 报告（如 ydata-profiling 导出）
- `docs/`：说明文档