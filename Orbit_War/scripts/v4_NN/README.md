# v4_NN — 路线 B（Hybrid MCTS + macro-index）

基于 v1 原子逻辑重组的 `generate_candidates_b`（~20 宏）+ macro-index 策略网络 + MCTS。

## 评估

每 **50 iter**（`EVAL_EVERY_ITERS`）自动并行评估 vs random / deepseek（2P+4P）/ starter（2P），结果写入 `logs/train_log.csv`。  
用 `log_watcher.ipynb` 可视化 loss 与胜率；也可手动跑 `eval_orbit_wars.py` 做额外抽检。

### 一键评估（默认 `checkpoints/latest_b.pt`，2P 各 8 局）

```bash
conda activate KK
cd Orbit_War/scripts/v4_NN
python eval_orbit_wars.py
```

输出示例：`vs random 2p: 6/8`、`vs starter 2p: …`、`vs deepseek 2p: …`  
评估设定：net + `candidates_b` + MCTS（48 sim），低置信时用 ROI fallback。

### 指定 checkpoint / 局数 / 4P

```bash
cd Orbit_War/scripts/v4_NN
python -c "
from eval_orbit_wars import load_net_from_checkpoint, evaluate_net_vs_deepseek, evaluate_net_vs_random

net = load_net_from_checkpoint('checkpoints/iter_100_b.pt')  # 或 latest_b.pt

for opp, fn in [('random', evaluate_net_vs_random), ('deepseek', evaluate_net_vs_deepseek)]:
    w2, t2 = fn(net, episodes=30, num_agents=2)
    w4, t4 = fn(net, episodes=15, num_agents=4)
    print(f'vs {opp} 2p: {w2}/{t2}  |  4p: {w4}/{t4}')
"
```

### 菜单覆盖率（IL 质量，不是胜率）

```bash
python coverage_report.py    # 默认 10 局 2P，输出 matched rate
```

### 冒烟（候选数量 + 单步 MCTS）

```bash
python smoke_checks.py
```

### 参考线

| 对手 | 含义 |
|------|------|
| random | 最弱内置 bot |
| starter | Kaggle 入门 bot |
| deepseek | 你的 v1 规则 agent（~800 LB 参考） |

vs deepseek 2P **> 50%**（30 局）时可考虑提交 `agent.py` 测 LB。

## 快速开始

```bash
conda activate KK
cd Orbit_War/scripts/v4_NN
python run.py                     # 训练（Ctrl+C 中断 → 再跑 run.py 续训）
```

### 训练曲线（`logs/train_log.csv`）

在 Jupyter 中打开 `log_watcher.ipynb`，或：

```bash
jupyter notebook log_watcher.ipynb
```

CSV 列：`iteration, loss, value_loss, policy_loss, wins_random_2p, wins_deepseek_2p, wins_random_4p, wins_deepseek_4p, wins_starter_2p`（评估行才有胜场数；2P/4P 默认 30/15 局）。

## 检查点

| 文件 | 说明 |
|------|------|
| `checkpoints/latest_b.pt` | 最新模型 + optimizer |
| `checkpoints/interrupt_b.pt` | Ctrl+C 中断保存 |
| `checkpoints/replay_b.pt` | replay buffer（续训无缝） |
| `checkpoints/pretrained_b.pt` | IL 预训练完成标记 |

## 提交

Kaggle notebook 中 import `agent.agent`；需同目录放置 `latest_b.pt` 或内嵌权重。

低置信度时 fallback：`evaluate`（v1）→ ROI 最高宏。

## 与 v3 区别

- 动作空间：`candidates_b` ~20 宏（非 450 朴素宏）
- Policy：macro-index softmax（非 src/tgt heatmap）
- 可从 `../v3_NN/checkpoints/latest.pt` 迁移 backbone+value
