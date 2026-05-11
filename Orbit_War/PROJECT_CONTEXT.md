# Orbit_War 目录说明（背景 / 状态 / 已知问题 / 后续注意）

本文档用于在对话上下文满或换人接手时快速对齐。  
竞赛与规则细节以 [data/README.md](data/README.md) 与 Kaggle `kaggle_environments` 中 `orbit_wars` 源码为准。

---

## 1. 背景

- **比赛**：Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)——在 100×100 连续棋盘上，多玩家通过从行星发射舰队争夺中立/敌方行星；支持 **2 人或 4 人**；单回合可返回 **多条**动作 `[[planet_id, angle, ships], ...]`；终局常为 **500 步**或仅剩一方有行星/舰队；官方环境对胜者的 `reward` 为 **+1**，非最高总船数的一方为 **-1**（并列最高分可多人 +1），与 README 中“总船数最高获胜”一致。
- **本仓库角色**：存放规则说明、`v1_rule` 下多种启发式单文件 Agent、以及 **NN+MCTS 试验管线**（`scripts/v3_NN`）。`data/`、`docs/overview.md` 为入门与本地测试说明。

---

## 2. 各文件夹在做什么

| 路径 | 作用 |
|------|------|
| [data/](data/) | 比赛说明摘要、`agents.md`、示例 `main.py` |
| [docs/](docs/) | 结构化概览（如 `overview.md`） |
| [scripts/v1_rule/](scripts/v1_rule/) | 纯规则/启发式 Agent：`v1_deepseek`、`v1_opus` 等；物理与打分逻辑常被 `v3_NN` 参考 |
| [scripts/v2_MCTS/](scripts/v2_MCTS/) | 另一套 MCTS 相关试验（notebook + deepseek 变体） |
| [scripts/v3_NN/](scripts/v3_NN/) | **当前主训练的 NN+MCTS**：ResNet式编码、`PolicyValueNetwork`（src/tgt/ship/value）、`physics.WorldState` 本地推演、自我对弈写 replay、见下节 |

---

## 3. `scripts/v3_NN` 当前在做什么（实现要点）

- **训练**：`train.py` / `run.py` → `Trainer`：`kaggle_environments.make("orbit_wars")` 自博弈，按 `TRAIN_TWO_PLAYER_PROB` 混入 **2p / 4p**；每步用 MCTS（`mcts.py`）在 **宏动作**（最多 `MAX_FLEETS_PER_TURN` 支舰队有序组合）上搜索；样本标签：**终局 `reward`（±1）** + MCTS visit 转化的 **src/tgt/ship** 边际监督。
- **本地物理**：`physics.py` 的 `WorldState.step` 已对官方步序做了一轮对齐：**回合开始彗星过期 → 发射 → 产能 → 行星路径（公转 + 彗星 path_index）→ 舰队移动（swept_pair_hit）→ 落地 → 再过期彗星 → 战斗**。发射点外推 **0.1**、`episode_seed` 缺失时彗星 RNG 与官方同为 **0**；`shipSpeed` / `sunRadius` / `boardSize` 从 env `configuration` 经 `build_world_from_obs` 传入 `WorldState`。
- **价值辅助**：[value_util.py](scripts/v3_NN/value_util.py) `env_terminal_value` 与环境“最高分且 max>0 → +1”一致；树上叶子仍会混合启发式船差（见 [mcts.py](scripts/v3_NN/mcts.py) `_terminal_value`）。
- **评估**：[eval_orbit_wars.py](scripts/v3_NN/eval_orbit_wars.py) 快速 vs 内置 `random_agent`（训练循环按 `EVAL_EVERY_ITERS` 尝试调用）。
- **配置**：[config.py](scripts/v3_NN/config.py)（模拟次数、并行局数、`OPPONENT_SAMPLES`、通道数 15、`SHIP_BUCKET_COUNT` 等）。
- **烟囱测试**：[smoke_checks.py](scripts/v3_NN/smoke_checks.py)。

依赖见仓库根目录 [requirements.txt](../requirements.txt)（含 `torch`、`kaggle-environments` 等）。

---

## 4. 当前已知问题 / 局限性

1. **彗星 / 配置对齐**：彗星路径 RNG 已与官方 ``env_info.get('seed',0) or 0`` 及同字符串种子对齐；若 `build_world_from_obs` 未传入与 env 一致的 `configuration`（如自定义 `boardSize`），仍可能与真环境分叉。神经网络特征图仍为固定 **100×100**，与可配置 `boardSize` 并用时需另做缩放或改编码。
2. **同时行动**：树里仍为「我方 macro + 各对手按策略 **采样** macro」一步推进；**不是**严格的联合动作空间搜索或纳什均衡求解。
3. **宏动作空间有限**：合法组合有上限/cap，与高手的「一单回合多舰队」全流程仍有差距。
4. **checkpoint 兼容性**：改过网络输入通道（15）与四路输出后，**旧 checkpoint 不可直接加载**。
5. **评估覆盖面**：`eval_orbit_wars` 主要为 **2p vs random**；4p、`v1_deepseek` 对打、超时预算（`actTimeout`）Stress 仍需补。
6. **训练效率**：默认 `MCTS_SIMULATIONS` 较高时每步很慢；单机需酌情降模拟数或小网络冒烟。

---

## 5. 后续需要注意哪些地方

| 优先级 | 事项 |
|--------|------|
| 高 | 强对齐线上：已支持从 `configuration` 读 `shipSpeed`/`sunRadius`/`boardSize` 与 `cometSpeed` 一并进 `WorldState`；若仍分叉，可做 **单步 env vs `WorldState.step` 对拍测试** 或与 env 共用浅层 rollout。 |
| 高 | **价值备份语义**：`_backup` 当前为沿路同号累加；若加强理论性，需明确「对谁的最优响应值」并与 `encode_state(perspective_player=…)` 统一，必要时每层取负（双人零和规范）。 |
| 中 | **STOP / 可变长度**：宏动作可考虑显式 STOP 或提高 K + progressive widening。 |
| 中 | **对手池**：除 `latest` 外混入历史权重或 `v1_deepseek` 作为 teacher/对手。 |
| 中 | **特征**：可补 `remainingOverageTime`、更细的 4p 槽位或实体编码（GNN/Transformer）以冲上限。 |
| 低 | `NUM_PARALLEL_GAMES` 实为顺序多局，真并行需多进程；replay 全量存 checkpoint 可能占盘，可改为截断或只存模型。 |
| 低 | 提交赛方须 **单文件或规定打包** `main.py`；训练代码中的 `train.py` 不能直接当提交，需导出轻量 `agent(obs)`。 |

---

## 6. 关键文件速查

- 规则汇总：[data/README.md](data/README.md)
- v3 训练入口：[scripts/v3_NN/train.py](scripts/v3_NN/train.py)、[run.py](scripts/v3_NN/run.py)
- MCTS：[scripts/v3_NN/mcts.py](scripts/v3_NN/mcts.py)
- 本地动力学：[scripts/v3_NN/physics.py](scripts/v3_NN/physics.py)
- 网络：[scripts/v3_NN/network.py](scripts/v3_NN/network.py)
- 特征：[scripts/v3_NN/features.py](scripts/v3_NN/features.py)
- 启发式参考：[scripts/v1_rule/v1_deepseek/main.py](scripts/v1_rule/v1_deepseek/main.py)

---

*最后更新：由开发笔记整理；若实现变更，请同步修改本节「当前已知问题」与「后续注意」。*
