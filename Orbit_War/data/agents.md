# Orbit Wars：快速上手

本指南将带你完成智能体构建、本地测试，以及提交到 Kaggle 的 Orbit Wars 竞赛。

## 游戏概览

Orbit Wars 是一款实时策略游戏，棋盘为 100x100，中心有一颗太阳。玩家通过在行星之间派遣舰队来占领行星。

- **行星**：每回合产出舰船（与其半径相关）
- **内圈行星**：绕中心太阳旋转；外圈行星保持静止
- **舰队**：从出发行星按指定角度直线飞行
- **舰队速度**：随舰队规模变化（1 艘 = 1/回合，更大舰队最高可达 6/回合）
- **战斗**：到达舰队的舰船数会从行星驻军中扣减；若驻军降到 0 以下则易主
- **太阳**：撞上太阳的舰队会被摧毁
- **彗星**：沿椭圆路径穿越棋盘的临时行星
- **胜利条件**：时间结束时舰船总数（行星 + 舰队）最高，或成为最后存活玩家

## 你的智能体

你的智能体是一个函数：接收观测数据，返回动作列表。

**观测字段：**
- `player` — 你的玩家 ID（0-3）
- `planets` — `[id, owner, x, y, radius, ships, production]` 列表（owner 为 -1 表示中立）
- `fleets` — `[id, owner, x, y, angle, from_planet_id, ships]` 列表
- `angular_velocity` — 内圈行星旋转速度（弧度/回合）

**动作格式：**
每条动作为 `[from_planet_id, angle_in_radians, num_ships]`。

**示例——最近行星狙击：**

```python
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # 找到最近的非己方行星
        nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))

        # 发送刚好够占领的舰船数
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves
```

## 本地测试

安装环境后，可在 Python 或 notebook 中运行对局：

```bash
pip install -e /path/to/kaggle-environments
```

```python
from kaggle_environments import make

env = make("orbit_wars", debug=True)
env.run(["main.py", "random"])

# 查看结果
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# 在 notebook 中渲染
env.render(mode="ipython", width=800, height=600)
```

## 查找竞赛

```bash
kaggle competitions list -s "orbit wars"
```

查看竞赛页面以阅读规则与评测细节：

```bash
kaggle competitions pages orbit-wars
kaggle competitions pages orbit-wars --content
```

## 接受竞赛规则

提交前，你**必须**在 Kaggle 网站接受规则。访问 `https://www.kaggle.com/competitions/orbit-wars` 并点击 “Join Competition”。

验证你已加入：

```bash
kaggle competitions list --group entered
```

## 下载竞赛数据

```bash
kaggle competitions download orbit-wars -p orbit-wars-data
```

## 提交你的智能体

提交包根目录必须包含带 `agent` 函数的 `main.py`。

**单文件智能体：**

```bash
kaggle competitions submit orbit-wars -f main.py -m "Nearest planet sniper v1"
```

**多文件智能体**——打包为 tar.gz，且 `main.py` 位于根目录：

```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit orbit-wars -f submission.tar.gz -m "Multi-file agent v1"
```

**Notebook 提交：**

```bash
kaggle competitions submit orbit-wars -k YOUR_USERNAME/orbit-wars-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

## 监控提交状态

查看提交状态：

```bash
kaggle competitions submissions orbit-wars
```

记下输出中的 submission ID——后续查询对局需要用到。

## 列出对局（Episodes）

当你的提交跑过一些比赛后：

```bash
kaggle competitions episodes <SUBMISSION_ID>
```

用于脚本处理的 CSV 输出：

```bash
kaggle competitions episodes <SUBMISSION_ID> -v
```

## 下载回放与日志

下载某场对局的回放 JSON（用于可视化或分析）：

```bash
kaggle competitions replay <EPISODE_ID>
kaggle competitions replay <EPISODE_ID> -p ./replays
```

下载智能体日志以调试其行为：

```bash
# 第一个智能体（索引 0）的日志
kaggle competitions logs <EPISODE_ID> 0

# 第二个智能体（索引 1）的日志
kaggle competitions logs <EPISODE_ID> 1 -p ./logs
```

## 查看排行榜

```bash
kaggle competitions leaderboard orbit-wars -s
```

## 典型工作流

```bash
# 本地测试
python -c "
from kaggle_environments import make
env = make('orbit_wars', debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

# 提交
kaggle competitions submit orbit-wars -f main.py -m "v1"

# 查看状态
kaggle competitions submissions orbit-wars

# 查看对局
kaggle competitions episodes <SUBMISSION_ID>

# 下载回放和日志
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0

# 查看排行榜
kaggle competitions leaderboard orbit-wars -s
```

