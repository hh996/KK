"""
Orbit Wars - 最近行星狙击智能体

一个简单的智能体：当舰船数足以确保占领时，优先夺取最近的非己方行星。

策略：
  对每颗己方行星，找到最近的非己方行星。
  若己方舰船数大于目标驻军，则只派出刚好可占领的数量
  （目标驻军 + 1）。否则等待并继续积累。

演示的关键概念：
  - 解析观测数据（行星、玩家 ID）
  - 使用 atan2 计算舰队方向角
  - 以 [from_planet_id, angle, num_ships] 格式返回动作
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

    # 解析为命名元组，便于可读字段访问：
    #   Planet(id, owner, x, y, radius, ships, production)
    #   owner == -1 表示中立，0-3 表示玩家 ID
    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # 找到最近的非己方行星
        nearest = None
        min_dist = float("inf")
        for t in targets:
            dist = math.sqrt((mine.x - t.x) ** 2 + (mine.y - t.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        # 占领目标需要派出比其驻军更多的舰船。
        # 精确发送 target_ships + 1 可确保占领。
        ships_needed = nearest.ships + 1

        # 只有在可承担时才发射，否则继续攒兵
        if mine.ships >= ships_needed:
            # atan2(dy, dx) 给出从我方行星指向目标的角度
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves
