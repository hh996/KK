"""
Orbit Wars - 改进版启发式智能体 v2
策略：综合考虑产能、距离、驻军的目标评分 + 保留防守兵力 + 轨道预测瞄准
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, ROTATION_RADIUS_LIMIT


def agent(obs):
    moves = []

    # 解析观测数据
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    angular_velocity = obs.get("angular_velocity", 0.03) if isinstance(obs, dict) else obs.angular_velocity
    initial_planets = obs.get("initial_planets", []) if isinstance(obs, dict) else obs.initial_planets
    comets = obs.get("comets", []) if isinstance(obs, dict) else obs.comets
    comet_planet_ids = set(obs.get("comet_planet_ids", [])) if isinstance(obs, dict) else set(obs.comet_planet_ids)

    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not my_planets or not targets:
        return moves

    # 构建初始位置映射
    initial_pos = {p[0]: (p[2], p[3]) for p in initial_planets} if initial_planets else {}

    # 构建彗星轨迹映射
    comet_paths = {}
    for comet_group in comets:
        for pid, path in zip(comet_group.get("planet_ids", []), comet_group.get("paths", [])):
            comet_paths[pid] = path

    # 计算每个目标的价值分数
    def evaluate_target(target):
        min_dist = float('inf')
        for my_p in my_planets:
            dist = math.hypot(my_p.x - target.x, my_p.y - target.y)
            min_dist = min(min_dist, dist)

        production_value = target.production ** 1.5
        distance_penalty = max(1, min_dist)
        defense_difficulty = max(1, target.ships)
        score = production_value / (distance_penalty * defense_difficulty)

        if target.production >= 4:
            score *= 1.3
        if min_dist < 20:
            score *= 1.2
        return score

    # 预测位置函数
    def predict_position(planet, time_steps):
        if planet.id in comet_planet_ids and planet.id in comet_paths:
            path = comet_paths[planet.id]
            future_index = int(time_steps)
            if future_index < len(path):
                return path[future_index]
            return path[-1] if path else (planet.x, planet.y)

        if planet.id in initial_pos:
            init_x, init_y = initial_pos[planet.id]
            center_x, center_y = 50, 50
            orbital_radius = math.hypot(init_x - center_x, init_y - center_y)
            if orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT:
                init_angle = math.atan2(init_y - center_y, init_x - center_x)
                future_angle = init_angle + angular_velocity * time_steps
                return (center_x + orbital_radius * math.cos(future_angle),
                        center_y + orbital_radius * math.sin(future_angle))
        return planet.x, planet.y

    # 为目标评分并排序
    target_scores = [(t, evaluate_target(t)) for t in targets]
    target_scores.sort(key=lambda x: x[1], reverse=True)

    # 计算可用兵力（保留防守）
    available = {}
    for p in my_planets:
        reserve = min(10, int(p.ships * 0.2))
        if p.ships > reserve:
            available[p.id] = {'planet': p, 'ships': p.ships - reserve}

    # 分配攻击
    for target, score in target_scores:
        if score <= 0:
            continue

        ships_needed = target.ships + 1
        ships_assigned = 0

        # 按距离排序攻击者
        attackers = []
        for pid, info in available.items():
            if info['ships'] <= 0:
                continue
            my_p = info['planet']
            dist = math.hypot(my_p.x - target.x, my_p.y - target.y)
            attackers.append((pid, dist, info))

        attackers.sort(key=lambda x: x[1])

        for pid, dist, info in attackers:
            if ships_assigned >= ships_needed:
                break

            remaining = ships_needed - ships_assigned
            to_send = min(info['ships'], remaining)

            if to_send >= 3:
                my_p = info['planet']

                # 预测目标位置并计算角度
                estimated_speed = 1.0 + 5.0 * (math.log(to_send) / math.log(1000)) ** 1.5
                travel_time = dist / max(1, estimated_speed)
                target_x, target_y = predict_position(target, travel_time)
                angle = math.atan2(target_y - my_p.y, target_x - my_p.x)

                moves.append([my_p.id, angle, int(to_send)])
                info['ships'] -= to_send
                ships_assigned += to_send

    return moves
