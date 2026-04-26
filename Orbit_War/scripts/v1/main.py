"""
Orbit Wars - 改进版启发式智能体 v1

策略改进：
1. 目标评分：综合考虑产能、距离、驻军
2. 兵力分配：保留防守兵力，剩余按价值分配
3. 轨道预测：瞄准移动行星的未来位置
4. 多源合击：协调多个行星攻击同一高价值目标
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT


def agent(obs):
    """主入口函数"""
    moves = []

    # 解析观测数据
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
    angular_velocity = obs.get("angular_velocity", 0.03) if isinstance(obs, dict) else obs.angular_velocity
    initial_planets = obs.get("initial_planets", []) if isinstance(obs, dict) else obs.initial_planets
    comets = obs.get("comets", []) if isinstance(obs, dict) else obs.comets
    comet_planet_ids = set(obs.get("comet_planet_ids", [])) if isinstance(obs, dict) else set(obs.comet_planet_ids)

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]

    # 分类行星
    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner not in [-1, player]]
    neutral_planets = [p for p in planets if p.owner == -1]

    if not my_planets:
        return moves

    # 构建初始位置映射（用于轨道预测）
    initial_pos = {p[0]: (p[2], p[3]) for p in initial_planets} if initial_planets else {}

    # 构建彗星轨迹映射
    comet_paths = {}
    for comet_group in comets:
        for pid, path in zip(comet_group.get("planet_ids", []), comet_group.get("paths", [])):
            comet_paths[pid] = path

    # 计算己方总兵力
    my_total_ships = sum(p.ships for p in my_planets) + sum(f.ships for f in fleets if f.owner == player)

    # 评估所有目标的价值
    targets = enemy_planets + neutral_planets
    if not targets:
        return moves

    target_scores = []
    for target in targets:
        score = evaluate_target(target, my_planets, planets, angular_velocity,
                               initial_pos, comet_paths, comet_planet_ids)
        target_scores.append((target, score))

    # 按价值排序
    target_scores.sort(key=lambda x: x[1], reverse=True)

    # 分配兵力
    moves = allocate_forces(my_planets, target_scores, angular_velocity, initial_pos,
                           comet_paths, comet_planet_ids, my_total_ships)

    return moves


def evaluate_target(target, my_planets, all_planets, angular_velocity, initial_pos, comet_paths, comet_planet_ids):
    """
    评估目标价值
    分数 = 产能^1.5 / (距离 * 驻军)
    距离取最近己方行星的距离
    """
    if not my_planets:
        return 0

    # 找到最近的己方行星（考虑轨道预测）
    min_distance = float('inf')
    for my_p in my_planets:
        dist = predict_distance(my_p, target, angular_velocity, initial_pos, comet_paths, comet_planet_ids)
        min_distance = min(min_distance, dist)

    # 基础价值
    production_value = target.production ** 1.5
    distance_penalty = max(1, min_distance)
    defense_difficulty = max(1, target.ships)

    score = production_value / (distance_penalty * defense_difficulty)

    # 额外加分：高产能行星
    if target.production >= 4:
        score *= 1.3

    # 额外加分：距离近的行星（快速扩张）
    if min_distance < 20:
        score *= 1.2

    return score


def predict_distance(from_planet, to_planet, angular_velocity, initial_pos, comet_paths, comet_planet_ids):
    """预测两行星之间的飞行距离（考虑轨道运动）"""
    # 获取当前位置
    x1, y1 = from_planet.x, from_planet.y
    x2, y2 = to_planet.x, to_planet.y

    # 粗略估计飞行时间
    dist = math.hypot(x2 - x1, y2 - y1)

    # 估计舰队速度（假设中等规模舰队）
    estimated_fleet_size = 20
    speed = 1.0 + (6.0 - 1.0) * (math.log(estimated_fleet_size) / math.log(1000)) ** 1.5
    travel_time = dist / max(1, speed)

    # 预测目标行星未来位置
    future_x2, future_y2 = predict_position(to_planet, travel_time, angular_velocity,
                                           initial_pos, comet_paths, comet_planet_ids)

    # 重新计算距离
    final_dist = math.hypot(future_x2 - x1, future_y2 - y1)
    return final_dist


def predict_position(planet, time_steps, angular_velocity, initial_pos, comet_paths, comet_planet_ids):
    """预测行星在未来某时刻的位置"""
    # 如果是彗星，使用预设路径
    if planet.id in comet_planet_ids and planet.id in comet_paths:
        path = comet_paths[planet.id]
        # 假设每回合移动一个路径点（简化）
        future_index = int(time_steps)
        if future_index < len(path):
            return path[future_index]
        else:
            return path[-1] if path else (planet.x, planet.y)

    # 如果是轨道行星，计算旋转后的位置
    if planet.id in initial_pos:
        init_x, init_y = initial_pos[planet.id]
        center_x, center_y = 50, 50
        orbital_radius = math.hypot(init_x - center_x, init_y - center_y)

        # 检查是否在内圈（会旋转）
        if orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT:
            init_angle = math.atan2(init_y - center_y, init_x - center_x)
            future_angle = init_angle + angular_velocity * time_steps
            future_x = center_x + orbital_radius * math.cos(future_angle)
            future_y = center_y + orbital_radius * math.sin(future_angle)
            return future_x, future_y

    # 静态行星或无法预测，返回当前位置
    return planet.x, planet.y


def allocate_forces(my_planets, target_scores, angular_velocity, initial_pos,
                   comet_paths, comet_planet_ids, my_total_ships):
    """分配兵力攻击目标"""
    moves = []

    # 计算每个己方行星的可用兵力（保留防守兵力）
    available_forces = {}
    for p in my_planets:
        # 保留策略：至少留 min(10, 20%兵力)，至少留1艘
        reserve = min(10, int(p.ships * 0.2))
        available = max(0, p.ships - reserve)
        if available > 0:
            available_forces[p.id] = {
                'planet': p,
                'available': available,
                'reserve': reserve
            }

    if not available_forces or not target_scores:
        return moves

    # 为每个目标分配攻击者
    assigned_attacks = {}  # target_id -> list of (from_planet, ships)

    for target, score in target_scores:
        if score <= 0:
            continue

        target_id = target.id
        assigned_attacks[target_id] = []
        total_ships_assigned = 0

        # 找到可以攻击这个目标的所有己方行星
        attackers = []
        for pid, force_info in available_forces.items():
            my_p = force_info['planet']
            dist = predict_distance(my_p, target, angular_velocity, initial_pos,
                                   comet_paths, comet_planet_ids)
            attackers.append((my_p, dist, force_info['available']))

        # 按距离排序
        attackers.sort(key=lambda x: x[1])

        # 计算需要多少兵力才能占领
        ships_needed = target.ships + 1

        # 优先从近的行星派兵
        for my_p, dist, available in attackers:
            if total_ships_assigned >= ships_needed:
                break

            if available <= 0:
                continue

            # 计算这次派多少兵
            remaining_needed = ships_needed - total_ships_assigned
            to_send = min(available, remaining_needed)

            # 至少派3艘（避免太小的舰队）
            if to_send >= 3:
                assigned_attacks[target_id].append((my_p, to_send))
                total_ships_assigned += to_send
                # 更新可用兵力
                available_forces[my_p.id]['available'] -= to_send

    # 生成移动指令
    for target_id, attackers in assigned_attacks.items():
        if not attackers:
            continue

        # 找到目标行星
        target = None
        for t, _ in target_scores:
            if t.id == target_id:
                target = t
                break

        if not target:
            continue

        # 为每个攻击者计算发射角度
        for my_p, ships_to_send in attackers:
            # 预测目标位置
            dist = math.hypot(target.x - my_p.x, target.y - my_p.y)
            estimated_speed = 1.0 + (6.0 - 1.0) * (math.log(ships_to_send) / math.log(1000)) ** 1.5
            travel_time = dist / max(1, estimated_speed)

            target_x, target_y = predict_position(target, travel_time, angular_velocity,
                                                initial_pos, comet_paths, comet_planet_ids)

            # 计算角度
            angle = math.atan2(target_y - my_p.y, target_x - my_p.x)

            moves.append([my_p.id, angle, int(ships_to_send)])

    return moves
