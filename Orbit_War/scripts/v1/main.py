"""
Orbit Wars - 改进版启发式智能体 v1.1

优化点：
1. 动态防守阈值 - 前期激进，后期保守
2. 真正的多源合击 - 协调多个行星同时攻击
3. 敌方威胁检测 - 检测来袭舰队，提前回防
4. 修复速度公式 - 使用正确的舰队速度计算
5. 产能优先 - 前期优先抢占高产能行星
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, ROTATION_RADIUS_LIMIT


def agent(obs):
    moves = []

    # 解析观测数据
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
    angular_velocity = obs.get("angular_velocity", 0.03) if isinstance(obs, dict) else obs.angular_velocity
    initial_planets = obs.get("initial_planets", []) if isinstance(obs, dict) else obs.initial_planets
    comets = obs.get("comets", []) if isinstance(obs, dict) else obs.comets
    comet_planet_ids = set(obs.get("comet_planet_ids", [])) if isinstance(obs, dict) else set(obs.comet_planet_ids)
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]

    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner not in [-1, player]]
    neutral_planets = [p for p in planets if p.owner == -1]
    enemy_fleets = [f for f in fleets if f.owner != player]

    if not my_planets:
        return moves

    # 游戏进度 (0-500回合)
    game_progress = step / 500.0

    # 构建初始位置映射
    initial_pos = {p[0]: (p[2], p[3]) for p in initial_planets} if initial_planets else {}

    # 构建彗星轨迹映射
    comet_paths = {}
    for comet_group in comets:
        for pid, path in zip(comet_group.get("planet_ids", []), comet_group.get("paths", [])):
            comet_paths[pid] = path

    # 计算舰队速度 (修复公式)
    def fleet_speed(ship_count):
        return 1.0 + 5.0 * (math.log(max(1, ship_count)) / math.log(1000)) ** 1.5

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

    # 检测敌方舰队威胁 - 返回受威胁的行星ID和威胁等级
    def detect_threats():
        threats = {}  # planet_id -> total_enemy_ships_coming
        for my_p in my_planets:
            threats[my_p.id] = 0
            for ef in enemy_fleets:
                # 计算敌方舰队是否朝我方行星移动
                dist_to_planet = math.hypot(ef.x - my_p.x, ef.y - my_p.y)
                if dist_to_planet < 15:  # 近距离威胁
                    # 估算到达时间
                    ef_speed = fleet_speed(ef.ships)
                    time_to_arrival = dist_to_planet / max(0.5, ef_speed)
                    if time_to_arrival < 10:  # 10回合内到达
                        threats[my_p.id] += ef.ships
        return threats

    threats = detect_threats()

    # 动态防守阈值 - 前期激进(保留5%), 后期保守(保留30%)
    def get_reserve_ratio():
        # 根据游戏进度调整: 前期0.05, 后期0.3
        return 0.05 + game_progress * 0.25

    reserve_ratio = get_reserve_ratio()

    # 计算可用兵力（考虑威胁）
    available = {}
    for p in my_planets:
        base_reserve = max(5, int(p.ships * reserve_ratio))
        # 如果有威胁，额外保留兵力
        threat_ships = threats.get(p.id, 0)
        reserve = max(base_reserve, threat_ships + 5)
        available_ships = max(0, p.ships - reserve)
        if available_ships > 0:
            available[p.id] = {
                'planet': p,
                'ships': available_ships,
                'reserve': reserve,
                'threat': threat_ships
            }

    # 目标评分 - 考虑游戏阶段
    def evaluate_target(target):
        min_dist = float('inf')
        closest_planet = None
        for my_p in my_planets:
            dist = math.hypot(my_p.x - target.x, my_p.y - target.y)
            if dist < min_dist:
                min_dist = dist
                closest_planet = my_p

        production_value = target.production ** 1.5
        distance_penalty = max(1, min_dist)
        defense_difficulty = max(1, target.ships)

        score = production_value / (distance_penalty * defense_difficulty)

        # 前期(0-30%)优先产能，后期(70-100%)优先距离
        if game_progress < 0.3:
            if target.production >= 4:
                score *= 2.0  # 前期高产能行星权重更高
            elif target.production >= 3:
                score *= 1.5
        elif game_progress > 0.7:
            if min_dist < 15:
                score *= 1.5  # 后期优先近的目标

        # 中立行星优先于敌方（容易占领）
        if target.owner == -1:
            score *= 1.2

        return score, min_dist, closest_planet

    # 评估所有目标
    targets = enemy_planets + neutral_planets
    if not targets:
        return moves

    target_info = []
    for t in targets:
        score, dist, closest = evaluate_target(t)
        target_info.append((t, score, dist, closest))

    # 按价值排序
    target_info.sort(key=lambda x: x[1], reverse=True)

    # 真正的多源合击 - 为高价值目标协调多个攻击者
    # 策略: 选择前N个高价值目标，为每个目标分配攻击者
    top_targets = target_info[:min(3, len(target_info))]

    for target, score, dist_to_closest, closest_planet in top_targets:
        if score <= 0:
            continue

        ships_needed = target.ships + 1
        ships_assigned = 0

        # 收集所有可以攻击这个目标的我方行星
        attackers = []
        for pid, info in available.items():
            if info['ships'] <= 0:
                continue
            my_p = info['planet']
            # 预测攻击距离
            init_dist = math.hypot(my_p.x - target.x, my_p.y - target.y)
            est_time = init_dist / fleet_speed(20)  # 估算时间
            future_tx, future_ty = predict_position(target, est_time)
            actual_dist = math.hypot(my_p.x - future_tx, my_p.y - future_ty)
            attackers.append((pid, actual_dist, info, future_tx, future_ty))

        # 按距离排序
        attackers.sort(key=lambda x: x[1])

        # 分配攻击者，直到满足所需兵力
        for pid, dist, info, future_tx, future_ty in attackers:
            if ships_assigned >= ships_needed:
                break

            remaining = ships_needed - ships_assigned
            # 根据距离决定是否参与：近的派全部，远的派部分
            if dist < 20:
                to_send = min(info['ships'], remaining)
            else:
                to_send = min(info['ships'] // 2, remaining)

            if to_send >= 3:
                my_p = info['planet']

                # 计算精确的发射角度（瞄准预测位置）
                travel_time = dist / fleet_speed(to_send)
                target_x, target_y = predict_position(target, travel_time)
                angle = math.atan2(target_y - my_p.y, target_x - my_p.x)

                moves.append([my_p.id, angle, int(to_send)])
                info['ships'] -= to_send
                ships_assigned += to_send

    return moves
