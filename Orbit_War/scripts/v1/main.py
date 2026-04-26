import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

def agent(obs):
    moves = []

    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
    angular_velocity = obs.get("angular_velocity", 0.03) if isinstance(obs, dict) else obs.angular_velocity
    comet_planet_ids = set(obs.get("comet_planet_ids", [])) if isinstance(obs, dict) else set(obs.comet_planet_ids)

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]

    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner not in [-1, player]]
    neutral_planets = [p for p in planets if p.owner == -1]
    enemy_fleets = [f for f in fleets if f.owner != player]

    if not my_planets:
        return moves

    # ---------- 核心工具 ----------
    def fleet_speed(ship_count):
        """与官方源码严格一致的速度计算"""
        if ship_count <= 1:
            return 1.0
        max_speed = 6.0
        return 1.0 + (max_speed - 1.0) * (math.log(ship_count) / math.log(1000)) ** 1.5

    def is_orbiting(planet):
        """判断行星是否绕日运动（使用半径精确判定）"""
        dist = math.hypot(planet.x - 50, planet.y - 50)
        return dist + planet.radius < 50

    def planet_position_at(planet, t):
        """预测 t 回合后的行星位置（仅轨道行星）"""
        if not is_orbiting(planet):
            return (planet.x, planet.y)
        dist = math.hypot(planet.x - 50, planet.y - 50)
        angle = math.atan2(planet.y - 50, planet.x - 50)
        future_angle = angle + angular_velocity * t
        return (50 + dist * math.cos(future_angle),
                50 + dist * math.sin(future_angle))

    def path_hits_sun(x1, y1, x2, y2, sx=50, sy=50, sr=10):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return False
        t = max(0, min(1, ((sx - x1)*dx + (sy - y1)*dy)/(dx*dx + dy*dy)))
        px, py = x1 + t*dx, y1 + t*dy
        return math.hypot(px - sx, py - sy) < sr + 2

    def intercept_calc(from_x, from_y, target_planet, ship_count):
        """
        迭代计算：
        - 发射角度（弧度）
        - 精确到达时间
        - 拦截点坐标
        """
        speed = fleet_speed(ship_count)
        cur_tx, cur_ty = target_planet.x, target_planet.y
        dist = math.hypot(cur_tx - from_x, cur_ty - from_y)
        T = dist / speed if speed > 0 else 999
        for _ in range(10):  # 10次迭代足够高精度
            fx, fy = planet_position_at(target_planet, T)
            dist = math.hypot(fx - from_x, fy - from_y)
            T = dist / speed if speed > 0 else 999
        angle = math.atan2(fy - from_y, fx - from_x)
        return angle, T, (fx, fy)

    # ---------- 敌方 incoming 收集 ----------
    incoming = {p.id: [] for p in planets}
    for ef in enemy_fleets:
        speed = fleet_speed(ef.ships)
        for p in planets:
            angle_to_planet = math.atan2(p.y - ef.y, p.x - ef.x)
            diff = abs(angle_to_planet - ef.angle)
            diff = min(diff, 2*math.pi - diff)
            if diff < 0.3:
                dist = math.hypot(ef.x - p.x, ef.y - p.y)
                t = dist / max(0.5, speed)
                incoming[p.id].append((ef.owner, ef.ships, t))

    # 预判敌人即将发动的进攻（用独立字典，不修改 Planet 元组）
    enemy_available = {}
    for ep in enemy_planets:
        avail = max(0, ep.ships - max(10, int(ep.ships * 0.2)))
        enemy_available[ep.id] = avail

    for ep_id, avail in enemy_available.items():
        if avail <= 0:
            continue
        ep = next(p for p in enemy_planets if p.id == ep_id)
        target_my = min(my_planets, key=lambda p: math.hypot(p.x - ep.x, p.y - ep.y))
        if target_my:
            dist = math.hypot(ep.x - target_my.x, ep.y - target_my.y)
            speed = fleet_speed(avail)
            eta = dist / max(0.5, speed)
            incoming[target_my.id].append((ep.owner, avail, eta))

    # 我方已派出的舰队（按目标行星统计，通过角度反推）
    my_incoming = {}
    for f in fleets:
        if f.owner == player:
            from_planet = next((p for p in planets if p.id == f.from_planet_id), None)
            if from_planet is None:
                continue
            best_target = None
            best_dist = float("inf")
            for t in planets:
                if t.id == from_planet.id:
                    continue
                angle_to_t = math.atan2(t.y - from_planet.y, t.x - from_planet.x)
                diff = abs(angle_to_t - f.angle)
                diff = min(diff, 2*math.pi - diff)
                if diff < 0.3:
                    dist = math.hypot(from_planet.x - t.x, from_planet.y - t.y)
                    if dist < best_dist:
                        best_dist = dist
                        best_target = t
            if best_target:
                my_incoming[best_target.id] = my_incoming.get(best_target.id, 0) + f.ships

    # 防守模拟（精准预测是否守得住）
    def simulate_hold(planet, incoming_events, T=30):
        owner = planet.owner
        ships = planet.ships
        events = sorted(incoming_events, key=lambda x: x[2])
        idx = 0
        for t in range(T):
            if owner != -1:
                ships += planet.production
            arrivals = []
            while idx < len(events) and events[idx][2] <= t:
                arrivals.append(events[idx])
                idx += 1
            if arrivals:
                forces = {owner: ships}
                for o, s, _ in arrivals:
                    forces[o] = forces.get(o, 0) + s
                sorted_forces = sorted(forces.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_forces) > 1:
                    o1, s1 = sorted_forces[0]
                    o2, s2 = sorted_forces[1]
                    if s1 > s2:
                        owner, ships = o1, s1 - s2
                    elif s2 > s1:
                        owner, ships = o2, s2 - s1
                    else:
                        owner, ships = -1, 0
                else:
                    owner, ships = sorted_forces[0]
        return owner == player

    used = set()

    # ---------- 1️⃣ 彗星撤离（使用拦截计算打移动目标） ----------
    for p in my_planets:
        if p.id in comet_planet_ids and p.ships > 10:
            best = None
            best_dist = 1e9
            for q in my_planets:
                if q.id != p.id and q.id not in comet_planet_ids:
                    d = math.hypot(p.x - q.x, p.y - q.y)
                    if d < best_dist:
                        best_dist = d
                        best = q
            if best:
                # 精确拦截
                angle, _, (hit_x, hit_y) = intercept_calc(p.x, p.y, best, int(p.ships * 0.8))
                if not path_hits_sun(p.x, p.y, hit_x, hit_y):
                    moves.append([p.id, angle, int(p.ships * 0.8)])
                    used.add(p.id)

    # ---------- 2️⃣ 撤退（精确拦截己方行星） ----------
    for p in my_planets:
        if p.id in used:
            continue
        if not simulate_hold(p, incoming[p.id]):
            best = None
            best_dist = 1e9
            for q in my_planets:
                if q.id != p.id:
                    d = math.hypot(p.x - q.x, p.y - q.y)
                    if d < best_dist:
                        best_dist = d
                        best = q
            if best:
                send_amount = int(p.ships * 0.8)
                angle, _, (hit_x, hit_y) = intercept_calc(p.x, p.y, best, send_amount)
                if not path_hits_sun(p.x, p.y, hit_x, hit_y):
                    moves.append([p.id, angle, send_amount])
                    used.add(p.id)

    # ---------- 3️⃣ 扩张（精确拦截 + 智能需求） ----------
    targets = neutral_planets + enemy_planets

    def score_target(t):
        my_dist = min(math.hypot(p.x - t.x, p.y - t.y) for p in my_planets)
        enemy_dist = 100
        if enemy_planets:
            enemy_dist = min(math.hypot(p.x - t.x, p.y - t.y) for p in enemy_planets)
        safety = enemy_dist - my_dist
        return t.production * 5 + safety - t.ships

    targets.sort(key=score_target, reverse=True)

    for t in targets[:5]:
        # 选最近的源，用于预估到达时间
        nearest_src = min(my_planets, key=lambda p: math.hypot(p.x - t.x, p.y - t.y))
        _, eta_typical, _ = intercept_calc(nearest_src.x, nearest_src.y, t, 30)

        # 预估到达时敌军驻军
        if t in enemy_planets:
            base = t.ships
            prod = t.production * eta_typical
            support = sum(s for o, s, time in incoming.get(t.id, []) if o == t.owner and time <= eta_typical)
            need = max(1, base + prod + support + 1)
        else:
            need = t.ships + 1

        already_sent = my_incoming.get(t.id, 0)
        need = max(1, need - already_sent)

        sources = sorted(my_planets, key=lambda p: math.hypot(p.x - t.x, p.y - t.y))
        total = 0
        for s in sources:
            if s.id in used:
                continue
            available = int(s.ships * 0.6)
            if available <= 3:
                continue
            send = min(available, need - total)
            if send < 3:
                continue

            angle, _, (hit_x, hit_y) = intercept_calc(s.x, s.y, t, send)
            if path_hits_sun(s.x, s.y, hit_x, hit_y):
                continue

            moves.append([s.id, angle, send])
            used.add(s.id)
            total += send
            if total >= need:
                break

    return moves