"""
Orbit Wars v3 - DeepSeek Base + 6 Targeted Improvements

继承 DeepSeek 全部特性（影响力地图、多人博弈分析、动态驻军、焦土撤退、对手采样），
在其上新增/强化：
  F1 +1 极小打击：ship_options 加入 target_ships+1（结合 timeline 预测兵力）
  F2 领先惩罚：3+ 玩家局中若 v_me > 1.3×top_opp，则 v_me ×= 0.85（避免被围攻）
  F3 太阳近距惩罚：距太阳 ≤ 16 的行星 PLANET_OWN_BONUS ×= 0.85
  F4 显式开局中立偏置：前 OPENING_TURN_LIMIT=80 步中立 ×1.6 / 敌方本营 ×0.6
  F7 双源 swarm 放宽：扫描所有 ETA 容差内的源对（不只是最近两颗）
  F8 严格时间预算：OPP_TIME_FRACTION=0.35 + OPP_MAX_EVAL=3，避免候选爆炸
"""

import math
import time
import random
from collections import defaultdict


# ============================================================
# Constants
# ============================================================

TOTAL_STEPS = 500
CENTER_X, CENTER_Y = 50.0, 50.0
SUN_R = 10.0
BOARD_SIZE = 100.0
MAX_SPEED = 6.0
ROTATION_LIMIT = 50.0
LAUNCH_CLEARANCE = 0.01
_LOG1000 = math.log(1000.0)

# 策略超参数（默认值，会被动态模式覆盖）
HORIZON_VALUE = 120
HORIZON_SIM = 80
BEAM_WIDTH = 6
MAX_TARGETS_PER_SRC = 4
MAX_SOURCES = 6
FRONTIER_BONUS = 4.0
PLANET_OWN_BONUS = 2.0
DETOUR_OFFSETS_DEG = (5, -5, 10, -10, 18, -18)
SOFT_DEADLINE = 0.85
OPP_TIME_FRACTION = 0.35           # F8: 原 0.55 偏松，候选多时常超时
OPP_MAX_EVAL = 3                   # F8: 原 5 偏多
MIN_GARRISON_BASE = 3

# ---- F1: +1 极小打击 ----
PLUS_ONE_ENABLED = True
PLUS_ONE_MIN_OFFSET = 1            # 直射 +1（最省舰船）
SNIPE_MARGIN_PLUS_ONE = 1          # 狙击中立 +1（中立不产兵，安全）

# ---- F2: 领先惩罚（仅 3+ 玩家） ----
AHEAD_PENALTY_RATIO = 1.3          # v_me > top_opp × this 触发
AHEAD_PENALTY_FACTOR = 0.85        # v_me 缩放因子

# ---- F3: 太阳近距惩罚 ----
SUN_PROXIMITY_THRESHOLD = 16.0
SUN_PROXIMITY_FACTOR = 0.85

# ---- F4: 显式开局阶段 ----
OPENING_TURN_LIMIT = 80
OPENING_NEUTRAL_BOOST = 1.6
OPENING_HOSTILE_PENALTY = 0.6
NEUTRAL_BOOST_MID = 1.2

# ---- F7: 双源 swarm 放宽 ----
SWARM_MAX_SOURCE_PAIRS = 6         # 每目标最多评估的源对数
SWARM_PROD_THRESHOLD = 3
SWARM_ETA_GAP_MAX = 4

# 早期策略（动态调整）
EARLY_GAME_LIMIT = 40
EARLY_NEUTRAL_BONUS = 2.0
EARLY_ENEMY_PENALTY = 0.5
COVERAGE_SHIP_MARGIN = 5

# 安全边际
SAFETY_MARGIN_THRESHOLD = 0.3
SAFETY_MIN_ABSOLUTE = 5
SAFETY_PENALTY_FACTOR = 0.7
OPP_SAMPLE_COUNT = 2
OPP_STYLE_TEMPERATURE = 1.5

# 多源协同
DUAL_SOURCE_MIN_SHIPS = 10
DUAL_ETA_TOLERANCE = 2
MAX_DUAL_CANDIDATES = 3

# 自适应束宽度
BEAM_CUTOFF_RATIO = 0.25

# 终局动态权重
LATE_GAME_THRESHOLD = 40
LATE_SHIP_BONUS_STEP = 0.02

# 焦土撤退
DOOMED_FALL_TURN = 8
DOOMED_EVAC_RATIO = 1.0

# 行星保卫
PAL_GUARD_RADIUS = 25.0
PAL_GUARD_BONUS = 0.5

# 精确时机狙击
SNIPE_ENABLED = True
SNIPE_VALUE_MULTIPLIER = 2.0

# 价值折现
VALUE_DISCOUNT = 0.985

# 影响力地图
INFLUENCE_DECAY = 0.06            # 指数衰减因子（与飞行时间相关）

# 多人博弈乘数
VULTURE_MULT = 2.5                # 攻击弱势敌人的倍数
STRONG_ENEMY_PENALTY = 0.6        # 攻击强势敌人的惩罚

# ROI评分新增
BREAKEVEN_PENALTY_SCALE = 50.0    # 回本惩罚的尺度
DISTANCE_DISCOUNT_SCALE = 30.0    # 距离折扣的尺度
THREAT_BONUS_THRESHOLD = -10.0    # 影响力低于此值视为高威胁，给予加成


# ============================================================
# Data classes
# ============================================================

class Planet:
    __slots__ = ('id', 'owner', 'x', 'y', 'radius', 'ships', 'production')
    def __init__(self, pid, owner, x, y, radius, ships, production):
        self.id = pid
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.ships = ships
        self.production = production


class Fleet:
    __slots__ = ('id', 'owner', 'x', 'y', 'angle', 'from_planet_id', 'ships')
    def __init__(self, fid, owner, x, y, angle, from_planet_id, ships):
        self.id = fid
        self.owner = owner
        self.x = x
        self.y = y
        self.angle = angle
        self.from_planet_id = from_planet_id
        self.ships = ships


class Atom:
    __slots__ = ('src_id', 'target_id', 'ships', 'angle', 'eta', 'value')
    def __init__(self, src_id, target_id, ships, angle, eta, value=0.0):
        self.src_id = src_id
        self.target_id = target_id
        self.ships = ships
        self.angle = angle
        self.eta = eta
        self.value = value


# ============================================================
# Physics layer (unchanged)
# ============================================================

def fleet_speed(ships, max_speed=MAX_SPEED):
    if ships <= 1:
        return 1.0
    s = 1.0 + (max_speed - 1.0) * (math.log(ships) / _LOG1000) ** 1.5
    return min(s, max_speed)


def get_launch_position(src, angle):
    r = src.radius + LAUNCH_CLEARANCE
    return src.x + r * math.cos(angle), src.y + r * math.sin(angle)


def _angle_norm(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def line_hits_circle(x0, y0, x1, y1, cx, cy, cr):
    dx, dy = x1 - x0, y1 - y0
    fx, fy = x0 - cx, y0 - cy
    a = dx * dx + dy * dy
    if a < 1e-12:
        return math.hypot(fx, fy) < cr
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - cr * cr
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 < t2)


def line_hits_sun(x0, y0, x1, y1):
    return line_hits_circle(x0, y0, x1, y1, CENTER_X, CENTER_Y, SUN_R)


def is_orbital(planet):
    d = math.hypot(planet.x - CENTER_X, planet.y - CENTER_Y)
    return d + planet.radius < ROTATION_LIMIT


def estimate_signed_omega(planet, init_pos, step, base_omega):
    if step <= 0 or base_omega <= 0:
        return base_omega
    ix, iy = init_pos
    theta_init = math.atan2(iy - CENTER_Y, ix - CENTER_X)
    theta_curr = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    delta_obs = _angle_norm(theta_curr - theta_init)
    best_omega, best_err = base_omega, float("inf")
    for sign in (+1, -1):
        expected_total = sign * base_omega * step
        diff = abs(_angle_norm(delta_obs - expected_total))
        if diff < best_err:
            best_err = diff
            best_omega = sign * base_omega
    return best_omega


def build_omega_map(planets, initial_planets, step, base_omega):
    init_by_id = {p[0]: (p[2], p[3]) for p in initial_planets}
    omega_map = {}
    for p in planets:
        if not is_orbital(p):
            omega_map[p.id] = 0.0
            continue
        init = init_by_id.get(p.id)
        if init is None:
            omega_map[p.id] = base_omega
            continue
        omega_map[p.id] = estimate_signed_omega(p, init, step, base_omega)
    return omega_map


def predict_orbit_position(planet, omega, t):
    dx = planet.x - CENTER_X
    dy = planet.y - CENTER_Y
    r = math.hypot(dx, dy)
    theta0 = math.atan2(dy, dx)
    theta = theta0 + omega * t
    return CENTER_X + r * math.cos(theta), CENTER_Y + r * math.sin(theta)


def predict_comet_position(comet_group, planet_id, t):
    if comet_group is None:
        return None
    pids = comet_group.get("planet_ids", [])
    if planet_id not in pids:
        return None
    idx = pids.index(planet_id)
    paths = comet_group.get("paths", [])
    if idx >= len(paths):
        return None
    path = paths[idx]
    path_index = comet_group.get("path_index", 0)
    f_idx = path_index + t
    i0 = int(math.floor(f_idx))
    i1 = i0 + 1
    if i0 < 0 or i1 >= len(path):
        i0 = max(0, min(i0, len(path) - 1))
        return path[i0][0], path[i0][1]
    frac = f_idx - i0
    x = path[i0][0] * (1.0 - frac) + path[i1][0] * frac
    y = path[i0][1] * (1.0 - frac) + path[i1][1] * frac
    return x, y


def predict_target_position(target, t, omega_map, cid_to_group, comet_ids):
    if target.id in comet_ids:
        pos = predict_comet_position(cid_to_group.get(target.id), target.id, t)
        if pos is not None:
            return pos
    omega = omega_map.get(target.id, 0.0)
    if abs(omega) > 1e-9:
        return predict_orbit_position(target, omega, t)
    return target.x, target.y


def compute_intercept(src, target, ships, omega_map, cid_to_group, comet_ids,
                     max_iter=30, t_tol=1e-3, ang_tol=1e-4):
    if ships <= 0:
        return None, None
    speed = fleet_speed(ships)
    if speed <= 0:
        return None, None
    angle = math.atan2(target.y - src.y, target.x - src.x)
    sx, sy = get_launch_position(src, angle)
    t_est = math.hypot(target.x - sx, target.y - sy) / speed
    last_a, last_t = angle, t_est
    for _ in range(max_iter):
        sx, sy = get_launch_position(src, angle)
        tx, ty = predict_target_position(target, t_est, omega_map, cid_to_group, comet_ids)
        new_dist = math.hypot(tx - sx, ty - sy)
        new_t = new_dist / speed
        new_angle = math.atan2(ty - sy, tx - sx)
        d_a = _angle_norm(new_angle - last_a)
        d_t = new_t - last_t
        if abs(d_t) < t_tol and abs(d_a) < ang_tol:
            angle, t_est = new_angle, new_t
            break
        angle = _angle_norm(last_a + 0.6 * d_a)
        t_est = max(0.1, last_t + 0.6 * d_t)
        last_a, last_t = angle, t_est

    sx, sy = get_launch_position(src, angle)
    speed_a = fleet_speed(ships)
    fx = sx + t_est * speed_a * math.cos(angle)
    fy = sy + t_est * speed_a * math.sin(angle)
    tx, ty = predict_target_position(target, t_est, omega_map, cid_to_group, comet_ids)
    miss = math.hypot(fx - tx, fy - ty)
    if miss > max(target.radius, 1.2):
        return None, None
    return angle, t_est


def path_blocked_by_other_planet(src, target, angle, eta, ships, planets,
                                 omega_map, cid_to_group, comet_ids):
    sx, sy = get_launch_position(src, angle)
    speed = fleet_speed(ships)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    n_frames = int(math.ceil(eta)) + 1
    prev_x, prev_y = sx, sy
    for k in range(1, n_frames + 1):
        fx = sx + k * speed * cos_a
        fy = sy + k * speed * sin_a
        tm = k - 0.5
        for p in planets:
            if p.id == src.id or p.id == target.id:
                continue
            if p.id in comet_ids:
                pos = predict_comet_position(cid_to_group.get(p.id), p.id, tm)
                if pos is None:
                    continue
                px, py = pos
            elif abs(omega_map.get(p.id, 0.0)) > 1e-9:
                px, py = predict_orbit_position(p, omega_map[p.id], tm)
            else:
                px, py = p.x, p.y
            if line_hits_circle(prev_x, prev_y, fx, fy, px, py, p.radius + 0.05):
                return True
        prev_x, prev_y = fx, fy
    return False


def trace_intercept(src, angle, target, speed, world, max_turns=HORIZON_SIM):
    sx, sy = get_launch_position(src, angle)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    prev_x, prev_y = sx, sy
    for k in range(1, max_turns + 1):
        fx = sx + k * speed * cos_a
        fy = sy + k * speed * sin_a
        if not (0 <= fx <= BOARD_SIZE and 0 <= fy <= BOARD_SIZE):
            return None
        if line_hits_sun(prev_x, prev_y, fx, fy):
            return None
        tm = k - 0.5
        if target.id in world.comet_ids:
            pos = predict_comet_position(world.cid_to_group.get(target.id), target.id, tm)
            if pos is None:
                continue
            px, py = pos
        elif abs(world.omega_map.get(target.id, 0.0)) > 1e-9:
            px, py = predict_orbit_position(target, world.omega_map[target.id], tm)
        else:
            px, py = target.x, target.y
        if line_hits_circle(prev_x, prev_y, fx, fy, px, py, target.radius):
            return float(k)
        prev_x, prev_y = fx, fy
    return None


def compute_intercept_with_detour(src, target, ships, world):
    angle, eta = compute_intercept(
        src, target, ships, world.omega_map, world.cid_to_group, world.comet_ids
    )
    if angle is None:
        return None

    speed = fleet_speed(ships)
    sx, sy = get_launch_position(src, angle)
    fx = sx + eta * speed * math.cos(angle)
    fy = sy + eta * speed * math.sin(angle)
    direct_blocked = path_blocked_by_other_planet(
        src, target, angle, eta, ships, world.planet_list,
        world.omega_map, world.cid_to_group, world.comet_ids
    )
    direct_sun = line_hits_sun(sx, sy, fx, fy)
    if not direct_blocked and not direct_sun:
        return angle, eta, 0.0

    for deg in DETOUR_OFFSETS_DEG:
        offset_rad = math.radians(deg)
        new_angle = _angle_norm(angle + offset_rad)
        new_eta = trace_intercept(src, new_angle, target, speed, world)
        if new_eta is None:
            continue
        nsx, nsy = get_launch_position(src, new_angle)
        nfx = nsx + new_eta * speed * math.cos(new_angle)
        nfy = nsy + new_eta * speed * math.sin(new_angle)
        if line_hits_sun(nsx, nsy, nfx, nfy):
            continue
        if path_blocked_by_other_planet(
            src, target, new_angle, new_eta, ships, world.planet_list,
            world.omega_map, world.cid_to_group, world.comet_ids
        ):
            continue
        return new_angle, new_eta, float(deg)

    return None


def predict_fleet_arrival(fleet, planets, omega_map, cid_to_group, comet_ids,
                          max_turns=HORIZON_SIM):
    fx0, fy0 = fleet.x, fleet.y
    speed = fleet_speed(fleet.ships)
    cos_a, sin_a = math.cos(fleet.angle), math.sin(fleet.angle)
    prev_x, prev_y = fx0, fy0
    for k in range(1, max_turns + 1):
        nx = fx0 + k * speed * cos_a
        ny = fy0 + k * speed * sin_a
        if not (0 <= nx <= BOARD_SIZE and 0 <= ny <= BOARD_SIZE):
            return (k, None)
        if line_hits_sun(prev_x, prev_y, nx, ny):
            return (k, None)
        tm = k - 0.5
        for p in planets:
            if p.id in comet_ids:
                pos = predict_comet_position(cid_to_group.get(p.id), p.id, tm)
                if pos is None:
                    continue
                px, py = pos
            elif abs(omega_map.get(p.id, 0.0)) > 1e-9:
                px, py = predict_orbit_position(p, omega_map[p.id], tm)
            else:
                px, py = p.x, p.y
            if line_hits_circle(prev_x, prev_y, nx, ny, px, py, p.radius):
                return (k, p.id)
        prev_x, prev_y = nx, ny
    return (max_turns + 1, None)


# ============================================================
# Influence Map & Multi-Player Analysis (NEW)
# ============================================================

def compute_influence_map(planets, player_id, omega_map, cid_to_group, comet_ids):
    """计算每颗行星的影响力值。正值 = 我方优势，负值 = 敌方威胁。"""
    n = len(planets)
    influence = [0.0] * n
    # 为所有行星建立快速查询的索引
    planet_list = list(planets.values()) if isinstance(planets, dict) else planets
    for i, p in enumerate(planet_list):
        for j, q in enumerate(planet_list):
            if i == j:
                continue
            dist = math.hypot(p.x - q.x, p.y - q.y)
            # 用航行时间代表“距离”，使用中位数舰船速度来估算
            approx_time = dist / fleet_speed(15) + 1.0
            decay = math.exp(-INFLUENCE_DECAY * approx_time)
            ships = q.ships
            if q.owner == player_id:
                influence[i] += ships * decay
            elif q.owner != -1:
                influence[i] -= ships * decay
    return influence


def analyze_players(planets, my_id):
    """分析各玩家实力，返回 {player_id: {total_ships, total_prod, planet_count, is_weak, is_strong}}。"""
    players = defaultdict(lambda: {"total_ships": 0, "total_production": 0, "planet_count": 0})
    for p in planets.values() if isinstance(planets, dict) else planets:
        if p.owner == -1:
            continue
        players[p.owner]["total_ships"] += p.ships
        players[p.owner]["total_production"] += p.production
        players[p.owner]["planet_count"] += 1

    my_strength = players[my_id]["total_ships"]
    for pid, info in players.items():
        if pid == my_id:
            info["is_weak"] = False
            info["is_strong"] = False
            continue
        info["is_weak"] = info["total_ships"] < my_strength * 0.6
        info["is_strong"] = info["total_ships"] > my_strength * 1.5
    return dict(players)


# ============================================================
# WorldState (updated with influence and player analysis)
# ============================================================

class WorldState:
    def __init__(self, player, step, planets, fleets, initial_planets, base_omega,
                 comets, comet_ids):
        self.my_id = player
        self.step = step
        self.planet_list = planets
        self.planets = {p.id: p for p in planets}
        self.fleets = fleets
        self.fleet_by_id = {f.id: f for f in fleets}
        self.comet_ids = set(comet_ids)
        self.cid_to_group = {}
        for g in comets:
            for pid in g.get("planet_ids", []):
                self.cid_to_group[pid] = g
        self.omega_map = build_omega_map(planets, initial_planets, step, base_omega)

        owners = set()
        for p in planets:
            if p.owner != -1:
                owners.add(p.owner)
        for f in fleets:
            owners.add(f.owner)
        owners.add(player)
        self.player_ids = sorted(owners)
        self.opponent_ids = [o for o in self.player_ids if o != player]
        self.remaining_steps = max(1, TOTAL_STEPS - step)

        self.fleet_arrivals = {}
        for f in fleets:
            self.fleet_arrivals[f.id] = predict_fleet_arrival(
                f, planets, self.omega_map, self.cid_to_group, self.comet_ids
            )

        self.covered_neutrals = set()
        for pid, p in self.planets.items():
            if p.owner != -1:
                continue
            total_incoming = 0
            for fid, (eta, tid) in self.fleet_arrivals.items():
                if tid == pid and self.fleet_by_id[fid].owner == player:
                    total_incoming += self.fleet_by_id[fid].ships
            if total_incoming >= p.ships + COVERAGE_SHIP_MARGIN:
                self.covered_neutrals.add(pid)

        # 实力比与动态模式
        my_total = sum(p.ships for p in planets if p.owner == player)
        enemy_total = 0
        for p in planets:
            if p.owner != player and p.owner != -1:
                enemy_total += p.ships
        for f in fleets:
            if f.owner == player:
                my_total += f.ships
            elif f.owner != -1:
                enemy_total += f.ships
        strength_ratio = my_total / max(1, enemy_total)

        if strength_ratio < 0.8:
            self.early_neutral_bonus = EARLY_NEUTRAL_BONUS * 1.5
            self.early_enemy_penalty = EARLY_ENEMY_PENALTY * 0.8
            self.dual_source_min_ships = max(5, DUAL_SOURCE_MIN_SHIPS - 5)
            self.beam_width = min(BEAM_WIDTH + 2, 8)
            self.min_garrison_base = max(1, MIN_GARRISON_BASE - 1)
            self.snipe_enabled = True
        elif strength_ratio > 1.5:
            self.early_neutral_bonus = EARLY_NEUTRAL_BONUS * 0.8
            self.early_enemy_penalty = EARLY_ENEMY_PENALTY * 1.2
            self.dual_source_min_ships = DUAL_SOURCE_MIN_SHIPS + 5
            self.beam_width = max(4, BEAM_WIDTH - 1)
            self.min_garrison_base = MIN_GARRISON_BASE + 2
            self.snipe_enabled = True
        else:
            self.early_neutral_bonus = EARLY_NEUTRAL_BONUS
            self.early_enemy_penalty = EARLY_ENEMY_PENALTY
            self.dual_source_min_ships = DUAL_SOURCE_MIN_SHIPS
            self.beam_width = BEAM_WIDTH
            self.min_garrison_base = MIN_GARRISON_BASE
            self.snipe_enabled = True

        # 影响力地图
        self.influence = compute_influence_map(
            self.planets, self.my_id, self.omega_map, self.cid_to_group, self.comet_ids
        )
        # 行星ID -> 影响力值的映射
        self.influence_by_id = {}
        for idx, p in enumerate(self.planet_list):
            self.influence_by_id[p.id] = self.influence[idx]

        # 多人分析
        self.player_analysis = analyze_players(self.planets, self.my_id)

        # 动态防守底线（结合影响力）
        self.dynamic_min_garrison = {}
        for p in planets:
            if p.owner == player:
                self.dynamic_min_garrison[p.id] = self._compute_dynamic_min(p)
            else:
                self.dynamic_min_garrison[p.id] = 0

        # 预计算时间线用于检测 doomed 行星
        self.projected_timelines = self._project_base_timelines()

        self._intercept_cache = {}
        self._top_targets_cache = {}
        self._candidate_cache = {}
        self._best_response_cache = {}

    def _compute_dynamic_min(self, planet):
        if not self.opponent_ids:
            return self.min_garrison_base
        # 基础距离估算
        min_eta = float("inf")
        for opp_id in self.opponent_ids:
            for opp_p in self.planet_list:
                if opp_p.owner != opp_id:
                    continue
                d = math.hypot(opp_p.x - planet.x, opp_p.y - planet.y)
                speed = fleet_speed(max(1, opp_p.ships))
                eta = d / speed if speed > 0 else float("inf")
                min_eta = min(min_eta, eta)
        # 结合影响力（负面影响力大 = 威胁大）
        infl = self.influence_by_id.get(planet.id, 0.0)
        threat_bonus = max(0, -infl * 0.5)  # 威胁越大，需要的驻军越多
        if min_eta <= 15:
            base = max(8, int(planet.ships * 0.5))
        elif min_eta <= 30:
            base = max(5, int(planet.ships * 0.3))
        else:
            base = self.min_garrison_base
        return min(int(planet.ships * 0.7), int(base + threat_bonus))  # 不能超过70%

    def _project_base_timelines(self):
        """F1: horizon 扩展到 HORIZON_SIM，方便 ship_options/snipe 拿到 t=eta 时的预测兵力。"""
        arrivals = defaultdict(list)
        for fid, (eta, tid) in self.fleet_arrivals.items():
            if tid is None:
                continue
            f = self.fleet_by_id[fid]
            arrivals[tid].append((eta, f.owner, f.ships))
        timelines = {}
        for pid, p in self.planets.items():
            timelines[pid] = simulate_planet_timeline(
                p, arrivals.get(pid, []), HORIZON_SIM
            )
        return timelines

    def predicted_ships_at(self, planet_id, t, default_owner=None):
        """F1: 返回 planet_id 在 t 回合后的 (owner, ships) 预测；用于 +1 极小打击决策。

        基于已有舰队的到达推演，不考虑当前回合即将派出的舰队（因为 generate_candidates
        阶段还不知道最终会发什么）。"""
        tl = self.projected_timelines.get(planet_id)
        if tl is None or len(tl) == 0:
            p = self.planets.get(planet_id)
            if p is None:
                return (default_owner, 0)
            return (p.owner, int(p.ships))
        idx = max(0, min(int(math.ceil(t)), len(tl) - 1))
        owner, ships = tl[idx]
        return (owner, int(math.ceil(ships)))

    def is_doomed(self, planet_id):
        if planet_id not in self.projected_timelines:
            return False
        tl = self.projected_timelines[planet_id]
        for t in range(1, DOOMED_FALL_TURN + 1):
            if t >= len(tl):
                break
            owner, _ = tl[t]
            if owner != self.my_id:
                return t
        return None

    def get_intercept(self, src_id, target_id, ships):
        key = (src_id, target_id, int(ships))
        if key in self._intercept_cache:
            return self._intercept_cache[key]
        src = self.planets[src_id]
        target = self.planets[target_id]
        result = compute_intercept_with_detour(src, target, ships, self)
        self._intercept_cache[key] = result
        return result


# ============================================================
# Forward simulation (unchanged)
# ============================================================

def simulate_planet_timeline(planet, arrivals, horizon, initial_ships=None):
    horizon = int(math.ceil(horizon))
    by_turn = defaultdict(list)
    for eta, owner, ships in arrivals:
        eta_int = max(1, int(math.ceil(eta)))
        if eta_int > horizon or ships <= 0:
            continue
        by_turn[eta_int].append((owner, int(ships)))

    owner = planet.owner
    garrison = float(initial_ships if initial_ships is not None else planet.ships)
    timeline = [(owner, garrison)]

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production
        if turn in by_turn:
            arr = by_turn[turn]
            by_owner = defaultdict(int)
            for o, s in arr:
                by_owner[o] += s
            sorted_attackers = sorted(by_owner.items(), key=lambda x: -x[1])
            top_owner, top_ships = sorted_attackers[0]
            second_ships = sorted_attackers[1][1] if len(sorted_attackers) >= 2 else 0
            if len(sorted_attackers) >= 2 and second_ships == top_ships:
                pass
            else:
                survivor_ships = top_ships - second_ships
                if survivor_ships > 0:
                    if top_owner == owner:
                        garrison += survivor_ships
                    else:
                        garrison -= survivor_ships
                        if garrison < 0:
                            owner = top_owner
                            garrison = -garrison
        timeline.append((owner, max(0.0, garrison)))

    return timeline


def project_state(world, all_actions):
    launched_from = defaultdict(int)
    for pid, target_id, eta, ships in all_actions:
        if target_id is not None and ships > 0:
            launched_from[pid] += ships

    arrivals = defaultdict(list)
    for fid, (eta, tid) in world.fleet_arrivals.items():
        if tid is None:
            continue
        f = world.fleet_by_id[fid]
        arrivals[tid].append((eta, f.owner, f.ships))
    for pid, target_id, eta, ships in all_actions:
        if target_id is None or ships <= 0:
            continue
        arrivals[target_id].append((eta, pid, ships))

    timelines = {}
    for pid, p in world.planets.items():
        init_ships = p.ships - launched_from.get(pid, 0)
        if p.owner != -1:
            init_ships += p.production
        init_ships = max(0, init_ships)
        timelines[pid] = simulate_planet_timeline(
            p, arrivals.get(pid, []), HORIZON_VALUE, initial_ships=init_ships
        )
    return timelines


# ============================================================
# Value function (unchanged)
# ============================================================

def _build_enemy_arrivals(world, all_actions):
    enemy_arrivals = defaultdict(lambda: defaultdict(float))
    for fid, (eta, tid) in world.fleet_arrivals.items():
        if tid is None:
            continue
        f = world.fleet_by_id[fid]
        if f.owner != world.my_id:
            eta_int = max(1, int(math.ceil(eta)))
            enemy_arrivals[tid][eta_int] += f.ships
    for pid, target_id, eta, ships in all_actions:
        if target_id is None or ships <= 0:
            continue
        if pid != world.my_id:
            eta_int = max(1, int(math.ceil(eta)))
            enemy_arrivals[target_id][eta_int] += ships
    return enemy_arrivals


def _V_full(timelines, world, player, horizon, arrivals_breakdown=None):
    enemy_positions = []
    for pid, tl in timelines.items():
        owner, _ = tl[horizon]
        if owner != player and owner != -1:
            p = world.planets[pid]
            enemy_positions.append((p.x, p.y))

    friend_positions = []
    for pid, tl in timelines.items():
        owner, _ = tl[horizon]
        if owner == player:
            p = world.planets[pid]
            friend_positions.append((p.x, p.y))

    total = 0.0
    remaining = world.remaining_steps
    late_mult = 1.0
    if remaining < LATE_GAME_THRESHOLD:
        late_mult = 1.0 + (LATE_GAME_THRESHOLD - remaining) * LATE_SHIP_BONUS_STEP

    for pid, tl in timelines.items():
        p = world.planets[pid]
        disc_prod = 0.0
        final_ships = 0.0
        owned_at_end = False

        for t in range(0, horizon + 1):
            owner, ships = tl[t]
            if owner == player:
                safety_penalty = 1.0
                if arrivals_breakdown is not None and t in arrivals_breakdown.get(pid, {}):
                    incoming = arrivals_breakdown[pid][t]
                    if incoming > 0 and (ships < SAFETY_MIN_ABSOLUTE or ships < incoming * SAFETY_MARGIN_THRESHOLD):
                        safety_penalty = SAFETY_PENALTY_FACTOR

                disc_prod += p.production * (VALUE_DISCOUNT ** t) * safety_penalty
                if t == horizon:
                    owned_at_end = True
                    final_ships = ships

        if owned_at_end:
            disc_prod += final_ships * late_mult

        # F3: 太阳近距惩罚 — 紧贴太阳的行星更易被偷家（路径短、绕路少）
        sun_d = math.hypot(p.x - CENTER_X, p.y - CENTER_Y)
        if owned_at_end and sun_d <= SUN_PROXIMITY_THRESHOLD:
            disc_prod *= SUN_PROXIMITY_FACTOR

        frontier_bonus = 0.0
        if owned_at_end and enemy_positions:
            min_d = min(math.hypot(p.x - ex, p.y - ey) for ex, ey in enemy_positions)
            if min_d < 40:
                frontier_bonus = FRONTIER_BONUS * p.production / (min_d + 5.0)

        guard_bonus = 0.0
        if owned_at_end:
            nearby_friends = sum(1 for fx, fy in friend_positions
                                 if math.hypot(p.x - fx, p.y - fy) <= PAL_GUARD_RADIUS and (fx != p.x or fy != p.y))
            guard_bonus = p.production * PAL_GUARD_BONUS * min(3, nearby_friends)

        total += disc_prod + frontier_bonus + guard_bonus

    return total


def delta_V(timelines, world, player, arrivals_breakdown=None):
    v_me = _V_full(timelines, world, player, HORIZON_VALUE, arrivals_breakdown)
    v_opp_max = 0.0
    v_opp_top = 0.0  # F2: 用于领先惩罚
    for opp in world.opponent_ids:
        v_opp = _V_full(timelines, world, opp, HORIZON_VALUE)
        if v_opp > v_opp_max:
            v_opp_max = v_opp
        if v_opp > v_opp_top:
            v_opp_top = v_opp

    # F2: 3+ 玩家局中若我方明显领先 top 对手，主动收敛避免被围攻
    if len(world.opponent_ids) >= 2 and v_opp_top > 0:
        if v_me > v_opp_top * AHEAD_PENALTY_RATIO:
            v_me *= AHEAD_PENALTY_FACTOR

    return v_me - v_opp_max


# ============================================================
# Candidate generation (updated make_atom with ROI multipliers)
# ============================================================

def comet_remaining_turns(comet_group, planet_id):
    if comet_group is None:
        return 999
    pids = comet_group.get("planet_ids", [])
    if planet_id not in pids:
        return 999
    idx = pids.index(planet_id)
    paths = comet_group.get("paths", [])
    path_index = comet_group.get("path_index", 0)
    if idx < len(paths):
        return max(0, len(paths[idx]) - path_index)
    return 0


def top_targets_for_player(src, world, player, top_k):
    cache_key = (src.id, player)
    cached = world._top_targets_cache.get(cache_key)
    if cached is not None:
        return cached

    candidates = []
    is_early = world.step < EARLY_GAME_LIMIT

    for tid, tgt in world.planets.items():
        if tgt.owner == player or tid == src.id:
            continue
        if tid in world.covered_neutrals:
            continue
        if tid in world.comet_ids:
            life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
            if life <= 1:
                continue

        d = math.hypot(src.x - tgt.x, src.y - tgt.y)
        if d < 1:
            continue

        proj_ships = max(int(tgt.ships) + 5, 10)
        speed = fleet_speed(proj_ships)
        eta = d / speed
        if eta > HORIZON_SIM * 0.8:
            continue

        # 使用影响力地图给予初始排序调整
        infl = world.influence_by_id.get(tid, 0.0)
        cost = max(1, int(tgt.ships) + 1)
        val = tgt.production / (cost * 0.4 + eta + 5.0)

        if tgt.owner == -1:
            val *= 1.2
            # 中立星影响力高代表接近我方，更容易占领
            if infl > 10:
                val *= 1.3
        else:
            # 敌方行星，影响力负值越大威胁越大，优先处理
            if infl < THREAT_BONUS_THRESHOLD:
                val *= 1.5

        # F4: 三阶段开局/中盘策略 — 比 deepseek 原"二阶段 (early/normal)"更连续
        # step <  EARLY_GAME_LIMIT(40)   : deepseek 原 2.0/0.5 极强偏置（探险阶段）
        # step <  OPENING_TURN_LIMIT(80) : 1.6/0.6 中等偏置（继续抢中立但不再硬怼敌方）
        # step >= OPENING_TURN_LIMIT     : 1.2/1.0 微偏中立（中盘后回归 ROI 主导）
        if is_early:
            if tgt.owner == -1:
                val *= world.early_neutral_bonus
            else:
                val *= world.early_enemy_penalty
        elif world.step < OPENING_TURN_LIMIT:
            if tgt.owner == -1:
                val *= OPENING_NEUTRAL_BOOST
            else:
                val *= OPENING_HOSTILE_PENALTY
        else:
            if tgt.owner == -1:
                val *= NEUTRAL_BOOST_MID

        candidates.append((val, tid))

    candidates.sort(reverse=True)
    result = [tid for _, tid in candidates[:top_k]]
    world._top_targets_cache[cache_key] = result
    return result


def ship_options(src, tgt, max_ships, world=None):
    """F1: 加入 target_ships+1 极小打击；若提供 world 则用 timeline 预测 t=eta 时的兵力。

    候选兵力档位（去重后排序）：
      - +1：极小打击（官方 baseline + 排行榜前列普遍用法），最省舰船
      - +2：deepseek 原档位，作 +1 失败时的 fallback
      - smart_margin：综合距离 + production×eta 的安全余量
      - +8 / max_ships：高威胁/全力压制
    """
    if max_ships <= 0:
        return []

    d = math.hypot(src.x - tgt.x, src.y - tgt.y)
    if world is not None:
        probe_ships = max(int(tgt.ships) + 5, 10)
        eta_est = d / fleet_speed(probe_ships)
        _owner, pred_ships = world.predicted_ships_at(tgt.id, eta_est, default_owner=tgt.owner)
        target_ships = max(1, pred_ships)
    else:
        target_ships = max(1, int(tgt.ships))

    options = set()
    if PLUS_ONE_ENABLED:
        options.add(min(max_ships, target_ships + PLUS_ONE_MIN_OFFSET))
    options.add(min(max_ships, target_ships + 2))
    options.add(min(max_ships, target_ships + 8))
    options.add(max_ships)
    prod_bonus = tgt.production * 2
    distance_bonus = int(d / 25)
    smart_margin = target_ships + 4 + prod_bonus + distance_bonus
    options.add(min(max_ships, smart_margin))
    return sorted(o for o in options if 1 <= o <= max_ships)


def make_atom(src, tgt, ships, world):
    aim = world.get_intercept(src.id, tgt.id, ships)
    if aim is None:
        return None
    angle, eta, _detour = aim
    if eta > HORIZON_SIM:
        return None

    remaining = HORIZON_VALUE
    if tgt.id in world.comet_ids:
        life = comet_remaining_turns(world.cid_to_group.get(tgt.id), tgt.id)
        remaining = min(remaining, life)

    # === 基础 ROI 计算 ===
    production_gain = tgt.production * max(0, remaining - int(eta))
    needed = ships  # 当前候选的舰船数（实际需按目标当前舰船估算，但用于排序可接受）
    # 粗略回本时间
    breakeven = needed / max(tgt.production, 0.1)
    breakeven_penalty = 1.0 / (1.0 + breakeven / BREAKEVEN_PENALTY_SCALE)
    distance_penalty = 1.0 / (1.0 + eta / DISTANCE_DISCOUNT_SCALE)

    # 威胁乘数：来自影响力地图
    infl = world.influence_by_id.get(tgt.id, 0.0)
    threat_mult = 1.0
    if infl < THREAT_BONUS_THRESHOLD and tgt.owner != -1:
        threat_mult = 1.5  # 高威胁敌方行星优先处理

    # 脆弱性乘数：来自多人分析
    vuln_mult = 1.0
    if tgt.owner != -1 and tgt.owner != src.owner:
        owner_info = world.player_analysis.get(tgt.owner, {})
        if owner_info.get("is_weak"):
            vuln_mult = VULTURE_MULT
        elif owner_info.get("is_strong"):
            vuln_mult = STRONG_ENEMY_PENALTY

    # 阶段乘数
    is_early = world.step < EARLY_GAME_LIMIT
    is_late = world.remaining_steps < LATE_GAME_THRESHOLD
    if is_early:
        phase_mult = 2.0 if tgt.owner == -1 else 0.4
    elif is_late:
        phase_mult = 1.5 if tgt.owner != -1 else 0.8
    else:
        phase_mult = 1.0

    # 最终启发式价值
    roi = production_gain / (needed * (eta + 1.0)) if needed > 0 else 0.0
    value = roi * threat_mult * vuln_mult * phase_mult * breakeven_penalty * distance_penalty

    # 保持对其他乘数的兼容性（如占领敌方行星的额外奖励）
    if tgt.owner != -1 and tgt.owner != src.owner:
        value *= 1.15

    return Atom(src.id, tgt.id, ships, angle, eta, value)


def generate_candidates(world, player_id, max_sets=None):
    if max_sets is None:
        max_sets = world.beam_width

    cache = world._candidate_cache
    if player_id in cache:
        return cache[player_id]

    my_planets = [p for p in world.planet_list if p.owner == player_id]
    if not my_planets:
        cache[player_id] = [[]]
        return cache[player_id]

    my_planets.sort(key=lambda p: -p.ships)
    my_planets = my_planets[:MAX_SOURCES]

    atoms = []
    for src in my_planets:
        min_garrison = world.dynamic_min_garrison.get(src.id, world.min_garrison_base)
        max_send = int(src.ships) - min_garrison
        if max_send <= 0:
            continue
        targets = top_targets_for_player(src, world, player_id, MAX_TARGETS_PER_SRC)
        for tid in targets:
            tgt = world.planets[tid]
            for ships in ship_options(src, tgt, max_send, world=world):
                atom = make_atom(src, tgt, ships, world)
                if atom:
                    atoms.append(atom)

    # 精确时机狙击
    if world.snipe_enabled and player_id == world.my_id:
        for tid, tgt in world.planets.items():
            if tgt.owner != -1 or tid in world.covered_neutrals:
                continue
            if tid in world.comet_ids:
                life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
                if life <= 2:
                    continue
            enemy_eta = float("inf")
            for fid, (eta, t_id) in world.fleet_arrivals.items():
                if t_id == tid and world.fleet_by_id[fid].owner != player_id:
                    if eta < enemy_eta:
                        enemy_eta = eta
            if enemy_eta == float("inf") or enemy_eta < 2:
                continue
            # F1: 中立不产兵，狙击用 +1 即可
            need_ships = int(tgt.ships) + SNIPE_MARGIN_PLUS_ONE
            for src in my_planets:
                min_g = world.dynamic_min_garrison.get(src.id, world.min_garrison_base)
                available = max(0, int(src.ships) - min_g)
                if available < need_ships:
                    continue
                aim = world.get_intercept(src.id, tid, need_ships)
                if aim is None:
                    continue
                angle, eta, _ = aim
                if eta <= enemy_eta - 1:
                    remaining = HORIZON_VALUE
                    if tid in world.comet_ids:
                        life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
                        remaining = min(remaining, life)
                    production_gain = tgt.production * max(0, remaining - int(eta))
                    cost = need_ships + int(eta) * 0.5
                    # 狙击价值再乘以乘数
                    value = (production_gain / (cost + 1.0)) * SNIPE_VALUE_MULTIPLIER
                    # 也应用影响力、多人分析等？可保持简单狙击加成就好
                    snipe_atom = Atom(src.id, tid, need_ships, angle, eta, value)
                    atoms.append(snipe_atom)
                    break

    atoms.sort(key=lambda a: -a.value)

    candidates = [[]]
    if not atoms:
        cache[player_id] = candidates
        return candidates

    best_val = atoms[0].value
    selected_atoms = []
    committed = defaultdict(int)

    for atom in atoms:
        if atom.value < best_val * BEAM_CUTOFF_RATIO:
            break

        tgt = world.planets[atom.target_id]
        if tgt.owner == -1:
            already = committed[atom.target_id]
            if already >= tgt.ships + COVERAGE_SHIP_MARGIN:
                continue
        if atom.src_id not in {a.src_id for a in selected_atoms}:
            selected_atoms.append(atom)
            committed[atom.target_id] += atom.ships
            if len(selected_atoms) >= max_sets:
                break

    cur_set = []
    for atom in selected_atoms:
        cur_set = list(cur_set) + [atom]
        candidates.append(cur_set)
    if [] not in candidates:
        candidates.append([])

    # 多源协同 — F7: 放宽到所有 ETA 容差内的源对（不只是最近两颗）
    dual_candidates = []
    for tid, tgt in world.planets.items():
        if tgt.owner == player_id:
            continue
        # 仅对值得 swarm 的目标（高产能或敌方守家）尝试
        if tgt.production < SWARM_PROD_THRESHOLD and tgt.owner == -1:
            continue
        if tgt.ships < world.dual_source_min_ships and tgt.owner == -1:
            continue
        remaining_life = HORIZON_VALUE
        if tid in world.comet_ids:
            life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
            remaining_life = min(remaining_life, life)
        if remaining_life < 15:
            continue

        sorted_srcs = sorted(my_planets, key=lambda src: math.hypot(src.x - tgt.x, src.y - tgt.y))
        if len(sorted_srcs) < 2:
            continue

        # F7: 枚举所有 (s_i, s_j) 源对（按距离排序），最多 SWARM_MAX_SOURCE_PAIRS 个
        # 用 +1 兵力试 ETA，节省探测开销；真实派船数仍按 total_needed 拆分。
        probe_etas = {}
        for s in sorted_srcs[:5]:  # 仅前 5 颗最近源做 ETA 探测，控制成本
            min_g = world.dynamic_min_garrison.get(s.id, world.min_garrison_base)
            avail = max(0, int(s.ships) - min_g)
            if avail < 2:
                continue
            aim = world.get_intercept(s.id, tid, max(int(tgt.ships) + 1, 5))
            if aim is None:
                continue
            probe_etas[s.id] = (aim[1], avail, s)

        pair_count = 0
        for i in range(len(sorted_srcs)):
            if pair_count >= SWARM_MAX_SOURCE_PAIRS:
                break
            si = sorted_srcs[i]
            if si.id not in probe_etas:
                continue
            eta_i, avail_i, _ = probe_etas[si.id]
            for j in range(i + 1, len(sorted_srcs)):
                if pair_count >= SWARM_MAX_SOURCE_PAIRS:
                    break
                sj = sorted_srcs[j]
                if sj.id not in probe_etas:
                    continue
                eta_j, avail_j, _ = probe_etas[sj.id]
                if abs(eta_i - eta_j) > SWARM_ETA_GAP_MAX:
                    continue
                total_needed = int(tgt.ships) + 8
                splits = [
                    (int(total_needed * 0.5), int(total_needed * 0.5)),
                    (int(total_needed * 0.6), int(total_needed * 0.4)),
                    (int(total_needed * 0.4), int(total_needed * 0.6)),
                ]
                emitted = False
                for send1, send2 in splits:
                    if send1 > avail_i or send2 > avail_j or send1 <= 0 or send2 <= 0:
                        continue
                    a1 = make_atom(si, tgt, send1, world)
                    a2 = make_atom(sj, tgt, send2, world)
                    if a1 is None or a2 is None:
                        continue
                    if abs(a1.eta - a2.eta) > DUAL_ETA_TOLERANCE:
                        continue
                    a1.value *= 1.2
                    a2.value *= 1.2
                    dual_candidates.append([a1, a2])
                    emitted = True
                    if len(dual_candidates) >= MAX_DUAL_CANDIDATES * 2:  # F7: 上限略放宽
                        break
                if emitted:
                    pair_count += 1
                if len(dual_candidates) >= MAX_DUAL_CANDIDATES * 2:
                    break
            if len(dual_candidates) >= MAX_DUAL_CANDIDATES * 2:
                break
        if len(dual_candidates) >= MAX_DUAL_CANDIDATES * 2:
            break

    candidates.extend(dual_candidates)
    cache[player_id] = candidates
    return candidates


# ============================================================
# Lookahead evaluation (unchanged)
# ============================================================

def _atom_to_action(atom, player_id):
    return (player_id, atom.target_id, atom.eta, atom.ships)


def _sample_opponent_actions(world, opp_id, temperature=OPP_STYLE_TEMPERATURE, count=OPP_SAMPLE_COUNT):
    candidates = generate_candidates(world, opp_id, max_sets=world.beam_width)
    if not candidates:
        return [ [] ]
    scores = [immediate_value(c) for c in candidates]
    max_score = max(scores) if scores else 1.0
    exp_scores = [math.exp((s - max_score) / max(temperature, 0.1)) for s in scores]
    total = sum(exp_scores)
    if total <= 0:
        return [candidates[0]]
    probs = [e / total for e in exp_scores]
    sampled = random.choices(candidates, weights=probs, k=count)
    unique = []
    for s in sampled:
        if s not in unique:
            unique.append(s)
    if not unique:
        unique.append(candidates[0])
    return unique


def evaluate(world, my_atoms, deadline):
    my_acts = [_atom_to_action(a, world.my_id) for a in my_atoms]
    all_scores = []
    for opp_id in world.opponent_ids:
        if time.perf_counter() > deadline:
            break
        opp_samples = _sample_opponent_actions(world, opp_id, count=OPP_SAMPLE_COUNT)
        for opp_atoms in opp_samples:
            full_acts = list(my_acts) + [_atom_to_action(a, opp_id) for a in opp_atoms]
            enemy_arrivals = _build_enemy_arrivals(world, full_acts)
            timelines = project_state(world, full_acts)
            dv = delta_V(timelines, world, world.my_id, arrivals_breakdown=enemy_arrivals)
            all_scores.append(dv)
    if not all_scores:
        return -float("inf")
    return sum(all_scores) / len(all_scores)


def immediate_value(atoms):
    return sum(a.value for a in atoms)


# ============================================================
# Agent entry point (unchanged)
# ============================================================

def materialize_actions(atoms):
    return [[a.src_id, float(a.angle), int(a.ships)] for a in atoms]


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def build_world(obs):
    player = _read(obs, "player", 0)
    step = int(_read(obs, "step", 0) or 0)
    base_omega = float(_read(obs, "angular_velocity", 0.0) or 0.0)
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    initial_planets = _read(obs, "initial_planets", []) or []
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    return WorldState(player, step, planets, fleets, initial_planets, base_omega,
                      comets, comet_ids)


def agent(obs, config=None):
    start = time.perf_counter()
    timeout = _read(config, "actTimeout", 1.0) if config else 1.0
    deadline = start + min(SOFT_DEADLINE, max(0.5, timeout * 0.85))

    try:
        world = build_world(obs)
    except Exception:
        return []

    my_planets = [p for p in world.planet_list if p.owner == world.my_id]
    if not my_planets:
        return []

    candidates = generate_candidates(world, world.my_id)
    if not candidates:
        return []

    candidates_sorted = sorted(candidates, key=lambda c: -immediate_value(c))
    fallback = candidates_sorted[0] if candidates_sorted else []

    best = []
    try:
        best_score = evaluate(world, [], deadline)
    except Exception:
        best_score = -float("inf")
        best = fallback

    for cand in candidates_sorted:
        if not cand:
            continue
        if time.perf_counter() > deadline - 0.05:
            break
        try:
            score = evaluate(world, cand, deadline)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best = cand

    final_moves = materialize_actions(best)

    # 焦土撤退
    remaining_time = deadline - time.perf_counter()
    if remaining_time > 0.05:
        safe_planets = [p for p in my_planets if not world.is_doomed(p.id)]
        if safe_planets:
            for planet in my_planets:
                doomed_turn = world.is_doomed(planet.id)
                if doomed_turn is None:
                    continue
                already_acting = any(m[0] == planet.id for m in final_moves)
                if already_acting:
                    continue
                garrison = world.dynamic_min_garrison.get(planet.id, 0)
                evacuable = int(planet.ships) - garrison
                if evacuable <= 0:
                    continue
                dest = min(safe_planets, key=lambda p: math.hypot(planet.x - p.x, planet.y - p.y))
                if dest.id == planet.id:
                    continue
                aim = compute_intercept_with_detour(planet, dest, evacuable, world)
                if aim is None:
                    continue
                angle, eta, _ = aim
                final_moves.append([planet.id, float(angle), evacuable])

    return final_moves