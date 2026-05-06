"""
Orbit Wars v2 – [FULL RULE BOOST + SNIPE + DYNAMIC MODE]
整合全部基于规则的改进：
  - P0: 安全边际 + 动态防守 + 对手采样
  - P1-1: 多源协同攻击（双源组合）
  - P1-2: 自适应束宽度（低价值原子提前截断）
  - P1-3: 终局动态权重（后期舰船加成）
  - P1-4: 防御性焦土（放弃注定失守的行星，撤退舰船）
  - P1-5: 行星保卫价值（附近友军加成）
  - P2-1: 精确时机狙击（抢在敌人到达前1回合夺取中立星）
  - P2-2: 动态模式切换（根据实力比调整攻击性、防守底线等）

保留所有已验证的核心机制：精确物理、迭代拦截、绕路重试、舰队扣除模拟、
早期中立优先、防重复派遣、智能余量、彗星寿命处理等。
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

# 策略超参数（默认值，会根据实力比动态调整）
HORIZON_VALUE = 120
HORIZON_SIM = 80
BEAM_WIDTH = 6
MAX_TARGETS_PER_SRC = 4
MAX_SOURCES = 6
FRONTIER_BONUS = 4.0
PLANET_OWN_BONUS = 2.0
DETOUR_OFFSETS_DEG = (5, -5, 10, -10, 18, -18)
SOFT_DEADLINE = 0.85
OPP_TIME_FRACTION = 0.55
OPP_MAX_EVAL = 5
MIN_GARRISON_BASE = 3

# 早期策略（动态调整时可能被覆盖）
EARLY_GAME_LIMIT = 40
EARLY_NEUTRAL_BONUS = 2.0
EARLY_ENEMY_PENALTY = 0.5
COVERAGE_SHIP_MARGIN = 5

# P0 安全边际
SAFETY_MARGIN_THRESHOLD = 0.3
SAFETY_MIN_ABSOLUTE = 5
SAFETY_PENALTY_FACTOR = 0.7
OPP_SAMPLE_COUNT = 2
OPP_STYLE_TEMPERATURE = 1.5

# P1-1 多源协同（动态调整）
DUAL_SOURCE_MIN_SHIPS = 10
DUAL_ETA_TOLERANCE = 2
MAX_DUAL_CANDIDATES = 3

# P1-2 自适应束宽度
BEAM_CUTOFF_RATIO = 0.25

# P1-3 终局动态权重
LATE_GAME_THRESHOLD = 40
LATE_SHIP_BONUS_STEP = 0.02

# P1-4 防御性焦土
DOOMED_FALL_TURN = 8
DOOMED_EVAC_RATIO = 1.0

# P1-5 行星保卫价值
PAL_GUARD_RADIUS = 25.0
PAL_GUARD_BONUS = 0.5

# P2-1 精确时机狙击
SNIPE_ENABLED = True
SNIPE_VALUE_MULTIPLIER = 2.0

# 价值折现
VALUE_DISCOUNT = 0.985


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
# Physics layer (unchanged from v2 fixed)
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
# WorldState
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

        # 计算实力比，用于动态模式切换
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
            # 落后：更激进
            self.early_neutral_bonus = EARLY_NEUTRAL_BONUS * 1.5
            self.early_enemy_penalty = EARLY_ENEMY_PENALTY * 0.8
            self.dual_source_min_ships = max(5, DUAL_SOURCE_MIN_SHIPS - 5)
            self.beam_width = min(BEAM_WIDTH + 2, 8)
            self.min_garrison_base = max(1, MIN_GARRISON_BASE - 1)
            self.snipe_enabled = True
        elif strength_ratio > 1.5:
            # 领先：更保守
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

        # 动态防守底线（基于敌我距离）
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
        min_eta = float("inf")
        for opp_id in self.opponent_ids:
            for opp_p in self.planet_list:
                if opp_p.owner != opp_id:
                    continue
                d = math.hypot(opp_p.x - planet.x, opp_p.y - planet.y)
                speed = fleet_speed(max(1, opp_p.ships))
                eta = d / speed if speed > 0 else float("inf")
                min_eta = min(min_eta, eta)
        if min_eta <= 15:
            return max(8, int(planet.ships * 0.5))
        elif min_eta <= 30:
            return max(5, int(planet.ships * 0.3))
        else:
            return self.min_garrison_base

    def _project_base_timelines(self):
        arrivals = defaultdict(list)
        for fid, (eta, tid) in self.fleet_arrivals.items():
            if tid is None:
                continue
            f = self.fleet_by_id[fid]
            arrivals[tid].append((eta, f.owner, f.ships))
        timelines = {}
        for pid, p in self.planets.items():
            timelines[pid] = simulate_planet_timeline(
                p, arrivals.get(pid, []), DOOMED_FALL_TURN + 5
            )
        return timelines

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
# Forward simulation
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
# Value function (with all enhancements)
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
    for opp in world.opponent_ids:
        v_opp = _V_full(timelines, world, opp, HORIZON_VALUE)
        if v_opp > v_opp_max:
            v_opp_max = v_opp
    return v_me - v_opp_max


# ============================================================
# Candidate generation (with all P1 + P2 enhancements)
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

def global_target_ranking(world, player_id, top_k=8):
    targets = []

    for p in world.planet_list:
        if p.owner == player_id:
            continue

        # 距离最近我方星球
        min_d = float("inf")
        for myp in world.planet_list:
            if myp.owner == player_id:
                d = math.hypot(myp.x - p.x, myp.y - p.y)
                min_d = min(min_d, d)

        if min_d == float("inf"):
            continue

        cost = p.ships + 1
        score = p.production / (cost * 0.5 + min_d + 5)

        if p.owner == -1:
            score *= 1.2

        targets.append((score, p.id))

    targets.sort(reverse=True)
    return [tid for _, tid in targets[:top_k]]


def ship_options(src, tgt, max_ships):
    if max_ships <= 0:
        return []

    target_ships = max(1, int(tgt.ships))

    options = set()

    # 精确击杀
    options.add(min(max_ships, target_ships + 2))

    # 稳定击杀
    options.add(min(max_ships, target_ships + 6))

    # 全力
    options.add(max_ships)

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
    production_gain = tgt.production * max(0, remaining - int(eta))
    cost = ships + int(eta) * 0.5
    value = production_gain / (cost + 1.0)
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

    # 单源原子生成
    atoms = []
    for src in my_planets:
        min_garrison = world.dynamic_min_garrison.get(src.id, world.min_garrison_base)
        max_send = int(src.ships) - min_garrison
        if max_send <= 0:
            continue
        global_targets = global_target_ranking(world, player_id)

        targets = [tid for tid in global_targets[:MAX_TARGETS_PER_SRC]]
        for tid in targets:
            tgt = world.planets[tid]
            for ships in ship_options(src, tgt, max_send):
                atom = make_atom(src, tgt, ships, world)
                if atom:
                    atoms.append(atom)

    # ======= P2-1 精确时机狙击 =======
    if world.snipe_enabled and player_id == world.my_id:  # 仅对我方启用狙击
        for tid, tgt in world.planets.items():
            if tgt.owner != -1 or tid in world.covered_neutrals:
                continue
            if tid in world.comet_ids:
                life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
                if life <= 2:
                    continue
            # 计算敌方到达最早eta
            enemy_eta = float("inf")
            for fid, (eta, t_id) in world.fleet_arrivals.items():
                if t_id == tid and world.fleet_by_id[fid].owner != player_id:
                    if eta < enemy_eta:
                        enemy_eta = eta
            if enemy_eta == float("inf") or enemy_eta < 2:
                continue
            need_ships = int(tgt.ships) + 2
            # 寻找能比敌人早至少1回合到达的源
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
                    # 狙击成功，生成高价值原子
                    remaining = HORIZON_VALUE
                    if tid in world.comet_ids:
                        life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
                        remaining = min(remaining, life)
                    production_gain = tgt.production * max(0, remaining - int(eta))
                    cost = need_ships + int(eta) * 0.5
                    value = (production_gain / (cost + 1.0)) * SNIPE_VALUE_MULTIPLIER
                    snipe_atom = Atom(src.id, tid, need_ships, angle, eta, value)
                    atoms.append(snipe_atom)
                    break  # 一个目标只找一个源狙击

    # 按价值排序
    atoms.sort(key=lambda a: -a.value)

    candidates = [[]]
    if not atoms:
        cache[player_id] = candidates
        return candidates

    best_val = atoms[0].value
    selected_atoms = []
    committed = defaultdict(int)

    for atom in atoms:
        # 自适应截断
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

    # ======= P1-1 多源协同攻击 =======
    dual_candidates = []
    for tid, tgt in world.planets.items():
        if tgt.owner == player_id:
            continue
        if tgt.ships < world.dual_source_min_ships and tgt.owner == -1:
            continue
        remaining_life = HORIZON_VALUE
        if tid in world.comet_ids:
            life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
            remaining_life = min(remaining_life, life)
        if remaining_life < 15:
            continue

        closest = sorted(my_planets, key=lambda src: math.hypot(src.x - tgt.x, src.y - tgt.y))
        if len(closest) < 2:
            continue
        s1, s2 = closest[0], closest[1]
        avail1 = max(0, int(s1.ships) - world.dynamic_min_garrison.get(s1.id, world.min_garrison_base))
        avail2 = max(0, int(s2.ships) - world.dynamic_min_garrison.get(s2.id, world.min_garrison_base))
        if avail1 < 2 or avail2 < 2:
            continue

        total_needed = int(tgt.ships) + 8
        splits = [(int(total_needed * 0.5), int(total_needed * 0.5)),
                  (int(total_needed * 0.6), int(total_needed * 0.4)),
                  (int(total_needed * 0.4), int(total_needed * 0.6))]
        for send1, send2 in splits:
            if send1 > avail1 or send2 > avail2 or send1 <= 0 or send2 <= 0:
                continue
            a1 = make_atom(s1, tgt, send1, world)
            a2 = make_atom(s2, tgt, send2, world)
            if a1 is None or a2 is None:
                continue
            if abs(a1.eta - a2.eta) > DUAL_ETA_TOLERANCE:
                continue
            a1.value *= 1.2
            a2.value *= 1.2
            dual_candidates.append([a1, a2])
            if len(dual_candidates) >= MAX_DUAL_CANDIDATES:
                break
        if len(dual_candidates) >= MAX_DUAL_CANDIDATES:
            break

    candidates.extend(dual_candidates)
    cache[player_id] = candidates
    return candidates


# ============================================================
# Lookahead evaluation
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

    worst_case_scores = []

    for opp_id in world.opponent_ids:
        if time.perf_counter() > deadline:
            break

        opp_samples = _sample_opponent_actions(world, opp_id, count=OPP_SAMPLE_COUNT)

        local_scores = []

        for opp_atoms in opp_samples:
            # ===== step 1 =====
            full_acts = list(my_acts) + [_atom_to_action(a, opp_id) for a in opp_atoms]

            enemy_arrivals = _build_enemy_arrivals(world, full_acts)
            timelines = project_state(world, full_acts)

            # ===== step 2（关键）=====
            # 构造下一状态 world（简化版）
            pseudo_world = world  # 这里偷懒，不重建world（速度优先）

            next_candidates = generate_candidates(pseudo_world, world.my_id)

            best_response = -float("inf")

            for next_atoms in next_candidates[:3]:  # 控制复杂度
                next_acts = full_acts + [_atom_to_action(a, world.my_id) for a in next_atoms]

                timelines2 = project_state(world, next_acts)

                dv = delta_V(timelines2, world, world.my_id, arrivals_breakdown=enemy_arrivals)

                if dv > best_response:
                    best_response = dv

            local_scores.append(best_response)

        if local_scores:
            # 🔥 Minimax（核心）
            worst_case_scores.append(min(local_scores))

    if not worst_case_scores:
        return -float("inf")

    # 混合（稳定性更好）
    return 0.7 * min(worst_case_scores) + 0.3 * (sum(worst_case_scores) / len(worst_case_scores))


def immediate_value(atoms):
    return sum(a.value for a in atoms)


# ============================================================
# Agent entry point
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

    # ======= P1-4 防御性焦土 =======
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