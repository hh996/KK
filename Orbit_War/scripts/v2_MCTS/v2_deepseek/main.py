"""
Orbit Wars - MCTS Enhanced with Fleet State, Node Reuse, Terminal Value, Comet Pounce
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

# Strategy params (defaults, may be overridden dynamically)
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

EARLY_GAME_LIMIT = 40
EARLY_NEUTRAL_BONUS = 2.0
EARLY_ENEMY_PENALTY = 0.5

SAFETY_MARGIN_THRESHOLD = 0.3
SAFETY_MIN_ABSOLUTE = 5
SAFETY_PENALTY_FACTOR = 0.7
OPP_SAMPLE_COUNT = 2
OPP_STYLE_TEMPERATURE = 1.5

DUAL_SOURCE_MIN_SHIPS = 10
DUAL_ETA_TOLERANCE = 1
MAX_DUAL_CANDIDATES = 3

BEAM_CUTOFF_RATIO = 0.25

LATE_GAME_THRESHOLD = 40          # Start increasing ship weight
LATE_SHIP_BONUS_STEP = 0.02
TERMINAL_TURNS = 30              # After this, purely maximize ships

DOOMED_FALL_TURN = 8
DOOMED_EVAC_RATIO = 1.0

PAL_GUARD_RADIUS = 25.0
PAL_GUARD_BONUS = 0.5

SNIPE_ENABLED = True
SNIPE_VALUE_MULTIPLIER = 2.0

VALUE_DISCOUNT = 0.985

INFLUENCE_DECAY = 0.06
VULTURE_MULT = 2.5
STRONG_ENEMY_PENALTY = 0.6
BREAKEVEN_PENALTY_SCALE = 50.0
DISTANCE_DISCOUNT_SCALE = 30.0
THREAT_BONUS_THRESHOLD = -10.0

EXCESS_PENALTY_FACTOR = 2.0

MIN_ATTACK_RATIO = 0.3
MIN_ACCEPT_RATIO = 1.0
NEUTRAL_ACCEPT_RATIO = 1.0

THIRD_PARTY_SENSE_ETA = 4
THIRD_PARTY_BONUS = 1.25

COMET_EARLY_BONUS = 1.5           # additional multiplier for fast comet capture
COMET_EARLY_LIFE_RATIO = 0.3      # arrive within first 30% of comet life



# ============================================================
# Data Classes
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
# Physics (unchanged from previous full version)
# ============================================================
# (Keep all physics functions: fleet_speed, get_launch_position, _angle_norm,
#  line_hits_circle, line_hits_sun, is_orbital, estimate_signed_omega,
#  build_omega_map, predict_orbit_position, predict_comet_position,
#  predict_target_position, compute_intercept, path_blocked_by_other_planet,
#  trace_intercept, compute_intercept_with_detour, predict_fleet_arrival)

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
# Influence Map & Player Analysis
# ============================================================

def compute_influence_map(planets, player_id, omega_map, cid_to_group, comet_ids):
    n = len(planets)
    influence = [0.0] * n
    planet_list = list(planets.values()) if isinstance(planets, dict) else planets
    for i, p in enumerate(planet_list):
        for j, q in enumerate(planet_list):
            if i == j:
                continue
            dist = math.hypot(p.x - q.x, p.y - q.y)
            approx_time = dist / fleet_speed(15) + 1.0
            decay = math.exp(-INFLUENCE_DECAY * approx_time)
            ships = q.ships
            if q.owner == player_id:
                influence[i] += ships * decay
            elif q.owner != -1:
                influence[i] -= ships * decay
    return influence

def analyze_players(planets, my_id):
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
# WorldState (enhanced)
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

        # 玩家集合
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

        # 舰队到达预测
        self.fleet_arrivals = {}
        for f in fleets:
            self.fleet_arrivals[f.id] = predict_fleet_arrival(
                f, planets, self.omega_map, self.cid_to_group, self.comet_ids
            )

        # 我方已在途舰队统计
        self.my_incoming = defaultdict(float)
        self.my_incoming_max_eta = {}
        for fid, (eta, tid) in self.fleet_arrivals.items():
            if tid is None:
                continue
            f = self.fleet_by_id[fid]
            if f.owner == player:
                self.my_incoming[tid] += f.ships
                prev = self.my_incoming_max_eta.get(tid, 0)
                self.my_incoming_max_eta[tid] = max(prev, eta)

        # 敌方已在途舰队统计
        self.enemy_incoming_by_target = defaultdict(lambda: defaultdict(float))
        for fid, (eta, tid) in self.fleet_arrivals.items():
            if tid is None:
                continue
            f = self.fleet_by_id[fid]
            if f.owner != player and f.owner != -1:
                turn = max(1, int(math.ceil(eta)))
                self.enemy_incoming_by_target[tid][turn] += f.ships

        # 已覆盖的中立星（已派遣足够占领）
        self.covered_neutrals = set()
        for pid, p in self.planets.items():
            if p.owner != -1:
                continue
            if self.my_incoming.get(pid, 0) >= p.ships + 1:
                self.covered_neutrals.add(pid)

        # 实力统计
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
        self.my_total = my_total
        self.enemy_total = enemy_total
        strength_ratio = my_total / max(1, enemy_total)

        # ========== 终局与绝对优势标志 ==========
        self.is_terminal = self.remaining_steps <= TERMINAL_TURNS
        self.is_absolute_dominant = (
            self.is_terminal and
            strength_ratio > 3.0 and
            len(self.opponent_ids) == 1
        )

        # ========== 动态攻势系数 ==========
        if self.is_absolute_dominant:
            # 终局碾压模式：防守底线接近零，攻击需求极低
            self.aggression = 0.1
            self.attack_discount = 0.5
        elif strength_ratio > 3.0:
            self.aggression = 0.15
            self.attack_discount = 0.6
        elif strength_ratio > 2.0:
            self.aggression = 0.25
            self.attack_discount = 0.7
        elif strength_ratio > 1.5:
            self.aggression = 0.4
            self.attack_discount = 0.8
        elif strength_ratio > 1.2:
            self.aggression = 0.6
            self.attack_discount = 0.9
        elif strength_ratio > 0.8:
            self.aggression = 1.0
            self.attack_discount = 1.0
        elif strength_ratio > 0.6:
            self.aggression = 1.3
            self.attack_discount = 1.0
        else:
            self.aggression = 1.8
            self.attack_discount = 1.0

        # 早期/晚期偏好调整
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
        self.influence = compute_influence_map(self.planets, self.my_id, self.omega_map, self.cid_to_group, self.comet_ids)
        self.influence_by_id = {}
        for idx, p in enumerate(self.planet_list):
            self.influence_by_id[p.id] = self.influence[idx]

        # 玩家分析
        self.player_analysis = analyze_players(self.planets, self.my_id)

        # 动态防守底线（调用 _compute_dynamic_min）
        self.dynamic_min_garrison = {}
        for p in planets:
            if p.owner == player:
                self.dynamic_min_garrison[p.id] = self._compute_dynamic_min(p)
            else:
                self.dynamic_min_garrison[p.id] = 0

        # 基础时间线（用于 doomed 判断）
        self.projected_timelines = self._project_base_timelines()

        # 缓存
        self._intercept_cache = {}
        self._top_targets_cache = {}
        self._candidate_cache = {}
        self._best_response_cache = {}
        self._extra_plans = None

    def _compute_dynamic_min(self, planet):
        if self.is_absolute_dominant:
            return 0                      # 终局碾压不留守军
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
        infl = self.influence_by_id.get(planet.id, 0.0)
        threat_bonus = max(0, -infl * 0.5)
        if min_eta <= 15:
            base = max(8, int(planet.ships * 0.5))
        elif min_eta <= 30:
            base = max(5, int(planet.ships * 0.3))
        else:
            base = self.min_garrison_base
        base = min(int(planet.ships * 0.7), base + int(threat_bonus))
        return max(1, int(base * self.aggression))

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
    terminal = remaining < TERMINAL_TURNS

    if terminal:
        # Only count final ships (no production discounting)
        for pid, tl in timelines.items():
            p = world.planets[pid]
            owner, ships = tl[horizon]
            if owner == player:
                total += ships
                # small frontier/guard bonuses still apply
                if enemy_positions:
                    min_d = min(math.hypot(p.x - ex, p.y - ey) for ex, ey in enemy_positions)
                    if min_d < 40:
                        total += FRONTIER_BONUS * p.production / (min_d + 5.0)
                nearby_friends = sum(1 for fx, fy in friend_positions
                                     if math.hypot(p.x - fx, p.y - fy) <= PAL_GUARD_RADIUS and (fx != p.x or fy != p.y))
                total += p.production * PAL_GUARD_BONUS * min(3, nearby_friends)
        return total

    late_mult = 1.0 + max(0, (LATE_GAME_THRESHOLD - remaining) * LATE_SHIP_BONUS_STEP)

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
# Fleet advancement helper for MCTS
# ============================================================

def advance_fleet(fleet, turns, planets, omega_map, cid_to_group, comet_ids):
    """Move fleet forward by `turns` steps. Returns (new_x, new_y) or None if destroyed."""
    speed = fleet_speed(fleet.ships)
    cos_a, sin_a = math.cos(fleet.angle), math.sin(fleet.angle)
    prev_x, prev_y = fleet.x, fleet.y
    for k in range(1, turns + 1):
        nx = fleet.x + k * speed * cos_a
        ny = fleet.y + k * speed * sin_a
        if not (0 <= nx <= BOARD_SIZE and 0 <= ny <= BOARD_SIZE):
            return None
        if line_hits_sun(prev_x, prev_y, nx, ny):
            return None
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
                return None
        prev_x, prev_y = nx, ny
    return nx, ny

# ============================================================
# MCTS with fleet state + node reuse
# ============================================================

# Global state for node reuse across turns
_mcts_state = {
    "last_root": None,
    "chosen_child_key": None,
    "last_step": -1,
    "last_world_hash": None
}

def world_fingerprint(world):
    """
    轻量化世界指纹，用于判定节点复用是否有效。
    包含行星：id, owner, ships, 移动行星的当前坐标（小数点后2位）
    包含舰队：总数、总舰船
    包含彗星：每颗彗星的剩余寿命
    """
    sig = []
    for p in sorted(world.planet_list, key=lambda p: p.id):
        # 轨道行星与彗星坐标变化快，格外记录位置
        if p.id in world.comet_ids or abs(world.omega_map.get(p.id, 0.0)) > 1e-9:
            pos = (round(p.x, 2), round(p.y, 2))
        else:
            pos = None
        sig.append((p.id, p.owner, int(p.ships), pos))
    # 舰队概览
    fleet_count = len(world.fleets)
    fleet_total = sum(f.ships for f in world.fleets)
    sig.append(('fleets', fleet_count, fleet_total))
    # 彗星寿命（如果彗星正在场上）
    for g in world.cid_to_group.values():
        for pid in g.get("planet_ids", []):
            life = comet_remaining_turns(g, pid)
            sig.append(('comet_life', pid, life))
    return tuple(sig)

def fast_forward_world(world, my_atoms, advance_turns=1):
    """
    推进世界 advance_turns 回合。
    保留彗星状态与轨道运动，新舰队通过 advance_fleet 碰撞检测。
    """
    all_actions = []
    for a in my_atoms:
        all_actions.append(_atom_to_action(a, world.my_id))

    # 对手采用规则最优快速响应（可保持确定性，也可后续增加采样）
    for opp_id in world.opponent_ids:
        opp_cands = generate_candidates(world, opp_id, max_sets=1)
        if opp_cands and opp_cands[0]:
            for a in opp_cands[0]:
                all_actions.append(_atom_to_action(a, opp_id))

    timelines = project_state(world, all_actions)

    # 更新行星——取时间线最后的状态
    new_planet_map = {}
    for pid, p in world.planets.items():
        tl = timelines.get(pid)
        if tl:
            owner, ships = tl[-1]
            new_planet_map[pid] = Planet(
                pid, owner, p.x, p.y, p.radius, max(0, int(ships)), p.production
            )
        else:
            new_planet_map[pid] = p
    new_planets = list(new_planet_map.values())

    # 舰队 ID 计数
    fleet_id_counter = max((f.id for f in world.fleets), default=0) + 1

    # 前进已有舰队
    new_fleets = []
    for f in world.fleets:
        pos = advance_fleet(f, advance_turns, new_planets, world.omega_map,
                            world.cid_to_group, world.comet_ids)
        if pos:
            new_fleets.append(Fleet(f.id, f.owner, pos[0], pos[1], f.angle, f.from_planet_id, f.ships))

    # 我方新发射舰队
    for a in my_atoms:
        src = world.planets[a.src_id]
        lx, ly = get_launch_position(src, a.angle)
        temp_f = Fleet(fleet_id_counter, world.my_id, lx, ly, a.angle, a.src_id, a.ships)
        fleet_id_counter += 1
        pos = advance_fleet(temp_f, advance_turns, new_planets, world.omega_map,
                            world.cid_to_group, world.comet_ids)
        if pos:
            new_fleets.append(Fleet(temp_f.id, temp_f.owner, pos[0], pos[1],
                                    temp_f.angle, temp_f.from_planet_id, temp_f.ships))

    # 对手新发射舰队
    for opp_id in world.opponent_ids:
        opp_cands = generate_candidates(world, opp_id, max_sets=1)
        if opp_cands and opp_cands[0]:
            for a in opp_cands[0]:
                src = world.planets[a.src_id]
                lx, ly = get_launch_position(src, a.angle)
                temp_f = Fleet(fleet_id_counter, opp_id, lx, ly, a.angle, a.src_id, a.ships)
                fleet_id_counter += 1
                pos = advance_fleet(temp_f, advance_turns, new_planets, world.omega_map,
                                    world.cid_to_group, world.comet_ids)
                if pos:
                    new_fleets.append(Fleet(temp_f.id, temp_f.owner, pos[0], pos[1],
                                            temp_f.angle, temp_f.from_planet_id, temp_f.ships))

    # 彗星路径推进
    new_comets = []
    for g in world.cid_to_group.values():
        grp = dict(g)
        grp["path_index"] = g["path_index"] + advance_turns
        new_comets.append(grp)
    new_comet_ids = set(world.comet_ids)

    # 组合新世界
    new_world = WorldState(
        player=world.my_id,
        step=world.step + advance_turns,
        planets=new_planets,
        fleets=new_fleets,
        initial_planets=[],         # 非必须，但 WorldState 的 omega 构建不依赖它（后面手动覆盖）
        base_omega=0.0,
        comets=new_comets,
        comet_ids=new_comet_ids
    )
    # 直接继承原 omega_map，避免轨道预测失效
    new_world.omega_map = world.omega_map
    return new_world


class MCTSNode:
    __slots__ = ('world', 'atoms', 'parent', 'children', 'visits', 'total_value', 'untried_actions')
    def __init__(self, world, atoms=None, parent=None):
        self.world = world
        self.atoms = atoms if atoms is not None else []
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = None

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c=1.4):
        best = None
        best_score = -float('inf')
        for child in self.children.values():
            exploit = child.total_value / max(1, child.visits)
            explore = c * math.sqrt(math.log(self.visits + 1) / max(1, child.visits))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def expand(self):
        if self.untried_actions is None:
            self.untried_actions = generate_candidates(self.world, self.world.my_id)
            if [] not in self.untried_actions:
                self.untried_actions.append([])
            random.shuffle(self.untried_actions)
        if not self.untried_actions:
            return None
        action_atoms = self.untried_actions.pop()
        next_world = fast_forward_world(self.world, action_atoms, advance_turns=1)
        child = MCTSNode(next_world, atoms=action_atoms, parent=self)
        key = tuple((a.src_id, a.target_id, a.ships, a.eta) for a in action_atoms)
        self.children[key] = child
        return child

    def rollout(self, depth=2, deadline=None):
        world = self.world
        total_atoms = list(self.atoms)
        for _ in range(depth):
            if deadline and time.perf_counter() > deadline:
                break
            cands = generate_candidates(world, world.my_id)
            if not cands:
                break
            best_cand = max(cands, key=lambda c: immediate_value(c))
            total_atoms.extend(best_cand)
            world = fast_forward_world(world, best_cand, advance_turns=1)
        try:
            return evaluate(world, total_atoms, deadline or time.perf_counter() + 0.1)
        except Exception:
            return immediate_value(total_atoms)

def mcts_search(init_world, iterations=200, max_seconds=0.8):
    global _mcts_state
    deadline = time.perf_counter() + max_seconds
    root = None

    # 节点复用：如果上一回合选择了某个子节点，且当前世界匹配该子节点世界，则复用
    last_root = _mcts_state.get("last_root")
    chosen_key = _mcts_state.get("chosen_child_key")
    if last_root and chosen_key is not None:
        expected_child = last_root.children.get(chosen_key)
        if expected_child and world_fingerprint(init_world) == world_fingerprint(expected_child.world):
            root = expected_child
            root.parent = None
            root.untried_actions = None   # 重新生成候选，因为环境可能有变化
            root.children = {}
            root.visits = 0
            root.total_value = 0.0

    if root is None:
        root = MCTSNode(init_world)

    # 将 root 存储到 world 对象上，供规则回退时参考
    init_world._mcts_root = root

    # 第一阶段：快速浅搜索（depth=2），大量迭代确立主方向
    shallow_iter = min(150, iterations)
    for _ in range(shallow_iter):
        if time.perf_counter() > deadline - 0.15:
            break
        node = root
        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            if node is None:
                break
        # Expansion
        if not node.is_fully_expanded():
            new_node = node.expand()
            if new_node is not None:
                node = new_node
        # Simulation
        value = node.rollout(depth=2, deadline=deadline)
        # Backpropagation
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    # 第二阶段：余时加深搜索（depth=3），进一步拓深关键路径
    if time.perf_counter() < deadline - 0.1:
        extra_iter = iterations - shallow_iter
        for _ in range(extra_iter):
            if time.perf_counter() > deadline - 0.05:
                break
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                if node is None:
                    break
            if not node.is_fully_expanded():
                new_node = node.expand()
                if new_node is not None:
                    node = new_node
            value = node.rollout(depth=3, deadline=deadline)
            while node is not None:
                node.visits += 1
                node.total_value += value
                node = node.parent

    # 保存复用信息
    if root.children:
        best_child = max(root.children.values(), key=lambda c: c.visits)
        _mcts_state["last_root"] = root
        _mcts_state["chosen_child_key"] = next(k for k, v in root.children.items() if v == best_child)
    else:
        _mcts_state["last_root"] = None
        _mcts_state["chosen_child_key"] = None
    _mcts_state["last_step"] = init_world.step

    if root.children:
        best_child = max(root.children.values(), key=lambda c: c.visits)
        return best_child.atoms
    return []

# ============================================================
# Precision Deployment + Comet Pounce
# ============================================================

def plan_global_assaults(world, max_plans=5):
    """
    生成多个整体作战方案，每个方案是一组 Atom 列表。
    覆盖：集中攻击一个高价值目标、多路分攻、均衡配置。
    """
    my_id = world.my_id
    my_planets = [p for p in world.planet_list if p.owner == my_id]
    if not my_planets:
        return []

    # 可用兵力池
    available = {}
    for p in my_planets:
        garrison = world.dynamic_min_garrison.get(p.id, 0)
        avail = int(p.ships) - garrison
        if avail > 0:
            available[p.id] = (p, avail)

    if not available:
        return []

    # 目标价值排序（简化）
    target_scores = []
    for pid, tgt in world.planets.items():
        if tgt.owner == my_id or is_saturated(tgt, world):
            continue
        if pid in world.comet_ids:
            life = comet_remaining_turns(world.cid_to_group.get(pid), pid)
            if life <= 1:
                continue
        d = min(math.hypot(p.x - tgt.x, p.y - tgt.y) for p in my_planets)
        ships = tgt.ships
        value = tgt.production / (d * 0.5 + ships + 1.0)
        target_scores.append((value, pid))

    if not target_scores:
        return []

    target_scores.sort(reverse=True)
    targets = [world.planets[pid] for _, pid in target_scores]

    plans = []

    # ---------- 计划1：全力围攻最高价值目标 ----------
    best_target = targets[0]
    plan1 = []
    sources = []
    for pid, (src, avail) in available.items():
        aim = world.get_intercept(pid, best_target.id, min(avail, 100))
        if aim is None:
            continue
        _, eta, _ = aim
        sources.append((eta, src, avail))
    sources.sort()
    # 估算需求（基于最大 eta）
    if sources:
        max_eta = max(s[0] for s in sources[:3]) if len(sources) >= 3 else sources[-1][0]
        needed = needed_for_capture(best_target, max_eta, world)
        used = 0
        for eta, src, avail in sources:
            if used >= needed:
                break
            send = min(avail, needed - used)
            atom = make_atom(src, best_target, send, world)
            if atom:
                plan1.append(atom)
                used += send
    if plan1 and sum(a.ships for a in plan1) >= needed_for_capture(best_target, max(a.eta for a in plan1), world):
        plans.append(plan1)

    # ---------- 计划2：攻取前2-3个高价值目标（各配刚好兵力） ----------
    plan2 = []
    temp_avail = dict(available)
    for target in targets[:3]:
        # 粗略估算需求
        needed = needed_for_capture(target, 5, world)  # 预估5回合
        best_src = None
        best_aim = None
        for pid, (src, avail) in temp_avail.items():
            if avail < needed:
                continue
            aim = world.get_intercept(pid, target.id, needed)
            if aim is None:
                continue
            if best_aim is None or aim[1] < best_aim[1]:
                best_aim = aim
                best_src = (pid, src, avail)
        if best_src:
            pid, src, avail = best_src
            atom = make_atom(src, target, needed, world)
            if atom:
                plan2.append(atom)
                temp_avail[pid] = (src, avail - needed)
    if plan2:
        plans.append(plan2)

    # ---------- 计划3：均衡攻击，允许每个目标由多个源协同 ----------
    plan3 = []
    temp_avail2 = dict(available)
    for target in targets[:2]:
        needed = needed_for_capture(target, 5, world)
        cand_srcs = []
        for pid, (src, avail) in temp_avail2.items():
            aim = world.get_intercept(pid, target.id, min(avail, needed))
            if aim:
                cand_srcs.append((aim[1], pid, src, avail))
        cand_srcs.sort()
        used = 0
        for _, pid, src, avail in cand_srcs:
            if used >= needed:
                break
            send = min(avail, needed - used)
            atom = make_atom(src, target, send, world)
            if atom:
                plan3.append(atom)
                used += send
                temp_avail2[pid] = (src, avail - send)
    if plan3:
        plans.append(plan3)
    
    # ---------- 计划4（新增）：全体压上攻击最强敌方行星 ----------
    my_id = world.my_id
    my_planets = [p for p in world.planet_list if p.owner == my_id]
    available = {}
    for p in my_planets:
        avail = int(p.ships) - world.dynamic_min_garrison.get(p.id, 0)
        if avail > 0:
            available[p.id] = (p, avail)

    if available and target_scores:   # 复用之前 target_scores
        # 找最高价值的敌方行星
        enemy_target = None
        for val, pid in target_scores:
            p = world.planets[pid]
            if p.owner != -1 and p.owner != my_id:
                enemy_target = p
                break
        if enemy_target:
            need = needed_for_capture(enemy_target, 5, world)
            # 收集所有可用源，按距离排序
            sources = []
            for pid, (src, avail) in available.items():
                aim = world.get_intercept(pid, enemy_target.id, min(avail, 100))
                if aim:
                    sources.append((aim[1], pid, src, avail))
            sources.sort()
            used = 0
            group = []
            for _, pid, src, avail in sources:
                if used >= need:
                    break
                send = min(avail, need - used)
                atom = make_atom(src, enemy_target, send, world)
                if atom:
                    atom.value *= 2.0   # 强攻加分
                    group.append(atom)
                    used += send
            if used >= need:
                plans.append(group)

    # 去重、裁剪
    final_plans = []
    seen = set()
    for p in plans:
        key = tuple((a.src_id, a.target_id, a.ships) for a in p)
        if key not in seen:
            seen.add(key)
            final_plans.append(p)
    return final_plans[:max_plans]

def plan_endgame_cleanup(world):
    """终局绝对优势时，将所有可用兵力分配给剩余敌方行星"""
    plans = []
    my_planets = [p for p in world.planet_list if p.owner == world.my_id]
    enemy_planets = [p for p in world.planet_list if p.owner != world.my_id and p.owner != -1]
    if not enemy_planets:
        return plans

    # 收集所有可用兵力（防守底线已为0）
    available = {}
    for p in my_planets:
        avail = int(p.ships)
        if avail > 0:
            available[p.id] = (p, avail)

    if not available:
        return plans

    # 对每个敌方行星，挑选最近的源派兵（至少满足需求）
    atoms = []
    for target in enemy_planets:
        needed = needed_for_capture(target, 5, world)  # eta 不重要，因为需求极低
        # 找最近的源
        best_src = None
        best_dist = float('inf')
        for pid, (src, avail) in available.items():
            d = math.hypot(src.x - target.x, src.y - target.y)
            if d < best_dist and avail >= needed:
                best_dist = d
                best_src = (pid, src, avail)
        if best_src:
            pid, src, avail = best_src
            send = min(avail, needed + 2)  # 稍微多派一点
            atom = make_atom(src, target, send, world)
            if atom:
                atoms.append(atom)
                available[pid] = (src, avail - send)
    if atoms:
        plans.append(atoms)
    return plans


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

def needed_for_capture(target, eta, world):
    if target.owner == -1:
        return target.ships + 1
    # 终局绝对优势：直接以当前驻军为目标，忽略生产及增援
    if getattr(world, 'is_terminal', False) and getattr(world, 'is_absolute_dominant', False):
        already = world.my_incoming.get(target.id, 0)
        need = target.ships + 1
        return max(1, need - already)

    base = target.ships
    prod = target.production * max(0, int(math.ceil(eta)))
    enemy_support = 0.0
    for turn, ships in world.enemy_incoming_by_target.get(target.id, {}).items():
        if turn <= math.ceil(eta):
            enemy_support += ships
    required = max(1, math.ceil(base + prod + enemy_support)) + 1
    already = world.my_incoming.get(target.id, 0)
    needed = max(1, required - already)

    aggress = getattr(world, 'aggression', 1.0)
    attack_discount = getattr(world, 'attack_discount', 1.0)
    if aggress < 1.0:
        needed = max(1, int((target.ships + 1) * attack_discount))
        needed = max(1, needed - already)
    return needed

def is_saturated(target, world):
    if target.owner == world.my_id:
        return True
    # 终局绝对优势时，永远不饱和，鼓励持续出兵
    if getattr(world, 'is_terminal', False) and getattr(world, 'is_absolute_dominant', False):
        return False
    if target.owner == -1:
        return world.my_incoming.get(target.id, 0) >= target.ships + 1
    need = needed_for_capture(target, world.my_incoming_max_eta.get(target.id, 2.0), world)
    return world.my_incoming.get(target.id, 0) >= need

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
        if is_saturated(tgt, world):
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
        infl = world.influence_by_id.get(tid, 0.0)
        cost = max(1, int(tgt.ships) + 1)
        distance_weight = 2.0
        val = tgt.production / (cost * 0.4 + distance_weight * eta + 5.0)
        distance_factor = 1.0 / (1.0 + eta / DISTANCE_DISCOUNT_SCALE)
        val *= distance_factor
        if tgt.owner == -1:
            val *= 1.2
            if infl > 10:
                val *= 1.3
        else:
            if infl < THREAT_BONUS_THRESHOLD:
                val *= 1.5
        # Comet early bonus
        if tid in world.comet_ids:
            life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
            if eta < life * COMET_EARLY_LIFE_RATIO:
                val *= COMET_EARLY_BONUS
        third_party_eta = float("inf")
        for opp_id in world.opponent_ids:
            if opp_id == player:
                continue
            for fid, (feta, ftid) in world.fleet_arrivals.items():
                if ftid == tid and world.fleet_by_id[fid].owner == opp_id:
                    if feta < THIRD_PARTY_SENSE_ETA and feta < third_party_eta:
                        third_party_eta = feta
        if third_party_eta < THIRD_PARTY_SENSE_ETA:
            val *= THIRD_PARTY_BONUS
        if is_early:
            if tgt.owner == -1:
                val *= world.early_neutral_bonus
            else:
                val *= world.early_enemy_penalty
        candidates.append((val, tid))
    candidates.sort(reverse=True)
    result = [tid for _, tid in candidates[:top_k]]
    world._top_targets_cache[cache_key] = result
    return result

def ship_options(src, tgt, max_ships, world, eta_estimate=None, remaining_need=None):
    if max_ships <= 0:
        return []
    if remaining_need is None or remaining_need <= 0:
        return []
    if max_ships < remaining_need:
        return []
    options = set()
    options.add(remaining_need)
    options.add(min(max_ships, remaining_need + 2))
    options.add(min(max_ships, remaining_need + 5))
    return sorted(o for o in options if 1 <= o <= max_ships)

def make_atom(src, tgt, ships, world, eta_precomputed=None):
    aim = world.get_intercept(src.id, tgt.id, ships)
    if aim is None:
        return None
    angle, eta, _detour = aim
    if eta > HORIZON_SIM:
        return None

    # 终局绝对优势 + 敌方行星：直接赋予极高价值，确保立刻被选
    if (getattr(world, 'is_terminal', False) and 
        getattr(world, 'is_absolute_dominant', False) and
        tgt.owner != -1 and tgt.owner != src.owner):
        value = 1000.0 / (eta + 1.0)
        return Atom(src.id, tgt.id, ships, angle, eta, value)

    remaining = HORIZON_VALUE
    if tgt.id in world.comet_ids:
        life = comet_remaining_turns(world.cid_to_group.get(tgt.id), tgt.id)
        remaining = min(remaining, life)

    need = needed_for_capture(tgt, eta, world)
    excess = max(0, ships - need)

    production_gain = tgt.production * max(0, remaining - int(eta))
    effective_cost = need + excess * EXCESS_PENALTY_FACTOR
    roi = production_gain / (effective_cost * (eta + 1.0)) if effective_cost > 0 else 0.0

    breakeven = effective_cost / max(tgt.production, 0.1)
    breakeven_penalty = 1.0 / (1.0 + breakeven / BREAKEVEN_PENALTY_SCALE)
    distance_penalty = 1.0 / (1.0 + eta / DISTANCE_DISCOUNT_SCALE)

    infl = world.influence_by_id.get(tgt.id, 0.0)
    threat_mult = 1.0
    if infl < THREAT_BONUS_THRESHOLD and tgt.owner != -1:
        threat_mult = 1.5

    vuln_mult = 1.0
    if tgt.owner != -1 and tgt.owner != src.owner:
        owner_info = world.player_analysis.get(tgt.owner, {})
        if owner_info.get("is_weak"):
            vuln_mult = VULTURE_MULT
        elif owner_info.get("is_strong"):
            vuln_mult = STRONG_ENEMY_PENALTY

    is_early = world.step < EARLY_GAME_LIMIT
    is_late = world.remaining_steps < LATE_GAME_THRESHOLD
    if is_early:
        phase_mult = 2.0 if tgt.owner == -1 else 0.4
    elif is_late:
        phase_mult = 1.5 if tgt.owner != -1 else 0.8
    else:
        phase_mult = 1.0

    capture_bonus = 1.0
    if tgt.owner != -1 and tgt.owner != src.owner:
        capture_bonus = 1.15

    # 优势终结加分（非终局但已占优时也鼓励歼敌）
    aggress = getattr(world, 'aggression', 1.0)
    if tgt.owner != -1 and tgt.owner != src.owner:
        if aggress < 0.5:
            phase_mult *= 3.0
            capture_bonus *= 2.0
        elif aggress < 0.8:
            phase_mult *= 2.0
            capture_bonus *= 1.5

    value = roi * threat_mult * vuln_mult * phase_mult * breakeven_penalty * distance_penalty * capture_bonus
    return Atom(src.id, tgt.id, ships, angle, eta, value)

def generate_candidates(world, player_id, max_sets=None, extra_plans=None):
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

    # ---- 1. 单源原子生成（原有逻辑） ----
    atoms = []
    for src in my_planets:
        min_garrison = world.dynamic_min_garrison.get(src.id, world.min_garrison_base)
        max_send = int(src.ships) - min_garrison
        if max_send <= 0:
            continue
        targets = top_targets_for_player(src, world, player_id, MAX_TARGETS_PER_SRC)
        for tid in targets:
            tgt = world.planets[tid]
            aim_est = world.get_intercept(src.id, tid, max_send)
            if aim_est is None:
                continue
            eta_est = aim_est[1]
            global_need = needed_for_capture(tgt, eta_est, world)
            remaining_need = global_need
            for ships in ship_options(src, tgt, max_send, world,
                                      eta_estimate=eta_est,
                                      remaining_need=remaining_need):
                atom = make_atom(src, tgt, ships, world, eta_precomputed=eta_est)
                if atom:
                    atoms.append(atom)

    # snipe 逻辑保留不变 ...

    atoms.sort(key=lambda a: -a.value)

    # ---- 2. 构建单源候选集（原有） ----
    candidates = [[]]
    if atoms:
        best_val = atoms[0].value
        selected_atoms = []
        committed_targets = set()
        target_eta_range = {}
        for atom in atoms:
            if atom.value < best_val * BEAM_CUTOFF_RATIO:
                break
            tid = atom.target_id
            tgt = world.planets[tid]
            if tid in committed_targets:
                continue
            # ... 原有筛选逻辑（需求满足、ETA容差等）
            accept_threshold = needed_for_capture(tgt, atom.eta, world) * (NEUTRAL_ACCEPT_RATIO if tgt.owner == -1 else MIN_ACCEPT_RATIO)
            if atom.ships < accept_threshold:
                continue
            if atom.src_id not in {a.src_id for a in selected_atoms}:
                selected_atoms.append(atom)
                committed_targets.add(tid)
                if len(selected_atoms) >= max_sets:
                    break
        cur_set = []
        for atom in selected_atoms:
            cur_set = list(cur_set) + [atom]
            candidates.append(cur_set)

    # ---- 3. 强制多源协同：寻找单源无法攻下但多源可行的目标 ----
    forced_multi = []
    for tid, tgt in world.planets.items():
        if tgt.owner == player_id or is_saturated(tgt, world):
            continue
        if tid in world.comet_ids:
            life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
            if life <= 1:
                continue

        # 估算需要多少船
        rough_eta = 5.0   # 预估
        need = needed_for_capture(tgt, rough_eta, world)

        # 检查是否有任何一个单源可以满足
        any_single_can = any(
            (int(p.ships) - world.dynamic_min_garrison.get(p.id, 0)) >= need
            for p in my_planets
        )
        if any_single_can:
            continue   # 单源已经够，不强制多源

        # 收集所有可用源，按距离排序
        sources = []
        total_avail = 0
        for p in my_planets:
            avail = int(p.ships) - world.dynamic_min_garrison.get(p.id, 0)
            if avail <= 0:
                continue
            aim = world.get_intercept(p.id, tid, min(avail, 100))
            if aim is None:
                continue
            eta = aim[1]
            sources.append((eta, p, avail))
            total_avail += avail

        if total_avail < need:
            continue   # 总兵力也不够

        # 按 eta 排序，依次分配直到满足需求
        sources.sort()
        dispatched = 0
        group = []
        for eta, src, avail in sources:
            if dispatched >= need:
                break
            send = min(avail, need - dispatched)
            atom = make_atom(src, tgt, send, world)
            if atom:
                group.append(atom)
                dispatched += send

        if dispatched >= need:
            # 给整个组合一个非常高的基础价值（强制前排）
            total_value = sum(a.value for a in group) + 500.0  # 大幅加分
            for a in group:
                a.value = total_value / len(group)
            forced_multi.append(group)

    # 将强制多源计划插入候选列表最前面
    for grp in forced_multi:
        candidates.insert(0, grp)

    # ---- 4. 原有双源协同候选 ----
    dual_candidates = []
    for tid, tgt in world.planets.items():
        # ... 原有双源生成逻辑 ...
        # 但要放宽条件：不论 target.ships 多少，只要不是饱和就尝试
        if tgt.owner == player_id or is_saturated(tgt, world):
            continue
        remaining_life = HORIZON_VALUE
        if tid in world.comet_ids:
            life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
            remaining_life = min(remaining_life, life)
            if remaining_life < 10:
                continue

        # 放宽：不再检查 tgt.ships < world.dual_source_min_ships
        closest = sorted(my_planets, key=lambda p: math.hypot(p.x - tgt.x, p.y - tgt.y))
        if len(closest) < 2:
            continue
        # 尝试前3近的来源
        for i in range(min(3, len(closest))):
            for j in range(i+1, min(3, len(closest))):
                s1, s2 = closest[i], closest[j]
                avail1 = max(0, int(s1.ships) - world.dynamic_min_garrison.get(s1.id, world.min_garrison_base))
                avail2 = max(0, int(s2.ships) - world.dynamic_min_garrison.get(s2.id, world.min_garrison_base))
                if avail1 < 2 or avail2 < 2:
                    continue
                aim1 = world.get_intercept(s1.id, tid, avail1)
                aim2 = world.get_intercept(s2.id, tid, avail2)
                if aim1 is None or aim2 is None:
                    continue
                _, eta1, _ = aim1
                _, eta2, _ = aim2
                if abs(eta1 - eta2) > 2:   # 放宽到2回合
                    continue
                max_eta = max(eta1, eta2)
                total_needed = needed_for_capture(tgt, max_eta, world)
                total_avail = avail1 + avail2
                if total_avail < total_needed:
                    continue
                send1 = min(avail1, max(1, int(total_needed * avail1 / total_avail)))
                send2 = min(avail2, total_needed - send1)
                if send1 + send2 < total_needed:
                    shortfall = total_needed - (send1 + send2)
                    if avail1 > send1:
                        add = min(shortfall, avail1 - send1)
                        send1 += add
                        shortfall -= add
                    if shortfall > 0 and avail2 > send2:
                        send2 += min(shortfall, avail2 - send2)
                if send1 <= 0 or send2 <= 0 or send1 + send2 < total_needed:
                    continue
                a1 = make_atom(s1, tgt, send1, world)
                a2 = make_atom(s2, tgt, send2, world)
                if a1 is None or a2 is None:
                    continue
                if abs(a1.eta - a2.eta) > 2:
                    continue
                a1.value *= 1.5   # 提高双源协同价值
                a2.value *= 1.5
                dual_candidates.append([a1, a2])
                if len(dual_candidates) >= MAX_DUAL_CANDIDATES:
                    break
            if len(dual_candidates) >= MAX_DUAL_CANDIDATES:
                break

    candidates.extend(dual_candidates)

    # ---- 5. 合并全局计划 ----
    if extra_plans is None and hasattr(world, '_extra_plans') and world._extra_plans:
        extra_plans = world._extra_plans
    if extra_plans and player_id == world.my_id:
        for plan in extra_plans:
            if plan not in candidates:
                candidates.insert(0, plan)   # 放在最前面

    # 移除重复
    unique_cands = []
    seen = set()
    for c in candidates:
        key = tuple((a.src_id, a.target_id, a.ships) for a in c)
        if key not in seen:
            seen.add(key)
            unique_cands.append(c)
    candidates = unique_cands

    cache[player_id] = candidates
    return candidates

# ============================================================
# Evaluation & Utility
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

def materialize_actions(atoms):
    return [[a.src_id, float(a.angle), int(a.ships)] for a in atoms]

# ============================================================
# Agent entry point
# ============================================================

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

    # 构建世界
    try:
        world = build_world(obs)
    except Exception:
        return []

    my_planets = [p for p in world.planet_list if p.owner == world.my_id]
    if not my_planets:
        return []

    # ========== 终局绝对优势：直接清场 ==========
    if world.is_terminal and world.is_absolute_dominant:
        cleanup_plans = plan_endgame_cleanup(world)
        if cleanup_plans:
            return materialize_actions(cleanup_plans[0])
        # 否则继续普通流程兜底
    # =========================================

    # 生成全局作战计划
    global_plans = []
    if deadline - time.perf_counter() > 0.25:
        try:
            global_plans = plan_global_assaults(world, max_plans=4)
        except Exception:
            pass
    world._extra_plans = global_plans

    # ---------- MCTS 搜索 ----------
    best_atoms = None
    try:
        mcts_budget = min(0.7 * (deadline - start), deadline - start - 0.05)
        best_atoms = mcts_search(world, iterations=200, max_seconds=mcts_budget)
    except Exception:
        pass

    # ---------- 纯规则回退 ----------
    if best_atoms is None:
        candidates = generate_candidates(world, world.my_id)
        if not candidates:
            return []
        candidates_sorted = sorted(candidates, key=lambda c: -immediate_value(c))
        fallback = candidates_sorted[0] if candidates_sorted else []
        best = fallback
        best_score = evaluate(world, [], deadline)

        mcts_root = getattr(world, '_mcts_root', None)
        for cand in candidates_sorted:
            if not cand:
                continue
            if time.perf_counter() > deadline - 0.05:
                break
            try:
                score = evaluate(world, cand, deadline)
            except Exception:
                continue

            # ---- 多源协同加分（新加入） ----
            if len(cand) > 1:
                score += 0.5

            # MCTS 访问次数奖励
            if mcts_root is not None and mcts_root.children:
                key = tuple((a.src_id, a.target_id, a.ships, a.eta) for a in cand)
                child = mcts_root.children.get(key)
                if child and child.visits > 0:
                    score += math.log(child.visits + 1) * 0.1

            if score > best_score:
                best_score = score
                best = cand
        final_moves = materialize_actions(best)
    else:
        final_moves = materialize_actions(best_atoms)

    # ---------- 焦土撤退（非终局碾压时才执行） ----------
    if not (world.is_terminal and world.is_absolute_dominant):
        remaining_time = deadline - time.perf_counter()
        if remaining_time > 0.05:
            safe_planets = [p for p in my_planets if not world.is_doomed(p.id)]
            if safe_planets:
                for planet in my_planets:
                    if planet.id in {m[0] for m in final_moves}:
                        continue
                    doomed_turn = world.is_doomed(planet.id)
                    if doomed_turn is None:
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

            # ---------- 彗星撤离 ----------
            if remaining_time > 0.05:
                EVACUATE_COMET_TURNS = 3
                safe_non_comet = [p for p in my_planets
                                  if p.id not in world.comet_ids and not world.is_doomed(p.id)]
                if safe_non_comet:
                    for comet_id in world.comet_ids:
                        comet = world.planets.get(comet_id)
                        if comet is None or comet.owner != world.my_id:
                            continue
                        life = comet_remaining_turns(world.cid_to_group.get(comet_id), comet_id)
                        if life <= 0 or life > EVACUATE_COMET_TURNS:
                            continue
                        available = int(comet.ships)
                        if available <= 0:
                            continue
                        if comet_id in {m[0] for m in final_moves}:
                            continue
                        dest = min(safe_non_comet, key=lambda p: math.hypot(comet.x - p.x, comet.y - p.y))
                        aim = compute_intercept_with_detour(comet, dest, available, world)
                        if aim is None:
                            continue
                        angle, eta, _ = aim
                        final_moves.append([comet.id, float(angle), int(available)])

    return final_moves