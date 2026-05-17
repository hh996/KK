"""
在本地复现 Kaggle `orbit_wars` 的单步规则与运动学，供 MCTS 对 `WorldState` 反复 `clone`/`step`。

自研物理引擎，经过与官方环境详细对比验证，确保行星旋转、生产、战斗等行为一致。
"""

import math
import random
import copy
from collections import defaultdict

from config import MAX_FLEETS_PER_TURN

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
TOTAL_STEPS = 500
CENTER_X, CENTER_Y = 50.0, 50.0
SUN_R = 10.0
BOARD_SIZE = 100.0
MAX_SPEED = 6.0
ROTATION_LIMIT = 50.0
LAUNCH_CLEARANCE = 0.1
_LOG1000 = math.log(1000.0)
HORIZON_SIM = 80
DETOUR_OFFSETS_DEG = (5, -5, 10, -10, 18, -18)
COMET_SPAWN_STEPS = frozenset((50, 150, 250, 350, 450))
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1


def _distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def point_to_segment_distance(p, v, w):
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return _distance(p, v)
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    proj = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return _distance(p, proj)


def swept_pair_hit(a0, b0, p_old, p_new, r_planet):
    adx = a0[0] - p_old[0]
    ady = a0[1] - p_old[1]
    dvx = (b0[0] - a0[0]) - (p_new[0] - p_old[0])
    dvy = (b0[1] - a0[1]) - (p_new[1] - p_old[1])
    aq = dvx * dvx + dvy * dvy
    bcoef = 2.0 * (adx * dvx + ady * dvy)
    ccoef = adx * adx + ady * ady - r_planet * r_planet
    if aq < 1e-12:
        return ccoef <= 0.0
    disc = bcoef * bcoef - 4.0 * aq * ccoef
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-bcoef - sq) / (2.0 * aq)
    t2 = (-bcoef + sq) / (2.0 * aq)
    return t2 >= 0.0 and t1 <= 1.0


# ---------------------------------------------------------------------------
# 实体
# ---------------------------------------------------------------------------
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
    def clone(self):
        return Planet(self.id, self.owner, self.x, self.y, self.radius, self.ships, self.production)


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
    def clone(self):
        return Fleet(self.id, self.owner, self.x, self.y, self.angle, self.from_planet_id, self.ships)


# ---------------------------------------------------------------------------
# 运动学与碰撞辅助
# ---------------------------------------------------------------------------
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


def line_hits_sun(x0, y0, x1, y1, sun_r=None):
    r = SUN_R if sun_r is None else float(sun_r)
    return line_hits_circle(x0, y0, x1, y1, CENTER_X, CENTER_Y, r)


def is_orbital(planet):
    d = math.hypot(planet.x - CENTER_X, planet.y - CENTER_Y)
    return d + planet.radius < ROTATION_LIMIT


def build_omega_map(planets, initial_planets, step, base_omega):
    omega_map = {}
    for p in planets:
        if is_orbital(p):
            omega_map[p.id] = base_omega
        else:
            omega_map[p.id] = 0.0
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
                     max_iter=30, t_tol=1e-3, ang_tol=1e-4, max_speed=None):
    ms = MAX_SPEED if max_speed is None else float(max_speed)
    if ships <= 0:
        return None, None
    speed = fleet_speed(ships, max_speed=ms)
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
    speed_a = fleet_speed(ships, max_speed=ms)
    fx = sx + t_est * speed_a * math.cos(angle)
    fy = sy + t_est * speed_a * math.sin(angle)
    tx, ty = predict_target_position(target, t_est, omega_map, cid_to_group, comet_ids)
    miss = math.hypot(fx - tx, fy - ty)
    if miss > max(target.radius, 1.2):
        return None, None
    return angle, t_est


def path_blocked_by_other_planet(src, target, angle, eta, ships, planets,
                                 omega_map, cid_to_group, comet_ids, max_speed=None):
    sx, sy = get_launch_position(src, angle)
    speed = fleet_speed(ships, max_speed=MAX_SPEED if max_speed is None else float(max_speed))
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
    bs = world.board_size
    sr = world.sun_r
    sx, sy = get_launch_position(src, angle)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    prev_x, prev_y = sx, sy
    for k in range(1, max_turns + 1):
        fx = sx + k * speed * cos_a
        fy = sy + k * speed * sin_a
        if not (0 <= fx <= bs and 0 <= fy <= bs):
            return None
        if line_hits_sun(prev_x, prev_y, fx, fy, sun_r=sr):
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
        src, target, ships, world.omega_map, world.cid_to_group, world.comet_ids,
        max_speed=world.ship_speed,
    )
    if angle is None:
        return None
    speed = fleet_speed(ships, max_speed=world.ship_speed)
    sx, sy = get_launch_position(src, angle)
    fx = sx + eta * speed * math.cos(angle)
    fy = sy + eta * speed * math.sin(angle)
    direct_blocked = path_blocked_by_other_planet(
        src, target, angle, eta, ships, world.planet_list,
        world.omega_map, world.cid_to_group, world.comet_ids,
        max_speed=world.ship_speed,
    )
    if not direct_blocked and not line_hits_sun(sx, sy, fx, fy, sun_r=world.sun_r):
        return angle, eta, 0.0
    for deg in DETOUR_OFFSETS_DEG:
        offset_rad = math.radians(deg)
        new_angle = _angle_norm(angle + offset_rad)
        new_eta = trace_intercept(src, new_angle, target, speed, world)
        if new_eta is None:
            continue
        nsx, nsy = get_launch_position(src, new_angle)
        if line_hits_sun(nsx, nsy, nsx + new_eta * speed * math.cos(new_angle), nsy + new_eta * speed * math.sin(new_angle), sun_r=world.sun_r):
            continue
        if not path_blocked_by_other_planet(
            src, target, new_angle, new_eta, ships, world.planet_list,
            world.omega_map, world.cid_to_group, world.comet_ids,
            max_speed=world.ship_speed,
        ):
            return new_angle, new_eta, float(deg)
    return None


def predict_fleet_arrival(fleet, planets, omega_map, cid_to_group, comet_ids,
                          max_turns=HORIZON_SIM, max_speed=None, board_size=None, sun_r=None):
    ms = MAX_SPEED if max_speed is None else float(max_speed)
    bs = BOARD_SIZE if board_size is None else float(board_size)
    sr = SUN_R if sun_r is None else float(sun_r)
    fx0, fy0 = fleet.x, fleet.y
    speed = fleet_speed(fleet.ships, max_speed=ms)
    cos_a, sin_a = math.cos(fleet.angle), math.sin(fleet.angle)
    prev_x, prev_y = fx0, fy0
    for k in range(1, max_turns + 1):
        nx = fx0 + k * speed * cos_a
        ny = fy0 + k * speed * sin_a
        if not (0 <= nx <= bs and 0 <= ny <= bs):
            return (k, None)
        if line_hits_sun(prev_x, prev_y, nx, ny, sun_r=sr):
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


def generate_comet_paths(
    initial_planets,
    angular_velocity,
    spawn_step,
    comet_planet_ids=None,
    comet_speed=4.0,
    rng=None,
    board_size=BOARD_SIZE,
    sun_radius=SUN_R,
):
    """从官方 orbit_wars.py 移植，替换硬编码常量为可配置参数。"""
    if rng is None:
        rng = random
    if comet_planet_ids is None:
        comet_planet_ids = set()
    else:
        comet_planet_ids = set(comet_planet_ids)

    center = board_size / 2.0

    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < sun_radius + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e ** 2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        dense = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            x = center + ex * math.cos(phi) - ey * math.sin(phi)
            y = center + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += _distance(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= board_size and 0 <= y <= board_size:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path[board_start: board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        paths = [
            [[y, x] for x, y in visible],
            [[board_size - x, y] for x, y in visible],
            [[x, board_size - y] for x, y in visible],
            [[board_size - y, board_size - x] for x, y in visible],
        ]

        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if planet[0] in comet_planet_ids:
                continue
            pr = _distance((planet[2], planet[3]), (center, center))
            if pr + planet[4] < ROTATION_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if _distance((cx, cy), (center, center)) < sun_radius + COMET_RADIUS:
                valid = False
                break

            sym_pts = [
                (cy, cx),
                (board_size - cx, cy),
                (cx, board_size - cy),
                (board_size - cy, board_size - cx),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if _distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - center
                dy = planet[3] - center
                orb_r = math.hypot(dx, dy)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = center + orb_r * math.cos(cur_angle)
                py = center + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if _distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------
class WorldState:
    def __init__(self, planets, fleets, initial_planets, step, base_omega,
                 comets, comet_ids, player_ids, my_id,
                 num_training_agents=None, episode_seed=None,
                 comet_speed=4.0, ship_speed=None, sun_radius=None, board_size=None,
                 _precomputed_fleet_arrivals=None):
        self.ship_speed = float(MAX_SPEED if ship_speed is None else ship_speed)
        self.sun_r = float(SUN_R if sun_radius is None else sun_radius)
        self.board_size = float(BOARD_SIZE if board_size is None else board_size)
        self.planet_list = [p.clone() for p in planets]
        self.fleets = [f.clone() for f in fleets]
        self.planets = {p.id: p for p in self.planet_list}
        self.fleet_by_id = {f.id: f for f in self.fleets}
        self.initial_planets = copy.deepcopy(initial_planets)
        self._initial_planets_dict = {ip[0]: ip for ip in self.initial_planets}
        self.step_count = int(step)
        self.base_omega = base_omega
        self.comets = copy.deepcopy(comets)
        self.comet_ids = set(comet_ids)
        self.episode_seed = episode_seed
        self.comet_speed = float(comet_speed)
        self.omega_map = build_omega_map(self.planet_list, self.initial_planets, self.step_count, base_omega)
        self.cid_to_group = {}
        for g in self.comets:
            for pid in g.get("planet_ids", []):
                self.cid_to_group[pid] = g
        raw_ids = sorted(player_ids)
        self.num_training_agents = num_training_agents if num_training_agents is not None else len(raw_ids)
        self.player_ids = list(range(self.num_training_agents)) if self.num_training_agents >= 2 else (raw_ids if raw_ids else [0, 1])
        self.my_id = my_id
        if _precomputed_fleet_arrivals is not None:
            self.fleet_arrivals = dict(_precomputed_fleet_arrivals)
        else:
            self.fleet_arrivals = {}
            for f in self.fleets:
                self.fleet_arrivals[f.id] = predict_fleet_arrival(
                    f, self.planet_list, self.omega_map, self.cid_to_group, self.comet_ids,
                    max_speed=self.ship_speed, board_size=self.board_size, sun_r=self.sun_r,
                )
        self._intercept_cache = {}

    def clone(self):
        cloned = WorldState(
            self.planet_list, self.fleets, copy.deepcopy(self.initial_planets),
            self.step_count, self.base_omega, copy.deepcopy(self.comets),
            set(self.comet_ids), list(self.player_ids), self.my_id,
            self.num_training_agents, episode_seed=self.episode_seed,
            comet_speed=self.comet_speed, ship_speed=self.ship_speed,
            sun_radius=self.sun_r, board_size=self.board_size,
            _precomputed_fleet_arrivals=self.fleet_arrivals,
        )
        return cloned

    @property
    def remaining_steps(self):
        return max(0, TOTAL_STEPS - self.step_count)

    def get_atomic_legal_actions(self, player_id=None):
        if player_id is None:
            player_id = self.my_id
        actions = []
        my_planets = [p for p in self.planet_list if p.owner == player_id]
        enemy_planets = [p for p in self.planet_list if p.owner != player_id]
        for src in my_planets:
            if src.ships <= 1:
                continue
            max_send = int(src.ships) - 1
            for tgt in enemy_planets:
                for ships in {max(1, tgt.ships+1), min(max_send, tgt.ships+5), max_send}:
                    if ships <= 0 or ships > max_send:
                        continue
                    key = (src.id, tgt.id, ships)
                    if key in self._intercept_cache:
                        result = self._intercept_cache[key]
                    else:
                        result = compute_intercept_with_detour(src, tgt, ships, self)
                        self._intercept_cache[key] = result
                    if result is not None:
                        angle, eta, _ = result
                        actions.append((src.id, tgt.id, ships, angle, eta))
        return actions

    def atomic_pair_compatible(self, player_id, a, b):
        sa, _, qa, _, _ = a
        sb, _, qb, _, _ = b
        pa = self.planets.get(sa)
        pb = self.planets.get(sb)
        if not pa or not pb:
            return False
        if pa.owner != player_id or pb.owner != player_id:
            return False
        if sa == sb:
            return pa.ships >= qa + qb
        return pa.ships >= qa and pb.ships >= qb

    def get_legal_macro_actions(self, player_id, max_fleets=2, max_macros=450):
        k = max(1, min(int(max_fleets), MAX_FLEETS_PER_TURN))
        atoms = self.get_atomic_legal_actions(player_id)
        macros = [()]                           # PASS：不发任何舰队
        macros += [(a,) for a in atoms]
        if k < 2 or len(atoms) < 2:
            return macros[:max_macros]
        cap_i = min(len(atoms), 36)
        for i in range(cap_i):
            for j in range(i + 1, cap_i):
                a_i, b_j = atoms[i], atoms[j]
                if self.atomic_pair_compatible(player_id, a_i, b_j):
                    macros.append((a_i, b_j))
                    if len(macros) >= max_macros:
                        return macros
        return macros[:max_macros]

    def get_legal_actions(self, player_id=None):
        if player_id is None:
            player_id = self.my_id
        return self.get_legal_macro_actions(player_id)

    def apply_action(self, action, player_id=None):
        if player_id is None:
            player_id = self.my_id
        src_id, target_id, ships, angle, eta = action
        src = self.planets.get(src_id)
        if src is None or src.owner != player_id or src.ships < ships:
            return False
        src.ships -= ships
        fx, fy = get_launch_position(src, angle)
        new_id = max([f.id for f in self.fleets] + [-1]) + 1
        fleet = Fleet(new_id, player_id, fx, fy, angle, src_id, ships)
        self.fleets.append(fleet)
        self.fleet_by_id[fleet.id] = fleet
        self.fleet_arrivals[fleet.id] = predict_fleet_arrival(
            fleet, self.planet_list, self.omega_map, self.cid_to_group, self.comet_ids,
            max_speed=self.ship_speed, board_size=self.board_size, sun_r=self.sun_r,
        )
        return True

    def is_terminal(self):
        if self.step_count >= TOTAL_STEPS:
            return True
        active = set()
        for p in self.planet_list:
            if p.owner != -1:
                active.add(p.owner)
        for f in self.fleets:
            active.add(f.owner)
        return len(active) <= 1

    def get_scores(self):
        scores = {pid: 0 for pid in self.player_ids}
        for p in self.planet_list:
            if p.owner != -1:
                scores[p.owner] = scores.get(p.owner, 0) + p.ships
        for f in self.fleets:
            if f.owner != -1:
                scores[f.owner] = scores.get(f.owner, 0) + f.ships
        return scores

    @staticmethod
    def _resolve_planet_combat(planet, arrivals):
        if not arrivals:
            return
        by_owner = defaultdict(int)
        for owner, ships in arrivals:
            if ships > 0:
                by_owner[owner] += int(ships)
        if not by_owner:
            return
        sorted_att = sorted(by_owner.items(), key=lambda x: -x[1])
        top_owner, top_ships = sorted_att[0]
        second_ships = sorted_att[1][1] if len(sorted_att) >= 2 else 0
        if len(sorted_att) >= 2 and second_ships == top_ships:
            return
        survivor = top_ships - second_ships
        if survivor <= 0:
            return
        if top_owner == planet.owner:
            planet.ships += survivor
        else:
            planet.ships -= survivor
            if planet.ships < 0:
                planet.owner = top_owner
                planet.ships = -planet.ships

    def _apply_comet_planet_removal(self, expired_pids):
        if not expired_pids:
            return
        expired_set = set(expired_pids)
        self.planet_list = [p for p in self.planet_list if p.id not in expired_set]
        self.planets = {p.id: p for p in self.planet_list}
        self.initial_planets = [p for p in self.initial_planets if p[0] not in expired_set]
        for pid in expired_set:
            self._initial_planets_dict.pop(pid, None)
        self.comet_ids = set(pid for pid in self.comet_ids if pid not in expired_set)
        for group in list(self.comets):
            new_ids = [pid for pid in group.get("planet_ids", []) if pid not in expired_set]
            group["planet_ids"] = new_ids
        self.comets = [g for g in self.comets if g.get("planet_ids")]
        self.cid_to_group = {}
        for g in self.comets:
            for pid in g.get("planet_ids", []):
                self.cid_to_group[pid] = g

    def _spawn_comets_matching_env(self):
        next_step = self.step_count + 1
        if next_step not in COMET_SPAWN_STEPS:
            return
        ep_seed = self.episode_seed if self.episode_seed is not None else 0
        comet_rng = random.Random(f"orbit_wars-comet-{ep_seed}-{next_step}")
        comet_paths = generate_comet_paths(
            self.initial_planets, self.base_omega, next_step,
            self.comet_ids, self.comet_speed, rng=comet_rng,
            board_size=self.board_size, sun_radius=self.sun_r,
        )
        if not comet_paths:
            return
        next_id = max((p.id for p in self.planet_list), default=-1) + 1
        comet_ships = min(comet_rng.randint(1,99), comet_rng.randint(1,99),
                          comet_rng.randint(1,99), comet_rng.randint(1,99))
        group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
        for i in range(len(comet_paths)):
            pid = next_id + i
            group["planet_ids"].append(pid)
            self.comet_ids.add(pid)
            row = [pid, -1, -99.0, -99.0, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
            self.initial_planets.append(list(row))
            self._initial_planets_dict[pid] = list(row)
            pl = Planet(pid, -1, -99.0, -99.0, COMET_RADIUS, comet_ships, COMET_PRODUCTION)
            self.planet_list.append(pl)
            self.planets[pid] = pl
            self.cid_to_group[pid] = group
        self.comets.append(group)

    def _build_planet_paths_this_tick(self):
        planet_paths = {}
        expired_after = []

        for p in self.planet_list:
            if p.id in self.comet_ids:
                continue
            old_pos = (p.x, p.y)
            init_p = self._initial_planets_dict.get(p.id)
            if init_p is not None:
                dx = init_p[2] - CENTER_X
                dy = init_p[3] - CENTER_Y
                r = math.hypot(dx, dy)
                if r + p.radius < ROTATION_LIMIT:
                    new_angle = math.atan2(dy, dx) + self.base_omega * self.step_count
                    nx = CENTER_X + r * math.cos(new_angle)
                    ny = CENTER_Y + r * math.sin(new_angle)
                    planet_paths[p.id] = (old_pos, (nx, ny), True)
                else:
                    planet_paths[p.id] = (old_pos, old_pos, True)
            else:
                planet_paths[p.id] = (old_pos, old_pos, True)

        for group in self.comets:
            group["path_index"] = int(group.get("path_index", -1)) + 1
            idx = group["path_index"]
            for i, pid in enumerate(group.get("planet_ids", [])):
                planet = self.planets.get(pid)
                if planet is None:
                    continue
                paths = group.get("paths", [])
                if i >= len(paths):
                    continue
                path = paths[i]
                old_pos = (planet.x, planet.y)
                if idx >= len(path):
                    expired_after.append(pid)
                    planet_paths[pid] = (old_pos, old_pos, True)
                else:
                    new_pos = (path[idx][0], path[idx][1])
                    check = old_pos[0] >= 0 and old_pos[1] >= 0
                    planet_paths[pid] = (old_pos, new_pos, check)
        return planet_paths, expired_after

    def step(self, actions_dict):
        # 1. 清理过期彗星（发射前）
        expired_before = []
        for group in self.comets:
            idx = group.get("path_index", -1)
            for pi, cid in enumerate(group.get("planet_ids", [])):
                paths = group.get("paths", [])
                if pi >= len(paths) or idx >= len(paths[pi]):
                    expired_before.append(cid)
        self._apply_comet_planet_removal(expired_before)

        # 2. 生成彗星
        self._spawn_comets_matching_env()

        # 3. 发射舰队
        for pid in sorted(actions_dict.keys()):
            for act in actions_dict[pid]:
                self.apply_action(act, pid)

        # 4. 生产
        for p in self.planet_list:
            if p.owner != -1:
                p.ships += p.production

        # 5. 计算行星本 tick 路径（包含旋转），并存储旧位置
        planet_paths, expired_after = self._build_planet_paths_this_tick()

        # 6. 移动舰队并检测碰撞（使用行星旧位置和新位置）
        planet_combat = defaultdict(list)
        new_fleets = []
        for f in self.fleets:
            speed = fleet_speed(f.ships, max_speed=self.ship_speed)
            old_pos = (f.x, f.y)
            new_x = f.x + speed * math.cos(f.angle)
            new_y = f.y + speed * math.sin(f.angle)
            hit = False
            for p in self.planet_list:
                path = planet_paths.get(p.id)
                if path is None or not path[2]:
                    continue
                p_old, p_new, _ = path
                if swept_pair_hit(old_pos, (new_x, new_y), p_old, p_new, p.radius):
                    planet_combat[p.id].append((f.owner, f.ships))
                    hit = True
                    break
            if hit:
                continue
            if not (0 <= new_x <= self.board_size and 0 <= new_y <= self.board_size):
                continue
            if line_hits_sun(old_pos[0], old_pos[1], new_x, new_y, sun_r=self.sun_r):
                continue
            f.x, f.y = new_x, new_y
            new_fleets.append(f)
        self.fleets = new_fleets
        self.fleet_by_id = {ff.id: ff for ff in self.fleets}
        self.fleet_arrivals = {}
        for ff in self.fleets:
            self.fleet_arrivals[ff.id] = predict_fleet_arrival(
                ff, self.planet_list, self.omega_map, self.cid_to_group, self.comet_ids,
                max_speed=self.ship_speed, board_size=self.board_size, sun_r=self.sun_r,
            )

        # 7. 应用行星新位置
        for p in self.planet_list:
            path = planet_paths.get(p.id)
            if path is not None:
                p.x, p.y = path[1]

        # 8. 清理过期彗星（移动后）
        self._apply_comet_planet_removal(expired_after)

        # 9. 结算战斗
        for planet_id, attackers in planet_combat.items():
            planet = self.planets.get(planet_id)
            if planet is None:
                continue
            self._resolve_planet_combat(planet, attackers)

        self.step_count += 1
        self.omega_map = build_omega_map(self.planet_list, self.initial_planets, self.step_count, self.base_omega)