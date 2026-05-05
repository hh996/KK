"""
Orbit Wars v2 - 统一增量 V 值函数 + 1 步 beam search 前瞻

核心策略：
  1. 每回合产生 <= BEAM_WIDTH 个候选动作集合（通过贪心 beam search）。
  2. 用统一的价值函数 V，在 HORIZON_VALUE 回合处评估每个候选，
     评估前会模拟 1 步（我方行动 + 对手简化最佳应对）。
  3. 选择使 V(我) - max V(对手) 最大的候选集合。
  4. 始终评估“无操作”作为基线；若超时则退回最高即时启发值的候选。

物理计算（延续 v1.2）：
  - 真实的舰队速度公式：1 + 5 * (ln(n) / ln(1000)) ^ 1.5
  - 对每个行星，通过初始状态反推其带符号的旋转角速度
  - 带阻尼的迭代截击计算
  - 与其他所有行星的线段-圆形碰撞检测，判断路径是否被阻挡
  - 直连被阻时在 +/- {5, 10, 18} 度方向尝试绕行
"""

import math
import time
from collections import defaultdict


# ============================================================
# 常数定义
# ============================================================

TOTAL_STEPS = 500                 # 游戏总回合数
CENTER_X, CENTER_Y = 50.0, 50.0   # 太阳中心
SUN_R = 10.0                      # 太阳半径
BOARD_SIZE = 100.0                # 棋盘大小
MAX_SPEED = 6.0                   # 舰队最大速度
ROTATION_LIMIT = 50.0             # 判定公转行星的半径限制
LAUNCH_CLEARANCE = 0.01           # 发射时的额外微小偏移
_LOG1000 = math.log(1000.0)       # 预计算常量

# 策略超参数
HORIZON_VALUE = 120          # V 值评估的时间跨度（回合数）
HORIZON_SIM = 80             # 追踪舰队到目标的最长时间（回合数）
BEAM_WIDTH = 6               # 候选集合中最多包含几个原子动作（堆叠）
MAX_TARGETS_PER_SRC = 4      # 每个源行星考虑的目标数
MAX_SOURCES = 6              # 只考虑兵力最强的 N 个源行星
PROD_DISCOUNT = 0.85         # 对未来生产力打折的因子（当前版本未显式使用，保留）
FRONTIER_BONUS = 4.0         # 己方行星离敌人越近的额外价值
PLANET_OWN_BONUS = 2.0       # 拥有一颗行星的固定生产力奖励
DETOUR_OFFSETS_DEG = (5, -5, 10, -10, 18, -18)  # 绕行尝试的角度偏移（度）
SOFT_DEADLINE = 0.85         # 每回合最大执行时间比例
OPP_TIME_FRACTION = 0.55     # 分配给对手模拟的时间比例
OPP_MAX_EVAL = 5             # 最多评估几个对手候选集合


# ============================================================
# 数据类：行星球和舰队
# ============================================================

class Planet:
    """行星对象，存储 id、拥有者、坐标、半径、舰船数、生产力。"""
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
    """舰队对象，存储 id、拥有者、当前位置、角度、来源行星、舰船数。"""
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
    """单个发射动作的提议（原子）。"""
    __slots__ = ('src_id', 'target_id', 'ships', 'angle', 'eta', 'value')
    def __init__(self, src_id, target_id, ships, angle, eta, value=0.0):
        self.src_id = src_id       # 发射源行星 id
        self.target_id = target_id # 目标行星 id
        self.ships = ships         # 发射的舰船数
        self.angle = angle         # 发射方向弧度
        self.eta = eta             # 预计到达时间（回合数）
        self.value = value         # 启发式价值，用于排序


# ============================================================
# 物理层：速度、位置预测、碰撞检测
# ============================================================

def fleet_speed(ships, max_speed=MAX_SPEED):
    """
    真实环境中的速度计算公式：
    speed = 1 + (max - 1) * (ln(ships) / ln(1000)) ^ 1.5
    当 ships <= 1 时，速度为 1.0；接近 1000 时接近最大值。
    """
    if ships <= 1:
        return 1.0
    s = 1.0 + (max_speed - 1.0) * (math.log(ships) / _LOG1000) ** 1.5
    return min(s, max_speed)


def get_launch_position(src, angle):
    """根据源行星和发射角度，计算舰队在行星边缘外的实际生成位置。"""
    r = src.radius + LAUNCH_CLEARANCE
    return src.x + r * math.cos(angle), src.y + r * math.sin(angle)


def _angle_norm(a):
    """将角度规范化到 [-π, π] 区间。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def line_hits_circle(x0, y0, x1, y1, cx, cy, cr):
    """
    判断线段 (x0,y0) -> (x1,y1) 是否与圆心 (cx,cy)、半径 cr 的圆相交。
    返回 True/False。
    """
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
    # 只要至少有一个交点在线段参数 t 的 [0,1] 范围内，即碰撞
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 < t2)


def line_hits_sun(x0, y0, x1, y1):
    """判断线段是否与太阳碰撞。"""
    return line_hits_circle(x0, y0, x1, y1, CENTER_X, CENTER_Y, SUN_R)


def is_orbital(planet):
    """判断行星是否为公转行星（到太阳距离 + 半径 < 限制）。"""
    d = math.hypot(planet.x - CENTER_X, planet.y - CENTER_Y)
    return d + planet.radius < ROTATION_LIMIT


def estimate_signed_omega(planet, init_pos, step, base_omega):
    """
    反推一颗公转行星的带符号角速度。
    通过与初始位置对比，计算当前角度与期望转向（+/- base_omega * step）的匹配度，
    选择误差最小的符号作为该行星的角速度。
    """
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
    """
    为所有行星建立 {planet_id: 角速度} 的字典。
    公转行星即使符号可能不同，也被正确估计；非公转行星角速度为0。
    """
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
    """根据角速度 omega 和经过的回合数 t，预测行星的未来位置。"""
    dx = planet.x - CENTER_X
    dy = planet.y - CENTER_Y
    r = math.hypot(dx, dy)
    theta0 = math.atan2(dy, dx)
    theta = theta0 + omega * t
    return CENTER_X + r * math.cos(theta), CENTER_Y + r * math.sin(theta)


def predict_comet_position(comet_group, planet_id, t):
    """预测彗星在 t 回合后的位置，通过线性插值路径点实现。"""
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
        # 如果索引超出范围，取最近的有效点
        i0 = max(0, min(i0, len(path) - 1))
        return path[i0][0], path[i0][1]
    frac = f_idx - i0
    x = path[i0][0] * (1.0 - frac) + path[i1][0] * frac
    y = path[i0][1] * (1.0 - frac) + path[i1][1] * frac
    return x, y


def predict_target_position(target, t, omega_map, cid_to_group, comet_ids):
    """根据目标行星的类型（彗星/公转/静止），预测它在 t 回合后的位置。"""
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
    """
    迭代求解固定点问题：找到 (发射角度, 到达时间) 使得舰队能命中运动的目标。
    使用带阻尼的迭代法，避免快速旋转目标导致震荡。
    返回 (angle, eta) 或 (None, None) 若无法命中。
    """
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
        # 0.6 的阻尼因子避免振荡
        angle = _angle_norm(last_a + 0.6 * d_a)
        t_est = max(0.1, last_t + 0.6 * d_t)
        last_a, last_t = angle, t_est

    # 最终检查，若落点与目标中心距离大于 max(目标半径, 1.2) 则认为失败
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
    """
    逐帧检查：发射舰队前往目标的路径是否被其他任何行星（包括友方）阻挡。
    检查线段与行星的碰撞，考虑行星自身的运动。
    若被阻挡返回 True，否则 False。
    """
    sx, sy = get_launch_position(src, angle)
    speed = fleet_speed(ships)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    n_frames = int(math.ceil(eta)) + 1
    prev_x, prev_y = sx, sy
    for k in range(1, n_frames + 1):
        fx = sx + k * speed * cos_a
        fy = sy + k * speed * sin_a
        tm = k - 0.5  # 取中点时间进行检查
        for p in planets:
            if p.id == src.id or p.id == target.id:
                continue
            # 预测该行星在时间 tm 的位置
            if p.id in comet_ids:
                pos = predict_comet_position(cid_to_group.get(p.id), p.id, tm)
                if pos is None:
                    continue
                px, py = pos
            elif abs(omega_map.get(p.id, 0.0)) > 1e-9:
                px, py = predict_orbit_position(p, omega_map[p.id], tm)
            else:
                px, py = p.x, p.y
            # 检查线段是否与行星圆相交
            if line_hits_circle(prev_x, prev_y, fx, fy, px, py, p.radius + 0.05):
                return True
        prev_x, prev_y = fx, fy
    return False


def trace_intercept(src, angle, target, speed, world, max_turns=HORIZON_SIM):
    """
    射线追踪：给定固定角度，计算舰队线段何时进入目标行星的预测圆形区域。
    若在 max_turns 内命中，返回回合数（浮点）；否则返回 None。
    """
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
    """
    先尝试直接拦截；若路径被阻挡或撞太阳，尝试若干角度偏转（DETOUR_OFFSETS_DEG），
    寻找一条不被阻挡且不撞太阳的路径。
    返回 (angle, eta, detour_degrees) 或 None。
    """
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

    # 尝试绕行
    for deg in DETOUR_OFFSETS_DEG:
        offset_rad = math.radians(deg)
        new_angle = _angle_norm(angle + offset_rad)
        new_eta = trace_intercept(src, new_angle, target, speed, world)
        if new_eta is None:
            continue
        # 再次检查碰撞
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
    """
    对已经存在的舰队进行射线追踪，预测它的到达时间 eta 和目标行星 id。
    若离开棋盘或撞太阳，返回 (max_turns+1, None)。
    """
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
# 世界状态快照与缓存
# ============================================================

class WorldState:
    """
    包含一个回合的完整环境快照，以及预计算的各种缓存，避免重复计算。
    """
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

        # 确定所有存活玩家（包括仍在星球或舰队上的）
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

        # 预测所有现有舰队的到达信息（后续候选评估中不变）
        self.fleet_arrivals = {}
        for f in fleets:
            self.fleet_arrivals[f.id] = predict_fleet_arrival(
                f, planets, self.omega_map, self.cid_to_group, self.comet_ids
            )

        # 缓存区
        self._intercept_cache = {}
        self._top_targets_cache = {}
        self._candidate_cache = {}
        self._best_response_cache = {}

    def get_intercept(self, src_id, target_id, ships):
        """
        获取 (源, 目标, 舰船数) 对应的拦截方案，带缓存。
        返回 (angle, eta, detour_deg) 或 None。
        """
        key = (src_id, target_id, int(ships))
        cached = self._intercept_cache.get(key)
        if cached is not None or key in self._intercept_cache:
            return cached
        src = self.planets[src_id]
        target = self.planets[target_id]
        result = compute_intercept_with_detour(src, target, ships, self)
        self._intercept_cache[key] = result
        return result


# ============================================================
# 前向模拟：计算行星上的舰船时间线
# ============================================================

def simulate_planet_timeline(planet, arrivals, horizon):
    """
    给定一个行星和所有到达事件列表 (eta, owner, ships)，
    模拟从当前到 horizon 回合的 owner 和 garrison 变化。
    战斗逻辑按照 README：
    1. 所有进攻方按 owner 分组，求和。
    2. 两个最大的进攻方交战，差值存活；若最大两个相等则都消灭。
    3. 若幸存者与行星拥有者相同则加入驻军，否则进攻驻军，占领后更新。
    返回一个列表，每个元素为 (owner, ships)，索引为回合。
    """
    horizon = int(math.ceil(horizon))
    by_turn = defaultdict(list)
    for eta, owner, ships in arrivals:
        eta_int = max(1, int(math.ceil(eta)))  # 到达回合取上整
        if eta_int > horizon or ships <= 0:
            continue
        by_turn[eta_int].append((owner, int(ships)))

    owner = planet.owner
    garrison = float(planet.ships)
    timeline = [(owner, garrison)]

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production   # 本回合生产
        if turn in by_turn:
            arr = by_turn[turn]
            by_owner = defaultdict(int)
            for o, s in arr:
                by_owner[o] += s
            # 找出前两名
            sorted_attackers = sorted(by_owner.items(), key=lambda x: -x[1])
            top_owner, top_ships = sorted_attackers[0]
            second_ships = sorted_attackers[1][1] if len(sorted_attackers) >= 2 else 0
            if len(sorted_attackers) >= 2 and second_ships == top_ships:
                # 平手 => 全部摧毁
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
    """
    基于现有舰队和给定新的行动列表，构建每个行星的时间线。
    all_actions：可迭代的 (player_id, target_id, eta, ships)
    target_id 可能是 None（舰队丢失），这种行动不参与。
    """
    arrivals = defaultdict(list)
    for fid, (eta, tid) in world.fleet_arrivals.items():
        if tid is None:
            continue
        f = world.fleet_by_id[fid]
        arrivals[tid].append((eta, f.owner, f.ships))
    for player_id, target_id, eta, ships in all_actions:
        if target_id is None or ships <= 0:
            continue
        arrivals[target_id].append((eta, player_id, ships))

    timelines = {}
    for pid, p in world.planets.items():
        timelines[pid] = simulate_planet_timeline(
            p, arrivals.get(pid, []), HORIZON_VALUE
        )
    return timelines


# ============================================================
# 价值函数 V 与 delta_V
# ============================================================

def V_at_horizon(timelines, world, player, horizon):
    """
    在指定 horizon 回合，计算玩家 player 的价值：
    V = 己方所有行星(舰船 + 固定拥有奖励 + 靠近敌人奖励)。
    注意：仍在途中的舰队已经通过到达事件被纳入时间线计算，在此处不再重复计算。
    """
    horizon = min(horizon, HORIZON_VALUE)
    # 收集所有敌方行星的坐标，用于 frontier 奖励计算
    enemy_positions = []
    for pid, tl in timelines.items():
        owner, _ = tl[horizon]
        if owner != player and owner != -1:
            p = world.planets[pid]
            enemy_positions.append((p.x, p.y))

    score = 0.0
    for pid, tl in timelines.items():
        owner, ships = tl[horizon]
        p = world.planets[pid]
        if owner == player:
            score += ships
            score += p.production * PLANET_OWN_BONUS
            # 离敌人越近的行星价值越高（前线加分）
            if enemy_positions:
                min_d = min(
                    math.hypot(p.x - ex, p.y - ey) for ex, ey in enemy_positions
                )
                score += FRONTIER_BONUS * p.production / (min_d + 5.0)
    return score


def delta_V(timelines, world, player):
    """计算 V(我) - max(V(对手))。"""
    v_me = V_at_horizon(timelines, world, player, HORIZON_VALUE)
    v_opp_max = 0.0
    for opp in world.opponent_ids:
        v_opp = V_at_horizon(timelines, world, opp, HORIZON_VALUE)
        if v_opp > v_opp_max:
            v_opp_max = v_opp
    return v_me - v_opp_max


# ============================================================
# 候选动作生成（贪心 beam）
# ============================================================

def top_targets_for_player(src, world, player, top_k):
    """
    对于一个给定的源行星 src，返回对 player 最有吸引力的 top_k 个目标 id。
    使用简单的启发式指标：生产力 / (成本 * 0.4 + 预计时间 + 5)，中立行星略有加权。
    """
    cache_key = (src.id, player)
    cached = world._top_targets_cache.get(cache_key)
    if cached is not None:
        return cached
    candidates = []
    for tid, tgt in world.planets.items():
        if tgt.owner == player or tid == src.id:
            continue
        d = math.hypot(src.x - tgt.x, src.y - tgt.y)
        if d < 1:
            continue
        proj_ships = max(int(tgt.ships) + 5, 10)  # 预估征服所需舰船数
        speed = fleet_speed(proj_ships)
        eta = d / speed
        if eta > HORIZON_SIM * 0.8:
            continue
        cost = max(1, int(tgt.ships) + 1)
        val = tgt.production / (cost * 0.4 + eta + 5.0)
        if tgt.owner == -1:
            val *= 1.2
        candidates.append((val, tid))
    candidates.sort(reverse=True)
    result = [tid for _, tid in candidates[:top_k]]
    world._top_targets_cache[cache_key] = result
    return result


def ship_options(src, tgt, max_ships):
    """
    返回可能发射的舰船数列表：刚刚好（驻军+2）、舒适（驻军+8）、全力。
    避免只测试单一数量。
    """
    if max_ships <= 0:
        return []
    target_ships = max(1, int(tgt.ships))
    options = set()
    options.add(min(max_ships, target_ships + 2))
    options.add(min(max_ships, target_ships + 8))
    options.add(min(max_ships, max(target_ships + 2, max_ships)))
    return sorted(o for o in options if 1 <= o <= max_ships)


def make_atom(src, tgt, ships, world):
    """尝试构建一个原子动作，若拦截可行则返回 Atom，否则 None。"""
    aim = world.get_intercept(src.id, tgt.id, ships)
    if aim is None:
        return None
    angle, eta, _detour = aim
    if eta > HORIZON_SIM:
        return None
    # 启发式即时价值，用于 beam 排序
    production_gain = tgt.production * max(0, HORIZON_VALUE - int(eta))
    cost = ships + int(eta) * 0.5
    value = production_gain / (cost + 1.0)
    if tgt.owner != -1 and tgt.owner != src.owner:
        value *= 1.15  # 夺取敌人行星价值更高
    return Atom(src.id, tgt.id, ships, angle, eta, value)


def generate_candidates(world, player_id, max_sets=BEAM_WIDTH):
    """
    贪心 beam 动作集合生成：
    为我的行星生成所有可行的原子动作，然后按“每源最佳一个”堆叠形成候选集合。
    还会加入一些备选方案（如第二佳原子）以提高多样性。
    返回候选集合列表，每个集合是一个 Atom 列表。
    """
    cache = world._candidate_cache
    if player_id in cache:
        return cache[player_id]

    my_planets = [p for p in world.planet_list if p.owner == player_id]
    if not my_planets:
        cache[player_id] = [[]]
        return cache[player_id]

    # 取舰船数最多的前 MAX_SOURCES 个行星作为攻击源
    my_planets.sort(key=lambda p: -p.ships)
    my_planets = my_planets[:MAX_SOURCES]

    atoms = []
    for src in my_planets:
        if src.ships <= 1:
            continue
        max_send = int(src.ships) - 1  # 保留至少一个
        targets = top_targets_for_player(src, world, player_id, MAX_TARGETS_PER_SRC)
        for tid in targets:
            tgt = world.planets[tid]
            for ships in ship_options(src, tgt, max_send):
                atom = make_atom(src, tgt, ships, world)
                if atom:
                    atoms.append(atom)

    candidates = [[]]
    if not atoms:
        cache[player_id] = candidates
        return candidates

    # 将同一源行星的原子动作中最佳的一个收集起来，按价值排序
    best_per_src = {}
    for atom in atoms:
        cur = best_per_src.get(atom.src_id)
        if cur is None or atom.value > cur.value:
            best_per_src[atom.src_id] = atom
    ranked = sorted(best_per_src.values(), key=lambda a: -a.value)

    cur_set = []
    for atom in ranked:
        cur_set = list(cur_set) + [atom]
        candidates.append(cur_set)
        if len(candidates) >= max_sets + 1:
            break

    # 对于价值最高的源，加入其第二佳候选作为备选集合
    if len(ranked) >= 1 and len(candidates) <= max_sets:
        top_src = ranked[0].src_id
        alts = [a for a in atoms if a.src_id == top_src and a is not ranked[0]]
        alts.sort(key=lambda a: -a.value)
        for alt in alts[:1]:
            candidates.append([alt])

    cache[player_id] = candidates
    return candidates


# ============================================================
# 一步前瞻评估：我方行动 + 对手最佳应对
# ============================================================

def _atom_to_action(atom, player_id):
    """将 Atom 转为 (player_id, target_id, eta, ships) 元组，用于投影。"""
    return (player_id, atom.target_id, atom.eta, atom.ships)


def best_response(world, opp_id, my_acts, deadline, max_eval=OPP_MAX_EVAL):
    """
    模拟对手 opp_id 在看到我方行动 my_acts 后的简化最佳应对：
    从对手的候选集合中选择使其 V 值最大的集合。
    返回对手的 Atom 列表（可能为空）。
    """
    cache_key = (opp_id, len(my_acts))
    if cache_key in world._best_response_cache:
        return world._best_response_cache[cache_key]
    candidates = generate_candidates(world, opp_id)
    if not candidates:
        world._best_response_cache[cache_key] = []
        return []

    base = list(my_acts)
    best = []
    best_v = -float("inf")
    for opp_atoms in candidates[:max_eval]:
        if time.perf_counter() > deadline:
            break
        all_acts = base + [_atom_to_action(a, opp_id) for a in opp_atoms]
        timelines = project_state(world, all_acts)
        v = V_at_horizon(timelines, world, opp_id, HORIZON_VALUE)
        if v > best_v:
            best_v = v
            best = opp_atoms

    world._best_response_cache[cache_key] = best
    return best


def evaluate(world, my_atoms, deadline):
    """
    一步前瞻评估：假设我执行 my_atoms，每个对手都会做出 best_response，
    然后计算 delta_V(我, 最强对手)。
    返回 delta_V 值。
    """
    my_acts = [_atom_to_action(a, world.my_id) for a in my_atoms]
    full_acts = list(my_acts)

    n_opps = max(1, len(world.opponent_ids))
    remaining = max(0.0, deadline - time.perf_counter())
    per_opp = max(0.02, remaining * OPP_TIME_FRACTION / n_opps)

    for opp_id in world.opponent_ids:
        if time.perf_counter() >= deadline:
            break
        opp_dl = min(deadline, time.perf_counter() + per_opp)
        opp_atoms = best_response(world, opp_id, my_acts, opp_dl)
        full_acts.extend(_atom_to_action(a, opp_id) for a in opp_atoms)

    timelines = project_state(world, full_acts)
    return delta_V(timelines, world, world.my_id)


def immediate_value(atoms):
    """计算候选集合的启发式总价值，用于基线比较。"""
    return sum(a.value for a in atoms)


# ============================================================
# Agent 入口函数
# ============================================================

def materialize_actions(atoms):
    """将 Atom 列表转换为比赛要求的动作格式 [[src_id, angle, ships], ...]"""
    return [[a.src_id, float(a.angle), int(a.ships)] for a in atoms]


def _read(obs, key, default=None):
    """读取观测值的辅助函数，兼容 dict 和对象属性。"""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def build_world(obs):
    """从原始观测数据结构构建 WorldState 快照。"""
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
    """
    主 agent 函数。
    1. 构建世界状态
    2. 生成候选动作集合
    3. 计算每个候选的 delta_V（含对手应对），选择最优
    4. 若超时则退回最高启发式价值的候选
    返回动作列表。
    """
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

    # 按启发式价值排序，作为备用基线
    candidates_sorted = sorted(candidates, key=lambda c: -immediate_value(c))
    fallback = candidates_sorted[0] if candidates_sorted else []

    # 总是评估“无操作”以建立 ΔV 基线
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

    return materialize_actions(best)