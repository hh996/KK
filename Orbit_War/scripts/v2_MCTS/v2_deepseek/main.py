"""
Orbit Wars - Full Rule Boost with Influence Map & Multi-Player Exploitation + Precision Deployment

整合特性：
  - 精确物理引擎（真实速度、带符号角速度、迭代拦截、绕路、路径阻挡检测）
  - 安全边际折扣、动态防守底线、对手采样
  - 多源协同、自适应束宽度、终局动态权重、焦土撤退、行星保卫
  - 精确时机狙击、动态模式切换
  - 影响力地图（威胁/优势评估）
  - 多人博弈分析（趁虚而入）
  - ROI复合评分（回本惩罚、距离折扣、威胁乘数等）
  - 精确需求计算（生产+敌方支援+已派遣），中立星不扣减已派遣
  - 智能舰船选项（刚好够、极小余量），禁止跨源碎片攻击
  - 超额派遣惩罚、饱和目标跳过
  - 第三方感知（其他玩家即将攻击的目标优先抢占）
"""

import math
import time
import random
from collections import defaultdict
import copy
import random

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

# 策略超参数
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

LATE_GAME_THRESHOLD = 40
LATE_SHIP_BONUS_STEP = 0.02

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

# 精确部署控制
MIN_ATTACK_RATIO = 0.3          # 单次攻击至少占需求的30%（仅用于 ship_options）
MIN_ACCEPT_RATIO = 1.0          # 敌方行星接受阈值 100%
NEUTRAL_ACCEPT_RATIO = 1.0      # 中立星强制 100%

# 第三方感知
THIRD_PARTY_SENSE_ETA = 4       # 其他玩家舰队在4回合内到达则触发感知
THIRD_PARTY_BONUS = 1.25        # 价值乘数

DISTANCE_WEIGHT = 2.0


# ============================================================
# Data classes
# ============================================================

# ============================================================
# MCTS 嵌入层（放在 agent 函数之前）
# ============================================================

class MCTSNode:
    __slots__ = ('world', 'atoms', 'parent', 'children',
                 'visits', 'total_value', 'untried_actions')
    def __init__(self, world, atoms=None, parent=None):
        self.world = world
        self.atoms = atoms if atoms is not None else []  # 从父节点到本节点执行的原子组合
        self.parent = parent
        self.children = {}          # tuple(atom组合标识) -> MCTSNode
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = None # 尚未展开的候选组合列表

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c=1.4):
        """UCB1 选择最佳子节点"""
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
        """展开一个未尝试的动作组合，生成新世界状态"""
        if self.untried_actions is None:
            # 第一次调用：用规则生成当前世界下所有可能的动作组合
            self.untried_actions = generate_candidates(self.world, self.world.my_id)
            # 确保空动作也被考虑（跳过本回合）
            if [] not in self.untried_actions:
                self.untried_actions.append([])
            random.shuffle(self.untried_actions)  # 避免顺序偏差

        if not self.untried_actions:
            return None

        # 取一个未尝试的动作组合
        action_atoms = self.untried_actions.pop()
        # 用动作组合和预估对手动作，将世界推进一个“宏步骤”（例如1回合）
        next_world = fast_forward_world(self.world, action_atoms, advance_turns=1)
        child = MCTSNode(next_world, atoms=action_atoms, parent=self)
        # 使用元组标识（Atom 列表不能直接 hash，这里以 Atom 的关键属性构成 key）
        key = tuple((a.src_id, a.target_id, a.ships, a.eta) for a in action_atoms)
        self.children[key] = child
        return child

    def rollout(self, depth=3, deadline=None):
        """
        从本节点世界开始，用快速策略（贪心选最高 immediate_value）模拟 depth 步，
        返回最终 delta_V 的估计值。
        """
        world = self.world
        total_atoms = list(self.atoms)
        for _ in range(depth):
            if deadline and time.perf_counter() > deadline:
                break
            cands = generate_candidates(world, world.my_id)
            if not cands:
                break
            # 贪心选择 immediate_value 最高的组合
            best_cand = max(cands, key=lambda c: immediate_value(c))
            total_atoms.extend(best_cand)
            # 带对手的快速推进
            world = fast_forward_world(world, best_cand, advance_turns=1)
        # 叶子节点估值：复用已有的 evaluate，但这里我们做一个轻量版本（不采样全部对手）
        try:
            return evaluate(world, total_atoms, deadline or time.perf_counter() + 0.1)
        except Exception:
            return immediate_value(total_atoms)


def fast_forward_world(world, my_atoms, advance_turns=1):
    """
    根据我方动作组合和对手的规则最佳动作，推进世界状态 advance_turns 回合。
    返回一个新的 WorldState。
    """
    # 构建所有玩家的动作序列（project_state 需要的格式）
    all_actions = []

    # 1) 我方的原子动作
    for a in my_atoms:
        all_actions.append(_atom_to_action(a, world.my_id))

    # 2) 对手也用相同的规则引擎来决定他们的动作（快速自对弈）
    #    为了速度，每个对手只选第一个候选组合（最优 immediate_value）
    for opp_id in world.opponent_ids:
        opp_cands = generate_candidates(world, opp_id, max_sets=1)
        if opp_cands and opp_cands[0]:
            for a in opp_cands[0]:
                all_actions.append(_atom_to_action(a, opp_id))

    # 3) 利用 project_state 得到时间线（它会模拟生产、战斗等）
    timelines = project_state(world, all_actions)

    # 4) 从时间线中提取 advance_turns 后的行星数组，构造新 WorldState
    #    这里简化：假设 advance_turns 比较小，直接从时间线里取最后时刻的状态
    #    更精确的做法是取第 advance_turns 步的 owner 和 ships，但那需要知道具体步长。
    #    为简便，我们使用 HORIZON_SIM 步之后的远景，因为 MCTS 会继续往下搜。
    #    我们实际上想获得“执行完这些动作后的世界”，project_state 已经是假设所有动作同时发生，
    #    我们可以直接取时间线中每颗行星最终 owner 和 ships（此时动作已经飞行、战斗完毕）。
    #    但由于舰队还在路上，我们需要从中提取“当前”的舰队状态，这是 project_state 没有的。
    #    为了工程可用，我们放宽要求：推进后世界的行星采用 timelines 最后一步的 owner/ships，
    #    但不保留舰队（视为已消失或在途中，MCTS 只需知道行星分布）。
    #    这足以支撑短深度规划。

    new_planets = []
    for pid, p in world.planets.items():
        # 寻找该行星在 timelines 中最后一刻的状态
        tl = timelines.get(pid, [])
        if tl:
            # 取最后一步（索引 -1）
            owner, ships = tl[-1]
            # 生产产能和半径不变
            new_planets.append(Planet(pid, owner, p.x, p.y, p.radius, max(0, int(ships)), p.production))
        else:
            new_planets.append(p)

    # 构建新的 WorldState（忽略舰队和初始行星的一些细节，但 omega 等信息保留）
    new_world = WorldState(
        player=world.my_id,
        step=world.step + advance_turns,
        planets=new_planets,
        fleets=[],  # 清净世界，舰队已结算或忽略
        initial_planets=[],  # 不重要
        base_omega=0.0,      # 后续会用 omega_map 复现
        comets=[], comet_ids=set()
    )
    return new_world


def mcts_search(init_world, iterations=150, max_seconds=0.8):
    """在给定世界状态下运行 MCTS，返回最佳原子组合（可能为空）"""
    deadline = time.perf_counter() + max_seconds
    root = MCTSNode(init_world)

    for _ in range(iterations):
        if time.perf_counter() > deadline:
            break

        node = root
        # 1. Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            if node is None:
                break

        # 2. Expansion
        if not node.is_fully_expanded():
            new_node = node.expand()
            if new_node is not None:
                node = new_node

        # 3. Simulation (rollout)
        value = node.rollout(depth=2, deadline=deadline)

        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    # 选择访问次数最多的子节点对应的动作
    if not root.children:
        return []  # 无动作可做
    best_child = max(root.children.values(), key=lambda c: c.visits)
    return best_child.atoms


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
# Physics (unchanged)
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

        # 我方已派遣统计
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

        # 敌方即将到达统计
        self.enemy_incoming_by_target = defaultdict(lambda: defaultdict(float))
        for fid, (eta, tid) in self.fleet_arrivals.items():
            if tid is None:
                continue
            f = self.fleet_by_id[fid]
            if f.owner != player and f.owner != -1:
                turn = max(1, int(math.ceil(eta)))
                self.enemy_incoming_by_target[tid][turn] += f.ships

        # 已覆盖的中立星（基于已派遣量）
        self.covered_neutrals = set()
        for pid, p in self.planets.items():
            if p.owner != -1:
                continue
            # 中立星不扣除已派遣，直接用当前守军+1判断是否足够
            if self.my_incoming.get(pid, 0) >= p.ships + 1:
                self.covered_neutrals.add(pid)

        # 动态实力比
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
        self.influence = compute_influence_map(self.planets, self.my_id, self.omega_map, self.cid_to_group, self.comet_ids)
        self.influence_by_id = {}
        for idx, p in enumerate(self.planet_list):
            self.influence_by_id[p.id] = self.influence[idx]

        # 多人分析
        self.player_analysis = analyze_players(self.planets, self.my_id)

        # 动态防守底线
        self.dynamic_min_garrison = {}
        for p in planets:
            if p.owner == player:
                self.dynamic_min_garrison[p.id] = self._compute_dynamic_min(p)
            else:
                self.dynamic_min_garrison[p.id] = 0

        # 时间线（用于 doomed 判断）
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
        infl = self.influence_by_id.get(planet.id, 0.0)
        threat_bonus = max(0, -infl * 0.5)
        if min_eta <= 15:
            base = max(8, int(planet.ships * 0.5))
        elif min_eta <= 30:
            base = max(5, int(planet.ships * 0.3))
        else:
            base = self.min_garrison_base
        return min(int(planet.ships * 0.7), base + int(threat_bonus))

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
# Precision Deployment Helpers (P0: neutral absolute demand)
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

def needed_for_capture(target, eta, world):
    """
    精确计算在舰队到达时占领目标所需的最小舰船数。
    对中立星：直接使用当前驻军+1，不考虑已派遣和敌方支援。
    对敌方行星：驻军 + 生产*eta + 敌方支援 - 我方已派遣 + 1。
    """
    if target is None:
        return 1
    # 中立星绝对需求
    if target.owner == -1:
        return target.ships + 1

    # 敌方行星
    base = target.ships
    prod = 0.0
    if target.owner != world.my_id:
        prod = target.production * max(0, int(math.ceil(eta)))
    enemy_support = 0.0
    for turn, ships in world.enemy_incoming_by_target.get(target.id, {}).items():
        if turn <= math.ceil(eta):
            enemy_support += ships
    required = max(1, math.ceil(base + prod + enemy_support)) + 1
    already = world.my_incoming.get(target.id, 0)
    needed = max(1, required - already)
    return needed

def is_saturated(target, world):
    """
    已饱和判断：对我方已派遣舰队足够占领的目标跳过。
    中立星：已派遣量 >= 守军+1 即为饱和。
    """
    if target.owner == world.my_id:
        return True
    if target.owner == -1:
        return world.my_incoming.get(target.id, 0) >= target.ships + 1
    # 敌方行星使用所需需求判断
    need = needed_for_capture(target, world.my_incoming_max_eta.get(target.id, 2.0), world)
    return world.my_incoming.get(target.id, 0) >= need


# ============================================================
# Candidate generation (P0: no cross-source pollution, P1: third-party sense)
# ============================================================

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
        distance_weight = 2.0  # 新参数，可放在常量区
        val = tgt.production / (cost * 0.4 + distance_weight * eta + 5.0)
        # 额外距离衰减（与 make_atom 保持一致）
        distance_factor = 1.0 / (1.0 + eta / DISTANCE_DISCOUNT_SCALE)
        val *= distance_factor

        if tgt.owner == -1:
            val *= 1.2
            if infl > 10:
                val *= 1.3
        else:
            if infl < THREAT_BONUS_THRESHOLD:
                val *= 1.5

        # P1: 第三方感知
        third_party_eta = float("inf")
        for opp_id in world.opponent_ids:
            if opp_id == player:
                continue
            for fid, (feta, ftid) in world.fleet_arrivals.items():
                if ftid == tid and world.fleet_by_id[fid].owner == opp_id:
                    if feta < THIRD_PARTY_SENSE_ETA and feta < third_party_eta:
                        third_party_eta = feta
        if third_party_eta < THIRD_PARTY_SENSE_ETA:
            val *= THIRD_PARTY_BONUS  # 优先抢占

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
    """
    生成刚好满足剩余需求的舰船数量，禁止碎片攻击。
    若 max_ships < remaining_need，则不生成任何选项（避免小原子）。
    """
    if max_ships <= 0:
        return []
    if remaining_need is None or remaining_need <= 0:
        return []
    if max_ships < remaining_need:
        return []  # 单源无法满足需求，放弃
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

    value = roi * threat_mult * vuln_mult * phase_mult * breakeven_penalty * distance_penalty * capture_bonus
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

    # 所有原子独立生成，不分跨源累积
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
            # 使用需求（中立星不扣减已派遣，敌方扣减）
            global_need = needed_for_capture(tgt, eta_est, world)
            # 剩余需求就是 global_need，因为不再跨源累积本回合计划
            remaining_need = global_need
            for ships in ship_options(src, tgt, max_send, world,
                                      eta_estimate=eta_est,
                                      remaining_need=remaining_need):
                atom = make_atom(src, tgt, ships, world, eta_precomputed=eta_est)
                if atom:
                    atoms.append(atom)

    # 狙击
    if world.snipe_enabled and player_id == world.my_id:
        for tid, tgt in world.planets.items():
            if tgt.owner != -1 or is_saturated(tgt, world):
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
            need_ships = needed_for_capture(tgt, enemy_eta - 1, world)  # 中立星不扣已派遣
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
                    send = min(available, need_ships)
                    remaining = HORIZON_VALUE
                    if tid in world.comet_ids:
                        life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
                        remaining = min(remaining, life)
                    production_gain = tgt.production * max(0, remaining - int(eta))
                    cost = send + int(eta) * 0.5
                    value = (production_gain / (cost + 1.0)) * SNIPE_VALUE_MULTIPLIER
                    snipe_atom = Atom(src.id, tid, send, angle, eta, value)
                    atoms.append(snipe_atom)
                    break

    atoms.sort(key=lambda a: -a.value)

    candidates = [[]]
    if not atoms:
        cache[player_id] = candidates
        return candidates

    # ---- 筛选阶段：禁止非协同多源攻击 ----
    best_val = atoms[0].value
    selected_atoms = []
    committed_targets = set()   # 记录已经有原子被选中的目标（普通候选，非双源协同）
    target_eta_range = {}

    for atom in atoms:
        if atom.value < best_val * BEAM_CUTOFF_RATIO:
            break

        tid = atom.target_id
        tgt = world.planets[tid]

        # 如果该目标已被选中，且当前原子不是双源协同，则跳过（禁止碎片化）
        if tid in committed_targets:
            continue

        # 计算需求阈值
        if tid in target_eta_range:
            current_max_eta = max(target_eta_range[tid][1], atom.eta)
        else:
            current_max_eta = atom.eta
        need = needed_for_capture(tgt, current_max_eta, world)
        # 中立星强制 100% 需求，敌方行星使用 MIN_ACCEPT_RATIO (1.0)
        accept_threshold = need * NEUTRAL_ACCEPT_RATIO if tgt.owner == -1 else need * MIN_ACCEPT_RATIO
        if atom.ships < accept_threshold:
            continue

        # 时间协同：如果该目标已有其他原子（不可能，因为已跳过），保留逻辑以防万一
        if tid in target_eta_range:
            min_et, max_et = target_eta_range[tid]
            if atom.eta < min_et - 1.0 or atom.eta > max_et + 1.0:
                continue

        # 避免同一源重复使用
        if atom.src_id not in {a.src_id for a in selected_atoms}:
            selected_atoms.append(atom)
            committed_targets.add(tid)
            if tid not in target_eta_range:
                target_eta_range[tid] = (atom.eta, atom.eta)
            else:
                min_et, max_et = target_eta_range[tid]
                target_eta_range[tid] = (min(min_et, atom.eta), max(max_et, atom.eta))
            if len(selected_atoms) >= max_sets:
                break

    cur_set = []
    for atom in selected_atoms:
        cur_set = list(cur_set) + [atom]
        candidates.append(cur_set)
    if [] not in candidates:
        candidates.append([])

    # ---- 双源协同（预先规划，时间+总量对齐） ----
    dual_candidates = []
    for tid, tgt in world.planets.items():
        if tgt.owner == player_id or is_saturated(tgt, world):
            continue
        if tgt.ships < world.dual_source_min_ships and tgt.owner == -1:
            continue
        remaining_life = HORIZON_VALUE
        if tid in world.comet_ids:
            life = comet_remaining_turns(world.cid_to_group.get(tid), tid)
            remaining_life = min(remaining_life, life)
        if remaining_life < 15:
            continue

        closest = sorted(my_planets, key=lambda p: math.hypot(p.x - tgt.x, p.y - tgt.y))
        if len(closest) < 2:
            continue
        s1, s2 = closest[0], closest[1]
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
        if abs(eta1 - eta2) > DUAL_ETA_TOLERANCE:
            continue

        max_eta = max(eta1, eta2)
        total_needed = needed_for_capture(tgt, max_eta, world)
        total_needed = max(total_needed, int(tgt.ships) + int(tgt.production * max_eta) + 1)

        total_avail = avail1 + avail2
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
        if abs(a1.eta - a2.eta) > DUAL_ETA_TOLERANCE:
            continue
        a1.value *= 1.2
        a2.value *= 1.2
        dual_candidates.append([a1, a2])
        if len(dual_candidates) >= MAX_DUAL_CANDIDATES:
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

    # 尝试 MCTS 搜索
    try:
        # 给 MCTS 留出 70% 的剩余时间，保证不超时
        mcts_budget = min(0.7 * (deadline - start), deadline - start - 0.05)
        best_atoms = mcts_search(world, iterations=200, max_seconds=mcts_budget)
        final_moves = materialize_actions(best_atoms)
    except Exception as e:
        # MCTS 出错时回退到纯规则
        best_atoms = None

    if best_atoms is None:
        # 纯规则逻辑（原来的候选选择）
        candidates = generate_candidates(world, world.my_id)
        if not candidates:
            return []
        candidates_sorted = sorted(candidates, key=lambda c: -immediate_value(c))
        fallback = candidates_sorted[0] if candidates_sorted else []
        best = fallback
        best_score = evaluate(world, [], deadline)

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

    # 焦土撤退逻辑保留（可以放在 MCTS 之外作为安全网）
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

    return final_moves