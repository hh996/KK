"""为 v4 WorldState 补齐 v1 候选生成所需的战略字段。"""
import math
from collections import defaultdict

from physics import fleet_speed, predict_fleet_arrival

MIN_GARRISON_BASE = 3
INFLUENCE_DECAY = 0.06
BEAM_WIDTH = 6
DUAL_SOURCE_MIN_SHIPS = 10
EARLY_NEUTRAL_BONUS = 2.0
EARLY_ENEMY_PENALTY = 0.5


def compute_influence_map(planet_list, player_id):
    n = len(planet_list)
    influence = [0.0] * n
    for i, p in enumerate(planet_list):
        for j, q in enumerate(planet_list):
            if i == j:
                continue
            dist = math.hypot(p.x - q.x, p.y - q.y)
            approx_time = dist / fleet_speed(15) + 1.0
            decay = math.exp(-INFLUENCE_DECAY * approx_time)
            if q.owner == player_id:
                influence[i] += q.ships * decay
            elif q.owner != -1:
                influence[i] -= q.ships * decay
    return influence


def analyze_players(planets, my_id):
    players = defaultdict(lambda: {"total_ships": 0, "total_production": 0, "planet_count": 0})
    for p in planets.values():
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


def enrich_world(world, player_id=None):
    """填充/刷新 v1 风格战略字段；MCTS 每步 step 后应重新调用。"""
    if player_id is None:
        player_id = world.my_id

    owners = set()
    for p in world.planet_list:
        if p.owner != -1:
            owners.add(p.owner)
    for f in world.fleets:
        owners.add(f.owner)
    owners.add(player_id)
    world.player_ids_found = sorted(owners)
    world.opponent_ids = [o for o in world.player_ids_found if o != player_id]

    world.fleet_arrivals = {}
    for f in world.fleets:
        world.fleet_arrivals[f.id] = predict_fleet_arrival(
            f, world.planet_list, world.omega_map, world.cid_to_group, world.comet_ids,
            max_speed=world.ship_speed, board_size=world.board_size, sun_r=world.sun_r,
        )

    world.my_incoming = defaultdict(float)
    world.my_incoming_max_eta = {}
    world.enemy_incoming_by_target = defaultdict(lambda: defaultdict(float))
    for fid, (eta, tid) in world.fleet_arrivals.items():
        if tid is None:
            continue
        f = world.fleet_by_id[fid]
        if f.owner == player_id:
            world.my_incoming[tid] += f.ships
            prev = world.my_incoming_max_eta.get(tid, 0)
            world.my_incoming_max_eta[tid] = max(prev, eta)
        elif f.owner != -1:
            turn = max(1, int(math.ceil(eta)))
            world.enemy_incoming_by_target[tid][turn] += f.ships

    world.covered_neutrals = set()
    for pid, p in world.planets.items():
        if p.owner != -1:
            continue
        if world.my_incoming.get(pid, 0) >= p.ships + 1:
            world.covered_neutrals.add(pid)

    my_total = sum(p.ships for p in world.planet_list if p.owner == player_id)
    enemy_total = 0
    for p in world.planet_list:
        if p.owner not in (-1, player_id):
            enemy_total += p.ships
    for f in world.fleets:
        if f.owner == player_id:
            my_total += f.ships
        elif f.owner != -1:
            enemy_total += f.ships
    strength_ratio = my_total / max(1, enemy_total)

    if strength_ratio < 0.8:
        world.early_neutral_bonus = EARLY_NEUTRAL_BONUS * 1.5
        world.early_enemy_penalty = EARLY_ENEMY_PENALTY * 0.8
        world.dual_source_min_ships = max(5, DUAL_SOURCE_MIN_SHIPS - 5)
        world.beam_width = min(BEAM_WIDTH + 2, 8)
        world.min_garrison_base = max(1, MIN_GARRISON_BASE - 1)
        world.snipe_enabled = True
    elif strength_ratio > 1.5:
        world.early_neutral_bonus = EARLY_NEUTRAL_BONUS * 0.8
        world.early_enemy_penalty = EARLY_ENEMY_PENALTY * 1.2
        world.dual_source_min_ships = DUAL_SOURCE_MIN_SHIPS + 5
        world.beam_width = max(4, BEAM_WIDTH - 1)
        world.min_garrison_base = MIN_GARRISON_BASE + 2
        world.snipe_enabled = True
    else:
        world.early_neutral_bonus = EARLY_NEUTRAL_BONUS
        world.early_enemy_penalty = EARLY_ENEMY_PENALTY
        world.dual_source_min_ships = DUAL_SOURCE_MIN_SHIPS
        world.beam_width = BEAM_WIDTH
        world.min_garrison_base = MIN_GARRISON_BASE
        world.snipe_enabled = True

    infl = compute_influence_map(world.planet_list, player_id)
    world.influence_by_id = {p.id: infl[i] for i, p in enumerate(world.planet_list)}
    world.player_analysis = analyze_players(world.planets, player_id)

    world.dynamic_min_garrison = {}
    for p in world.planet_list:
        if p.owner == player_id:
            world.dynamic_min_garrison[p.id] = _compute_dynamic_min(world, p, player_id)
        else:
            world.dynamic_min_garrison[p.id] = 0

    world._intercept_cache = {}
    world._top_targets_cache = {}
    world._candidate_cache = {}
    world._enriched = True


def _compute_dynamic_min(world, planet, player_id):
    if not world.opponent_ids:
        return world.min_garrison_base
    min_eta = float("inf")
    for opp_id in world.opponent_ids:
        for opp_p in world.planet_list:
            if opp_p.owner != opp_id:
                continue
            d = math.hypot(opp_p.x - planet.x, opp_p.y - planet.y)
            speed = fleet_speed(max(1, opp_p.ships))
            eta = d / speed if speed > 0 else float("inf")
            min_eta = min(min_eta, eta)
    infl = world.influence_by_id.get(planet.id, 0.0)
    threat_bonus = max(0, -infl * 0.5)
    if min_eta <= 15:
        base = max(8, int(planet.ships * 0.5))
    elif min_eta <= 30:
        base = max(5, int(planet.ships * 0.3))
    else:
        base = world.min_garrison_base
    return min(int(planet.ships * 0.7), base + int(threat_bonus))
