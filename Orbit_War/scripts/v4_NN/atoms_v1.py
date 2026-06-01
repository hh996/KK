"""v1 原子工厂，适配 enrich 后的 v4 WorldState。"""
import math
from collections import namedtuple

from physics import HORIZON_SIM, compute_intercept_with_detour, fleet_speed

Atom = namedtuple("Atom", ("src_id", "target_id", "ships", "angle", "eta", "value"))

HORIZON_VALUE = 120
EARLY_GAME_LIMIT = 40
MAX_TARGETS_PER_SRC = 4
MAX_SOURCES = 6
FRONTIER_BONUS = 4.0
THIRD_PARTY_SENSE_ETA = 4
THIRD_PARTY_BONUS = 1.25
DISTANCE_DISCOUNT_SCALE = 30.0
THREAT_BONUS_THRESHOLD = -10.0
BREAKEVEN_PENALTY_SCALE = 50.0
EXCESS_PENALTY_FACTOR = 2.0
VULTURE_MULT = 2.5
STRONG_ENEMY_PENALTY = 0.6
SNIPE_VALUE_MULTIPLIER = 2.0
NEUTRAL_ACCEPT_RATIO = 1.0
MIN_ACCEPT_RATIO = 1.0


def atom_tuple(atom):
    return (atom.src_id, atom.target_id, atom.ships, atom.angle, atom.eta)


def macro_key(macro):
    if not macro:
        return ()
    return tuple(sorted(atom_tuple(a) for a in macro))


def macro_to_env(macro):
    return [[a.src_id, float(a.angle), int(a.ships)] for a in macro]


def macro_score(macro):
    if not macro:
        return 0.0
    return sum(a.value for a in macro)


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


def needed_for_capture(target, eta, world, player_id=None):
    if player_id is None:
        player_id = world.my_id
    if target is None:
        return 1
    if target.owner == -1:
        return target.ships + 1
    base = target.ships
    prod = 0.0
    if target.owner != player_id:
        prod = target.production * max(0, int(math.ceil(eta)))
    enemy_support = 0.0
    for turn, ships in world.enemy_incoming_by_target.get(target.id, {}).items():
        if turn <= math.ceil(eta):
            enemy_support += ships
    required = max(1, math.ceil(base + prod + enemy_support)) + 1
    already = world.my_incoming.get(target.id, 0)
    return max(1, required - already)


def is_saturated(target, world, player_id=None):
    if player_id is None:
        player_id = world.my_id
    if target.owner == player_id:
        return True
    if target.owner == -1:
        return world.my_incoming.get(target.id, 0) >= target.ships + 1
    need = needed_for_capture(
        target, world.my_incoming_max_eta.get(target.id, 2.0), world, player_id
    )
    return world.my_incoming.get(target.id, 0) >= need


def world_get_intercept(world, src_id, target_id, ships):
    key = (src_id, target_id, int(ships))
    if key in world._intercept_cache:
        return world._intercept_cache[key]
    src = world.planets.get(src_id)
    tgt = world.planets.get(target_id)
    if src is None or tgt is None:
        return None
    result = compute_intercept_with_detour(src, tgt, ships, world)
    world._intercept_cache[key] = result
    return result


def top_targets_for_player(src, world, player_id, top_k):
    cache_key = (src.id, player_id)
    cached = world._top_targets_cache.get(cache_key)
    if cached is not None:
        return cached
    candidates = []
    is_early = world.step_count < EARLY_GAME_LIMIT
    for tid, tgt in world.planets.items():
        if tgt.owner == player_id or tid == src.id:
            continue
        if is_saturated(tgt, world, player_id):
            continue
        if tid in world.covered_neutrals:
            continue
        if tid in world.comet_ids:
            if comet_remaining_turns(world.cid_to_group.get(tid), tid) <= 1:
                continue
        d = math.hypot(src.x - tgt.x, src.y - tgt.y)
        if d < 1:
            continue
        proj_ships = max(int(tgt.ships) + 5, 10)
        eta = d / fleet_speed(proj_ships)
        if eta > HORIZON_SIM * 0.8:
            continue
        infl = world.influence_by_id.get(tid, 0.0)
        cost = max(1, int(tgt.ships) + 1)
        val = tgt.production / (cost * 0.4 + 2.0 * eta + 5.0)
        val *= 1.0 / (1.0 + eta / DISTANCE_DISCOUNT_SCALE)
        if tgt.owner == -1:
            val *= 1.2
            if infl > 10:
                val *= 1.3
        else:
            if infl < THREAT_BONUS_THRESHOLD:
                val *= 1.5
        third_party_eta = float("inf")
        for opp_id in world.opponent_ids:
            for fid, (feta, ftid) in world.fleet_arrivals.items():
                if ftid == tid and world.fleet_by_id[fid].owner == opp_id:
                    if feta < THIRD_PARTY_SENSE_ETA and feta < third_party_eta:
                        third_party_eta = feta
        if third_party_eta < THIRD_PARTY_SENSE_ETA:
            val *= THIRD_PARTY_BONUS
        if is_early:
            val *= world.early_neutral_bonus if tgt.owner == -1 else world.early_enemy_penalty
        candidates.append((val, tid))
    candidates.sort(reverse=True)
    result = [tid for _, tid in candidates[:top_k]]
    world._top_targets_cache[cache_key] = result
    return result


def ship_options(src, tgt, max_ships, world, player_id, remaining_need=None):
    if max_ships <= 0 or remaining_need is None or remaining_need <= 0:
        return []
    if max_ships < remaining_need:
        return []
    opts = {remaining_need, min(max_ships, remaining_need + 2), min(max_ships, remaining_need + 5)}
    return sorted(o for o in opts if 1 <= o <= max_ships)


def make_atom(src, tgt, ships, world, player_id, eta_precomputed=None):
    aim = world_get_intercept(world, src.id, tgt.id, ships)
    if aim is None:
        return None
    angle, eta, _ = aim
    if eta > HORIZON_SIM:
        return None
    remaining = HORIZON_VALUE
    if tgt.id in world.comet_ids:
        remaining = min(remaining, comet_remaining_turns(world.cid_to_group.get(tgt.id), tgt.id))
    need = needed_for_capture(tgt, eta, world, player_id)
    excess = max(0, ships - need)
    production_gain = tgt.production * max(0, remaining - int(eta))
    effective_cost = need + excess * EXCESS_PENALTY_FACTOR
    roi = production_gain / (effective_cost * (eta + 1.0)) if effective_cost > 0 else 0.0
    breakeven_penalty = 1.0 / (1.0 + (effective_cost / max(tgt.production, 0.1)) / BREAKEVEN_PENALTY_SCALE)
    distance_penalty = 1.0 / (1.0 + eta / DISTANCE_DISCOUNT_SCALE)
    infl = world.influence_by_id.get(tgt.id, 0.0)
    threat_mult = 1.5 if infl < THREAT_BONUS_THRESHOLD and tgt.owner != -1 else 1.0
    vuln_mult = 1.0
    if tgt.owner not in (-1, player_id):
        owner_info = world.player_analysis.get(tgt.owner, {})
        if owner_info.get("is_weak"):
            vuln_mult = VULTURE_MULT
        elif owner_info.get("is_strong"):
            vuln_mult = STRONG_ENEMY_PENALTY
    is_early = world.step_count < EARLY_GAME_LIMIT
    is_late = world.remaining_steps < 40
    if is_early:
        phase_mult = 2.0 if tgt.owner == -1 else 0.4
    elif is_late:
        phase_mult = 1.5 if tgt.owner != -1 else 0.8
    else:
        phase_mult = 1.0
    capture_bonus = 1.15 if tgt.owner not in (-1, player_id) else 1.0
    value = roi * threat_mult * vuln_mult * phase_mult * breakeven_penalty * distance_penalty * capture_bonus
    return Atom(src.id, tgt.id, ships, angle, eta, value)


def build_attack_atoms(world, player_id):
    my_planets = [p for p in world.planet_list if p.owner == player_id]
    if not my_planets:
        return []
    my_planets.sort(key=lambda p: -p.ships)
    my_planets = my_planets[:MAX_SOURCES]
    atoms = []
    for src in my_planets:
        min_g = world.dynamic_min_garrison.get(src.id, world.min_garrison_base)
        max_send = int(src.ships) - min_g
        if max_send <= 0:
            continue
        for tid in top_targets_for_player(src, world, player_id, MAX_TARGETS_PER_SRC):
            tgt = world.planets[tid]
            aim_est = world_get_intercept(world, src.id, tid, max_send)
            if aim_est is None:
                continue
            eta_est = aim_est[1]
            remaining_need = needed_for_capture(tgt, eta_est, world, player_id)
            for ships in ship_options(src, tgt, max_send, world, player_id, remaining_need):
                atom = make_atom(src, tgt, ships, world, player_id)
                if atom:
                    atoms.append(atom)
    if getattr(world, "snipe_enabled", True):
        atoms.extend(_build_snipe_atoms(world, player_id, my_planets))
    atoms.sort(key=lambda a: -a.value)
    return atoms


def _build_snipe_atoms(world, player_id, my_planets):
    atoms = []
    for tid, tgt in world.planets.items():
        if tgt.owner != -1 or is_saturated(tgt, world, player_id):
            continue
        if tid in world.comet_ids and comet_remaining_turns(world.cid_to_group.get(tid), tid) <= 2:
            continue
        enemy_eta = float("inf")
        for fid, (eta, t_id) in world.fleet_arrivals.items():
            if t_id == tid and world.fleet_by_id[fid].owner != player_id:
                enemy_eta = min(enemy_eta, eta)
        if enemy_eta == float("inf") or enemy_eta < 2:
            continue
        need_ships = needed_for_capture(tgt, enemy_eta - 1, world, player_id)
        for src in my_planets:
            min_g = world.dynamic_min_garrison.get(src.id, world.min_garrison_base)
            available = max(0, int(src.ships) - min_g)
            if available < need_ships:
                continue
            aim = world_get_intercept(world, src.id, tid, need_ships)
            if aim is None:
                continue
            angle, eta, _ = aim
            if eta <= enemy_eta - 1:
                send = min(available, need_ships)
                remaining = HORIZON_VALUE
                if tid in world.comet_ids:
                    remaining = min(remaining, comet_remaining_turns(world.cid_to_group.get(tid), tid))
                production_gain = tgt.production * max(0, remaining - int(eta))
                cost = send + int(eta) * 0.5
                value = (production_gain / (cost + 1.0)) * SNIPE_VALUE_MULTIPLIER
                atoms.append(Atom(src.id, tid, send, angle, eta, value))
                break
    return atoms


def build_reinforce_atoms(world, player_id, max_count=3, eta_horizon=12):
    from config import REINFORCE_DEFICIT_SCALE

    atoms = []
    my_planets = [p for p in world.planet_list if p.owner == player_id]
    safe = [p for p in my_planets if p.production >= 1]
    for tgt in my_planets:
        threat = 0.0
        min_eta = float("inf")
        for turn, ships in world.enemy_incoming_by_target.get(tgt.id, {}).items():
            if turn <= eta_horizon:
                threat += ships
                min_eta = min(min_eta, float(turn))
        if threat <= 0:
            continue
        eta_arr = min_eta if min_eta < float("inf") else 1.0
        need = threat - tgt.ships + tgt.production * max(0, int(math.ceil(eta_arr))) + 1
        need = int(math.ceil(need * REINFORCE_DEFICIT_SCALE))
        if need <= 0:
            continue
        donors = sorted(
            [p for p in safe if p.id != tgt.id],
            key=lambda p: math.hypot(p.x - tgt.x, p.y - tgt.y),
        )
        for src in donors[:4]:
            min_g = world.dynamic_min_garrison.get(src.id, world.min_garrison_base)
            avail = int(src.ships) - min_g
            if avail < need:
                continue
            send = min(avail, need)
            aim = world_get_intercept(world, src.id, tgt.id, send)
            if aim is None:
                continue
            angle, eta, _ = aim
            value = tgt.production * 3.0 / (send + eta + 1.0)
            atoms.append(Atom(src.id, tgt.id, send, angle, eta, value))
            break
    atoms.sort(key=lambda a: -a.value)
    return atoms[:max_count]


def atoms_compatible(world, player_id, a, b):
    sa, qa = a.src_id, a.ships
    sb, qb = b.src_id, b.ships
    pa = world.planets.get(sa)
    pb = world.planets.get(sb)
    if not pa or not pb:
        return False
    if pa.owner != player_id or pb.owner != player_id:
        return False
    if sa == sb:
        return pa.ships >= qa + qb
    return pa.ships >= qa and pb.ships >= qb
