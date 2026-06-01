"""路线 B 专用候选宏生成：v1 原子重组 + 增补，上限 CANDIDATE_CAP。"""
import math
from itertools import combinations

from config import (
    BEAM_CUTOFF_RATIO,
    BEAM_WIDTH,
    CANDIDATE_CAP,
    MATCH_ANGLE_TOL_DEG,
    MATCH_QUALITY_THRESHOLD,
    MAX_COMBO_MACROS,
    MAX_DUAL_SWARMS,
    MAX_SOURCES_PROBE,
    MAX_TRIPLE_SWARMS,
    MAX_TWO_TARGET_MACROS,
    REINFORCE_ETA_HORIZON,
    REINFORCE_MAX,
    SWARM_ETA_GAP,
    SWARM_MIN_TARGET_PROD,
    TWO_TARGET_ETA_GAP,
)
from world_enrich import enrich_world
from atoms_v1 import (
    Atom,
    build_attack_atoms,
    build_reinforce_atoms,
    atoms_compatible,
    macro_key,
    macro_score,
    needed_for_capture,
    world_get_intercept,
    make_atom,
)


def _probe_sources(world, tgt, player_id, my_planets):
    sorted_srcs = sorted(my_planets, key=lambda s: math.hypot(s.x - tgt.x, s.y - tgt.y))
    probe = {}
    for s in sorted_srcs[:MAX_SOURCES_PROBE]:
        min_g = world.dynamic_min_garrison.get(s.id, world.min_garrison_base)
        avail = max(0, int(s.ships) - min_g)
        if avail < 2:
            continue
        aim = world_get_intercept(world, s.id, tgt.id, max(int(tgt.ships) + 1, 5))
        if aim is None:
            continue
        probe[s.id] = (aim[1], avail, s)
    return probe


def _split_sends(total_needed, avails):
    """avails: list of (src_id, avail); return list of sends or None."""
    n = len(avails)
    total_avail = sum(a for _, a in avails)
    if total_avail < total_needed:
        return None
    sends = [max(1, int(total_needed * a / total_avail)) for _, a in avails]
    for _ in range(8):
        ssum = sum(sends)
        if ssum == total_needed:
            break
        if ssum < total_needed:
            for i in range(n):
                if sends[i] < avails[i][1]:
                    sends[i] += 1
                    break
        else:
            for i in range(n):
                if sends[i] > 1:
                    sends[i] -= 1
                    break
    if sum(sends) < total_needed:
        return None
    for i, (_, av) in enumerate(avails):
        if sends[i] > av or sends[i] <= 0:
            return None
    return sends


def _build_dual_swarms(world, player_id, my_planets):
    swarms = []
    for tid, tgt in world.planets.items():
        if tgt.owner == player_id:
            continue
        probe = _probe_sources(world, tgt, player_id, my_planets)
        ids = list(probe.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                eta_i, avail_i, si = probe[ids[i]]
                eta_j, avail_j, sj = probe[ids[j]]
                if abs(eta_i - eta_j) > SWARM_ETA_GAP:
                    continue
                max_eta = max(eta_i, eta_j)
                total_needed = needed_for_capture(tgt, max_eta, world, player_id)
                for ratio in ((0.5, 0.5), (0.6, 0.4), (0.4, 0.6)):
                    send1 = min(avail_i, max(1, int(total_needed * ratio[0])))
                    send2 = min(avail_j, total_needed - send1)
                    if send1 + send2 < total_needed:
                        continue
                    a1 = make_atom(si, tgt, send1, world, player_id)
                    a2 = make_atom(sj, tgt, send2, world, player_id)
                    if a1 is None or a2 is None:
                        continue
                    if abs(a1.eta - a2.eta) > SWARM_ETA_GAP:
                        continue
                    swarms.append((macro_score([a1, a2]), (a1, a2)))
    swarms.sort(key=lambda x: -x[0])
    return [m for _, m in swarms[:MAX_DUAL_SWARMS]]


def _build_triple_swarms(world, player_id, my_planets):
    swarms = []
    for tid, tgt in world.planets.items():
        if tgt.owner == player_id:
            continue
        if tgt.owner == -1 and tgt.production < SWARM_MIN_TARGET_PROD:
            continue
        probe = _probe_sources(world, tgt, player_id, my_planets)
        if len(probe) < 3:
            continue
        for combo in combinations(probe.keys(), 3):
            etas = [probe[sid][0] for sid in combo]
            if max(etas) - min(etas) > SWARM_ETA_GAP:
                continue
            avails = [(sid, probe[sid][1]) for sid in combo]
            max_eta = max(etas)
            total_needed = needed_for_capture(tgt, max_eta, world, player_id)
            sends = _split_sends(total_needed, avails)
            if sends is None:
                continue
            atoms = []
            ok = True
            for sid, send in zip(combo, sends):
                s = probe[sid][2]
                a = make_atom(s, tgt, send, world, player_id)
                if a is None:
                    ok = False
                    break
                atoms.append(a)
            if ok and len(atoms) == 3:
                swarms.append((macro_score(atoms), tuple(atoms)))
    swarms.sort(key=lambda x: -x[0])
    return [m for _, m in swarms[:MAX_TRIPLE_SWARMS]]


def _build_single_macros(atoms, world, player_id):
    if not atoms:
        return []
    best_val = atoms[0].value
    selected = []
    seen_target = set()
    seen_source = set()
    for atom in atoms:
        if atom.value < best_val * BEAM_CUTOFF_RATIO:
            break
        if atom.target_id in seen_target or atom.src_id in seen_source:
            continue
        need = needed_for_capture(
            world.planets[atom.target_id], atom.eta, world, player_id
        )
        if atom.ships < need:
            continue
        selected.append(atom)
        seen_target.add(atom.target_id)
        seen_source.add(atom.src_id)
        if len(selected) >= BEAM_WIDTH:
            break
    return [tuple([a]) for a in selected]


def _build_combos(attacks, reinforces, world, player_id):
    combos = []
    for atk in attacks[:3]:
        for ref in reinforces[:2]:
            if len(atk) != 1 or len(ref) != 1:
                continue
            a, r = atk[0], ref[0]
            if a.src_id == r.src_id:
                continue
            if not atoms_compatible(world, player_id, a, r):
                continue
            combos.append((macro_score([a, r]), (a, r)))
    combos.sort(key=lambda x: -x[0])
    return [m for _, m in combos[:MAX_COMBO_MACROS]]


def _build_two_target(attacks, world, player_id):
    singles = [a for mac in attacks if len(mac) == 1 for a in mac]
    pairs = []
    for i, a1 in enumerate(singles[:4]):
        for a2 in singles[i + 1 : 4]:
            if a1.target_id == a2.target_id or a1.src_id == a2.src_id:
                continue
            if abs(a1.eta - a2.eta) > TWO_TARGET_ETA_GAP:
                continue
            if not atoms_compatible(world, player_id, a1, a2):
                continue
            pairs.append((macro_score([a1, a2]), (a1, a2)))
    pairs.sort(key=lambda x: -x[0])
    return [m for _, m in pairs[:MAX_TWO_TARGET_MACROS]]


def generate_candidates_b(world, player_id):
    enrich_world(world, player_id)
    cache = getattr(world, "_candidate_cache_b", {})
    if player_id in cache:
        return cache[player_id]

    my_planets = [p for p in world.planet_list if p.owner == player_id]
    macros = [()]
    if not my_planets:
        cache[player_id] = macros
        world._candidate_cache_b = cache
        return macros

    attack_atoms = build_attack_atoms(world, player_id)
    reinforce_atoms = build_reinforce_atoms(
        world, player_id, max_count=REINFORCE_MAX, eta_horizon=REINFORCE_ETA_HORIZON
    )

    singles_atk = _build_single_macros(attack_atoms, world, player_id)
    singles_ref = [tuple([a]) for a in reinforce_atoms]
    dual = _build_dual_swarms(world, player_id, my_planets)
    triple = _build_triple_swarms(world, player_id, my_planets)
    combos = _build_combos(singles_atk, singles_ref, world, player_id)
    two_tgt = _build_two_target(singles_atk, world, player_id)

    ordered = []
    seen = set()
    for group in (singles_ref, dual, triple, combos, two_tgt, singles_atk):
        for mac in group:
            k = macro_key(mac)
            if k in seen:
                continue
            seen.add(k)
            ordered.append(mac)

    ordered.sort(key=lambda m: -macro_score(m))
    for mac in ordered:
        if len(macros) >= CANDIDATE_CAP:
            break
        macros.append(mac)

    cache[player_id] = macros
    world._candidate_cache_b = cache
    return macros


def match_quality_score(env_action, macro):
    """env 动作与宏的一致程度，1.0 为完全匹配。"""
    if not env_action and not macro:
        return 1.0
    if bool(env_action) != bool(macro):
        return 0.0
    ea_by_src = {int(e[0]): (float(e[1]), float(e[2])) for e in env_action}
    if not ea_by_src:
        return 0.0
    covered = 0
    for src_id, (ea_angle, ea_ships) in ea_by_src.items():
        for atom in macro:
            if atom.src_id != src_id:
                continue
            da = abs(float(atom.angle) - ea_angle) % 360
            da = min(da, 360 - da)
            ship_tol = max(1.0, 0.03 * max(1.0, ea_ships))
            if da <= MATCH_ANGLE_TOL_DEG and abs(float(atom.ships) - ea_ships) <= ship_tol:
                covered += 1
                break
    return covered / len(ea_by_src)


def is_good_macro_match(env_action, macro, threshold=None):
    if threshold is None:
        threshold = MATCH_QUALITY_THRESHOLD
    return match_quality_score(env_action, macro) >= threshold


def env_action_to_teacher_macro(world, player_id, env_action):
    """将 deepseek env 动作转为可执行的 teacher 宏（用于 IL / 诊断）。"""
    if not env_action:
        return ()
    atoms = []
    for entry in env_action:
        src_id = int(entry[0])
        angle = float(entry[1])
        ships = int(entry[2])
        src = world.planets.get(src_id)
        if src is None or src.owner != player_id:
            return None
        if ships <= 0 or ships > int(src.ships):
            return None
        target_id = -1
        eta = 15.0
        value = 12.0
        for tid, tgt in world.planets.items():
            if tgt.owner == player_id or tid == src_id:
                continue
            aim = world_get_intercept(world, src_id, tid, ships)
            if aim is None:
                continue
            aim_angle, aim_eta, _ = aim
            da = abs(float(aim_angle) - angle) % 360
            da = min(da, 360 - da)
            if da <= MATCH_ANGLE_TOL_DEG and aim_eta < eta:
                target_id = tid
                eta = float(aim_eta)
        atoms.append(Atom(src_id, target_id, ships, angle, eta, value))
    return tuple(atoms) if atoms else None


def inject_teacher_macro(macros, env_action, world, player_id):
    """匹配不足时在候选表插入 teacher 宏。返回 (macros, chosen_macro, idx)。"""
    matched, idx = match_env_action_to_macro(env_action, macros)
    if is_good_macro_match(env_action, matched):
        return macros, matched, idx

    teacher = env_action_to_teacher_macro(world, player_id, env_action)
    if teacher is None:
        return macros, matched, idx

    tk = macro_key(teacher)
    for i, mac in enumerate(macros):
        if macro_key(mac) == tk:
            return macros, mac, i

    new_macros = list(macros)
    insert_pos = 1 if new_macros and not new_macros[0] else 0
    new_macros.insert(insert_pos, teacher)
    while len(new_macros) > CANDIDATE_CAP:
        if len(new_macros) > 1:
            new_macros.pop()
        else:
            break
    return new_macros, teacher, insert_pos


def match_env_action_to_macro(env_action, macros):
    """将 env [[src, angle, ships],...] 匹配到最近候选宏。"""
    if not macros:
        return None, -1
    if not env_action:
        for i, mac in enumerate(macros):
            if not mac:
                return mac, i
        return macros[0], 0
    ea_by_src = {int(e[0]): (float(e[1]), float(e[2])) for e in env_action}
    best_idx, best_score = 0, -1.0
    for i, mac in enumerate(macros):
        score = 0.0
        for atom in mac:
            if atom.src_id in ea_by_src:
                ea_angle, ea_ships = ea_by_src[atom.src_id]
                da = abs(float(atom.angle) - ea_angle) % 360
                da = min(da, 360 - da)
                score += 1.0 / (1.0 + da) + 0.1 / (1.0 + abs(float(atom.ships) - ea_ships))
        if score > best_score:
            best_score = score
            best_idx = i
    return macros[best_idx], best_idx


def roi_fallback_macro(macros):
    best, best_s = None, -1.0
    for mac in macros:
        if not mac:
            continue
        s = macro_score(mac)
        if s > best_s:
            best_s = s
            best = mac
    return best if best is not None else (macros[0] if macros else ())
