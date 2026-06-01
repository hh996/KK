"""统计 deepseek 着法未被 candidates_b 精确覆盖的类型（诊断用）。"""
import importlib.util
import os
from collections import Counter

from kaggle_environments import make

from candidates_b import (
    generate_candidates_b,
    inject_teacher_macro,
    is_good_macro_match,
    match_env_action_to_macro,
    match_quality_score,
)
from config import MATCH_QUALITY_THRESHOLD
from train import build_world_from_obs, _read

_V1_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "v1_rule", "v1_deepseek", "main.py")
)


def _classify_gap(env_action, macros, quality):
    if not macros or len(macros) <= 1:
        return "no_candidates"
    if not env_action:
        return "pass_not_in_list"
    n_fleets = len(env_action)
    matched, _ = match_env_action_to_macro(env_action, macros)
    if quality >= MATCH_QUALITY_THRESHOLD:
        return "ok"
    if n_fleets == 1:
        src = int(env_action[0][0])
        ea_angle, ea_ships = float(env_action[0][1]), int(env_action[0][2])
        if matched and matched:
            atom = next((a for a in matched if a.src_id == src), None)
            if atom is None:
                return "single_src_missing"
            da = abs(float(atom.angle) - ea_angle) % 360
            da = min(da, 360 - da)
            if da > 8:
                return "single_angle_gap"
            if abs(int(atom.ships) - ea_ships) > max(1, 0.03 * ea_ships):
                return "single_ships_gap"
        return "single_other"
    if n_fleets == 2:
        return "dual_fleet_gap"
    if n_fleets >= 3:
        return "triple_plus_gap"
    return "unknown"


def run_gap_report(n_games=20, num_agents=2, verbose=False):
    spec = importlib.util.spec_from_file_location("v1ds_gap", _V1_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    deepseek = mod.agent

    stats = Counter()
    teacher_inject = 0
    total_steps = 0
    good_before = 0
    good_after = 0

    for g in range(n_games):
        env = make("orbit_wars", debug=False, configuration={"episodeSteps": 500, "seed": 2000 + g})

        def wrap_agent(pid):
            def agent(obs, config):
                nonlocal total_steps, good_before, good_after, teacher_inject
                world = build_world_from_obs(
                    obs, pid, num_agents,
                    episode_seed=(getattr(env, "info", None) or {}).get("seed"),
                )
                macros = generate_candidates_b(world, pid)
                ds_action = deepseek(obs, config) or []
                matched, _ = match_env_action_to_macro(ds_action, macros)
                q = match_quality_score(ds_action, matched)
                total_steps += 1
                if is_good_macro_match(ds_action, matched):
                    good_before += 1
                kind = _classify_gap(ds_action, macros, q)
                stats[kind] += 1
                _, chosen, _ = inject_teacher_macro(macros, ds_action, world, pid)
                if is_good_macro_match(ds_action, chosen):
                    good_after += 1
                if not is_good_macro_match(ds_action, matched):
                    teacher_inject += 1
                if verbose and kind not in ("ok",):
                    print(f"  g{g} step gap={kind} fleets={len(ds_action)} q={q:.2f}")
                return ds_action
            return agent

        env.run([wrap_agent(i) for i in range(num_agents)])

    print(f"games={n_games} steps={total_steps}")
    print(f"good_match_before={good_before}/{total_steps} ({good_before / max(1, total_steps):.1%})")
    print(f"good_match_after_teacher={good_after}/{total_steps} ({good_after / max(1, total_steps):.1%})")
    print(f"teacher_inject_steps={teacher_inject}")
    print("gap_breakdown:")
    for k, v in stats.most_common():
        print(f"  {k}: {v} ({v / max(1, total_steps):.1%})")
    return stats


if __name__ == "__main__":
    run_gap_report(n_games=10)
