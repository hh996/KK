"""统计 deepseek 走法被 candidates_b 覆盖的比例。"""
import importlib.util
import os

from kaggle_environments import make

from candidates_b import (
    generate_candidates_b,
    inject_teacher_macro,
    is_good_macro_match,
    match_env_action_to_macro,
    match_quality_score,
)
from train import build_world_from_obs, _read

_V1_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "v1_rule", "v1_deepseek", "main.py"))


def run_coverage(n_games=20, num_agents=2):
    spec = importlib.util.spec_from_file_location("v1ds_cov", _V1_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    deepseek = mod.agent

    total_steps = 0
    fuzzy_matched = 0
    strict_matched = 0
    strict_after_teacher = 0
    teacher_added = 0
    empty_match = 0

    for g in range(n_games):
        env = make("orbit_wars", debug=False, configuration={"episodeSteps": 500, "seed": 1000 + g})
        step_records = []

        def wrap_agent(pid):
            def agent(obs, config):
                world = build_world_from_obs(
                    obs, pid, num_agents,
                    episode_seed=(getattr(env, "info", None) or {}).get("seed"),
                )
                macros = generate_candidates_b(world, pid)
                ds_action = deepseek(obs, config) or []
                mac, idx = match_env_action_to_macro(ds_action, macros)
                q = match_quality_score(ds_action, mac)
                _, chosen, new_idx = inject_teacher_macro(macros, ds_action, world, pid)
                step_records.append((ds_action, mac, idx, len(macros), q, chosen, new_idx))
                return ds_action
            return agent

        env.run([wrap_agent(i) for i in range(num_agents)])
        for ds_action, mac, idx, n_cand, q, chosen, new_idx in step_records:
            total_steps += 1
            if mac is not None and idx >= 0:
                if not ds_action and not mac:
                    fuzzy_matched += 1
                elif ds_action and mac:
                    fuzzy_matched += 1
                elif not ds_action:
                    empty_match += 1
            if is_good_macro_match(ds_action, mac):
                strict_matched += 1
            if is_good_macro_match(ds_action, chosen):
                strict_after_teacher += 1
            if new_idx == 1 or (new_idx == 0 and ds_action):
                if not is_good_macro_match(ds_action, mac):
                    teacher_added += 1

    fuzzy_rate = fuzzy_matched / max(1, total_steps)
    strict_rate = strict_matched / max(1, total_steps)
    after_rate = strict_after_teacher / max(1, total_steps)
    print(
        f"games={n_games} steps={total_steps} "
        f"fuzzy={fuzzy_matched} ({fuzzy_rate:.1%}) "
        f"strict={strict_matched} ({strict_rate:.1%}) "
        f"strict+teacher={strict_after_teacher} ({after_rate:.1%}) "
        f"teacher_added={teacher_added} empty_ds={empty_match}"
    )
    return after_rate


if __name__ == "__main__":
    run_coverage(n_games=10)
