"""
验证 physics.py 的 WorldState.step 与官方 kaggle_environments 行为是否一致。
双方均使用空动作（不发射任何舰队）。
"""
import random
import numpy as np
from kaggle_environments import make
from physics import (
    Planet, Fleet, WorldState,
    MAX_SPEED, SUN_R, BOARD_SIZE as PHYS_BOARD_SIZE
)
from train import build_world_from_obs

# ---------- 空动作 agent ----------
def empty_agent(obs, config):
    return []

def run_official_env(seed=None, steps=30):
    """运行官方环境若干步（双方均不发射），返回每步的快照。"""
    config = {"episodeSteps": 500}
    if seed is not None:
        config["seed"] = seed
    env = make("orbit_wars", debug=False, configuration=config)
    env.run([empty_agent, empty_agent])
    return env.steps[:steps+1]

def run_physics_sim(initial_obs, seed, steps=30):
    world = build_world_from_obs(
        initial_obs, player_id=0, num_training_agents=2, episode_seed=seed,
    )
    p0 = world.planet_list[0]
    print(f"After build: ships={p0.ships}, production={p0.production}")
    print(f"base_omega: {world.base_omega}")

    snapshots = []
    for step_idx in range(steps + 1):
        p0 = world.planet_list[0]
        # 每次循环都打印，看看船只数何时变
        print(f"Step {step_idx} BEFORE snap: ships={p0.ships}, production={p0.production}")
        planets_snap = [(p.id, p.owner, p.x, p.y, p.ships, p.production) for p in world.planet_list]
        fleets_snap = [(f.id, f.owner, f.x, f.y, f.ships, f.angle) for f in world.fleets]
        snapshots.append((planets_snap, fleets_snap))
        if world.is_terminal():
            break
        world.step({pid: [] for pid in world.player_ids})
    return snapshots

def compare_snapshots(official_steps, physics_snapshots, max_diff=0.01):
    print("official angular_velocity:", official[0][0]["observation"]["angular_velocity"])
    for step_idx in range(min(len(official_steps), len(physics_snapshots))):
        off = official_steps[step_idx]
        phy_planets, phy_fleets = physics_snapshots[step_idx]

        obs0 = off[0]["observation"]
        off_planets = obs0["planets"] if obs0 else []
        off_fleets = obs0["fleets"] if obs0 else []

        off_p_dict = {p[0]: p for p in off_planets}
        phy_p_dict = {p[0]: p for p in phy_planets}  # p 是 (id, owner, x, y, ships, production)

        for pid, p in off_p_dict.items():
            pp = phy_p_dict.get(pid)
            if pp is None:
                print(f"Step {step_idx}: Planet {pid} missing in physics")
                continue
            if p[1] != pp[1]:
                print(f"Step {step_idx} Planet {pid}: owner off={p[1]} phy={pp[1]}")
            if abs(p[2] - pp[2]) > max_diff or abs(p[3] - pp[3]) > max_diff:
                print(f"Step {step_idx} Planet {pid}: pos off=({p[2]:.3f},{p[3]:.3f}) phy=({pp[2]:.3f},{pp[3]:.3f})")
            if abs(p[5] - pp[4]) > 0.5:   # 关键修正：pp[4] 是 ships
                print(f"Step {step_idx} Planet {pid}: ships off={p[5]} phy={pp[4]}")
        # 舰队检查保持不变
        if off_fleets or phy_fleets:
            print(f"Step {step_idx}: unexpected fleets off={len(off_fleets)} phy={len(phy_fleets)}")
    print("Comparison done.")

if __name__ == "__main__":
    SEED = 42
    official = run_official_env(seed=SEED, steps=60)
    init_obs = official[0][0]["observation"]
    physics_snaps = run_physics_sim(init_obs, seed=SEED, steps=60)
    compare_snapshots(official, physics_snaps)