"""v4 路线 B 冒烟测试。"""
import time
import random

from kaggle_environments import make

from candidates_b import generate_candidates_b, macro_key
from config import CANDIDATE_CAP
from train import build_world_from_obs, _read
from network import PolicyValueNetworkB
from config import CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS, MAX_MACRO_SLOTS
from mcts_b import MCTSB


def test_candidate_counts(n_games=5):
    env = make("orbit_wars", debug=False, configuration={"episodeSteps": 500, "seed": 42})
    counts = []

    def rec_agent(obs, config):
        pid = _read(obs, "player", 0)
        w = build_world_from_obs(obs, pid, 2, episode_seed=42)
        macros = generate_candidates_b(w, pid)
        counts.append(len(macros))
        assert len(macros) <= CANDIDATE_CAP
        assert () in macros or macros[0] == ()
        for m in macros:
            macro_key(m)
        return []

    env.run([rec_agent, rec_agent])
    for _ in range(n_games - 1):
        env = make("orbit_wars", debug=False, configuration={"episodeSteps": 200, "seed": random.randint(0, 9999)})
        env.run([rec_agent, rec_agent])
    p95 = sorted(counts)[int(len(counts) * 0.95)] if counts else 0
    print(f"candidate counts: min={min(counts)} max={max(counts)} p95={p95} n={len(counts)}")
    assert p95 <= CANDIDATE_CAP + 1
    return True


def test_mcts_one_step():
    env = make("orbit_wars", debug=False, configuration={"episodeSteps": 500, "seed": 1})
    obs = None

    def cap(obs_in, config):
        nonlocal obs
        obs = obs_in
        return []

    env.run([cap, cap])
    obs = env.steps[1][0].observation if hasattr(env.steps[1][0], "observation") else env.steps[1][0]["observation"]
    world = build_world_from_obs(obs, 0, 2, episode_seed=1)
    macros = generate_candidates_b(world, 0)
    net = PolicyValueNetworkB(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS, MAX_MACRO_SLOTS)
    t0 = time.perf_counter()
    bm, probs = MCTSB(net, num_simulations=8).run(world, macros, obs_for_deepseek=obs)
    dt = time.perf_counter() - t0
    print(f"MCTS 8 sim: {dt:.3f}s macro={len(bm) if bm else 0} probs_sum={probs.sum():.2f}")
    assert bm is not None or not macros
    return True


if __name__ == "__main__":
    test_candidate_counts()
    test_mcts_one_step()
    print("smoke_checks OK")
