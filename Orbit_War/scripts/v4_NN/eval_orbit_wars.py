"""评估 v4 Hybrid agent vs random / deepseek / starter。"""
import importlib
import importlib.util
import os
import random

import torch
from kaggle_environments import make

from config import (
    BOARD_SIZE, CHANNELS, C_PUCT, DEVICE, MAX_MACRO_SLOTS,
    MCTS_SIMULATIONS, EVAL_MCTS_SIMULATIONS, RES_BLOCKS, RES_FILTERS, orbit_wars_config,
)
from mcts_b import MCTSB
from train import build_world_from_obs, _read
from candidates_b import generate_candidates_b, roi_fallback_macro
from atoms_v1 import macro_to_env
from network import PolicyValueNetworkB

_V1_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "v1_rule", "v1_deepseek", "main.py"))
_EVAL_SIMS = min(EVAL_MCTS_SIMULATIONS, MCTS_SIMULATIONS)


def _make_net_agent(net, player_id, env, num_agents):
    def agent(obs, config):
        world = build_world_from_obs(
            obs, player_id, num_agents,
            episode_seed=(getattr(env, "info", None) or {}).get("seed"),
        )
        if world.is_terminal():
            return []
        macros = generate_candidates_b(world, player_id)
        if not macros:
            return []
        bm, probs = MCTSB(net, num_simulations=_EVAL_SIMS, c_puct=C_PUCT).run(
            world, macros, obs_for_deepseek=obs, config=config
        )
        if bm is None or (probs is not None and probs.max() < 0.1):
            bm = roi_fallback_macro(macros)
        return macro_to_env(bm) if bm is not None else []
    return agent


def evaluate_net_vs_agent(net, opponent_fn, episodes=12, num_agents=2, seed=42):
    net.eval()
    rng = random.Random(seed)
    wins = 0
    for _ in range(episodes):
        hero = rng.randrange(num_agents)
        env = make("orbit_wars", debug=False, configuration=orbit_wars_config(
            seed=rng.randint(0, 2 ** 30 - 1),
        ))
        roster = [
            _make_net_agent(net, i, env, num_agents) if i == hero else opponent_fn
            for i in range(num_agents)
        ]
        env.run(roster)
        rr = env.steps[-1][hero].reward if hero < len(env.steps[-1]) else 0
        if rr is not None and float(rr) > 0:
            wins += 1
    return wins, episodes


def evaluate_net_vs_random(net, episodes=12, num_agents=2):
    orb = importlib.import_module("kaggle_environments.envs.orbit_wars.orbit_wars")
    return evaluate_net_vs_agent(net, orb.random_agent, episodes, num_agents)


def evaluate_net_vs_starter(net, episodes=12, num_agents=2):
    orb = importlib.import_module("kaggle_environments.envs.orbit_wars.orbit_wars")
    return evaluate_net_vs_agent(net, orb.starter_agent, episodes, num_agents)


def evaluate_net_vs_deepseek(net, episodes=12, num_agents=2):
    spec = importlib.util.spec_from_file_location("v1ds_eval", _V1_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return evaluate_net_vs_agent(net, mod.agent, episodes, num_agents)


def load_net_from_checkpoint(path=None):
    net = PolicyValueNetworkB(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS, MAX_MACRO_SLOTS).to(DEVICE)
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "checkpoints", "latest_b.pt")
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        net.load_state_dict(ck["model_state_dict"])
    net.eval()
    return net


if __name__ == "__main__":
    net = load_net_from_checkpoint()
    for name, fn in [("random", evaluate_net_vs_random), ("starter", evaluate_net_vs_starter), ("deepseek", evaluate_net_vs_deepseek)]:
        w, t = fn(net, episodes=8, num_agents=2)
        print(f"vs {name} 2p: {w}/{t}")
