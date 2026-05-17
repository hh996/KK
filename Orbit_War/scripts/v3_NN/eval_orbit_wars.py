"""
评估模块：net_agent（带 MCTS）对阵各类对手。

公开函数：
  evaluate_net_vs_random(net, ...)        → vs 内置 random_agent
  evaluate_net_vs_deepseek(net, ...)      → vs v1_deepseek rule-based agent
  evaluate_net_vs_checkpoint(net, path, ...) → vs 历史 checkpoint（双方均带 MCTS）
"""

import importlib
import importlib.util
import os
import random

import torch

from kaggle_environments import make

from config import (
    C_PUCT,
    CHANNELS,
    BOARD_SIZE,
    DEVICE,
    MAX_GAME_STEPS,
    MCTS_SIMULATIONS,
    RES_BLOCKS,
    RES_FILTERS,
)
from mcts import MCTS

_V1_DEEPSEEK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "v1_rule", "v1_deepseek", "main.py")
)

_EVAL_SIMULATIONS = min(48, MCTS_SIMULATIONS)


def macro_to_env_moves(macro):
    return [[atom[0], atom[3], atom[2]] for atom in macro]


# ---------------------------------------------------------------------------
# 公共 helper：构建带 MCTS 的 agent 函数
# ---------------------------------------------------------------------------

def _make_net_agent(net, player_id, env, num_agents):
    """返回一个 Kaggle-compatible agent 函数，内部使用 net + MCTS 决策。"""
    import train as tm

    def agent(obs, config):
        info = getattr(env, "info", None) or {}
        episode_seed = info.get("seed")
        cs = float(tm._read(config, "cometSpeed", 4.0) or 4.0)
        sp = float(tm._read(config, "shipSpeed", tm.MAX_SPEED) or tm.MAX_SPEED)
        su = float(tm._read(config, "sunRadius", tm.SUN_R) or tm.SUN_R)
        bd = float(tm._read(config, "boardSize", tm.PHYS_BOARD_SIZE) or tm.PHYS_BOARD_SIZE)
        world = tm.build_world_from_obs(
            obs, player_id, num_agents,
            episode_seed=episode_seed, comet_speed=cs,
            ship_speed=sp, sun_radius=su, board_size=bd,
        )
        if world.is_terminal():
            return []
        macs = world.get_legal_macro_actions(player_id)
        if not macs:
            return []
        mc = MCTS(net, num_simulations=_EVAL_SIMULATIONS, c_puct=C_PUCT)
        bm, _ = mc.run(world, macs)
        return macro_to_env_moves(bm) if bm is not None else []

    return agent


# ---------------------------------------------------------------------------
# 通用评估循环：net 对阵任意 opponent_fn
# ---------------------------------------------------------------------------

def evaluate_net_vs_agent(net, opponent_fn, episodes=12, num_agents=2, seed=42):
    """
    net（带 MCTS）对阵 opponent_fn（任意 Kaggle-compatible agent）。
    hero 位置随机，返回 (wins, total_episodes)。
    """
    net.eval()
    rng = random.Random(seed)
    wins = 0
    for _ in range(episodes):
        hero = rng.randrange(num_agents)
        env = make(
            "orbit_wars",
            debug=False,
            configuration={
                "episodeSteps": MAX_GAME_STEPS,
                "seed": rng.randint(0, 2 ** 30 - 1),
            },
        )
        roster = [
            _make_net_agent(net, i, env, num_agents) if i == hero else opponent_fn
            for i in range(num_agents)
        ]
        env.run(roster)
        rr = env.steps[-1][hero].reward if hero < len(env.steps[-1]) else 0
        if rr is not None and float(rr) > 0:
            wins += 1
    return wins, episodes


# ---------------------------------------------------------------------------
# vs random_agent
# ---------------------------------------------------------------------------

def evaluate_net_vs_random(net, episodes=12, num_agents=2):
    orb_mod = importlib.import_module("kaggle_environments.envs.orbit_wars.orbit_wars")
    return evaluate_net_vs_agent(net, orb_mod.random_agent, episodes, num_agents)


# ---------------------------------------------------------------------------
# vs v1_deepseek rule-based agent
# ---------------------------------------------------------------------------

def evaluate_net_vs_deepseek(net, episodes=12, num_agents=2):
    """net 对阵 v1_deepseek 启发式 agent。"""
    spec = importlib.util.spec_from_file_location("v1_deepseek_main", _V1_DEEPSEEK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return evaluate_net_vs_agent(net, mod.agent, episodes, num_agents)


# ---------------------------------------------------------------------------
# vs 历史 checkpoint（双方均带 MCTS）
# ---------------------------------------------------------------------------

def evaluate_net_vs_checkpoint(net_current, checkpoint_path, episodes=12, num_agents=2):
    """
    当前模型 vs 历史 checkpoint，双方都使用 MCTS。
    用于追踪自对比 ELO / 版本间进步幅度。
    返回 (current_wins, total_episodes)。
    """
    from network import PolicyValueNetwork

    old_net = PolicyValueNetwork(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS).to(DEVICE)
    try:
        ck = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        old_net.load_state_dict(ck["model_state_dict"])
    except Exception as e:
        print(f"[eval] 无法加载 checkpoint {checkpoint_path}: {e}")
        return 0, 0
    old_net.eval()
    net_current.eval()

    rng = random.Random(42)
    wins = 0
    for _ in range(episodes):
        hero = rng.randrange(num_agents)
        env = make(
            "orbit_wars",
            debug=False,
            configuration={
                "episodeSteps": MAX_GAME_STEPS,
                "seed": rng.randint(0, 2 ** 30 - 1),
            },
        )
        roster = [
            _make_net_agent(net_current if i == hero else old_net, i, env, num_agents)
            for i in range(num_agents)
        ]
        env.run(roster)
        rr = env.steps[-1][hero].reward if hero < len(env.steps[-1]) else 0
        if rr is not None and float(rr) > 0:
            wins += 1
    return wins, episodes
