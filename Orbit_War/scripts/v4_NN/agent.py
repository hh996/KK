"""Hybrid 提交 agent：candidates_b + MCTS + evaluate/ROI fallback。"""
import os
import time
import importlib.util

import torch

from config import (
    BOARD_SIZE, CHANNELS, RES_BLOCKS, RES_FILTERS, MAX_MACRO_SLOTS,
    MCTS_SIMULATIONS, SUBMIT_MCTS_SIMULATIONS, CHECKPOINT_DIR, LATEST_CHECKPOINT, CHECKPOINT_SUFFIX,
    DEVICE, CONF_THRESHOLD, HYBRID_V1_ON_LOW_CONF,
)
from network import PolicyValueNetworkB
from train import build_world_from_obs, _read
from candidates_b import generate_candidates_b, roi_fallback_macro
from atoms_v1 import macro_to_env
from mcts_b import MCTSB, _evaluate_pick

_net = None
_v1_agent = None


def _load_net():
    global _net
    if _net is not None:
        return _net
    _net = PolicyValueNetworkB(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS, MAX_MACRO_SLOTS)
    path = os.path.join(CHECKPOINT_DIR, f"{LATEST_CHECKPOINT}{CHECKPOINT_SUFFIX}.pt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "checkpoints", f"{LATEST_CHECKPOINT}{CHECKPOINT_SUFFIX}.pt")
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        _net.load_state_dict(ck["model_state_dict"])
    _net.to(DEVICE)
    _net.eval()
    return _net


def _load_v1_fallback():
    global _v1_agent
    if _v1_agent is not None:
        return _v1_agent
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "v1_rule", "v1_deepseek", "main.py"))
    spec = importlib.util.spec_from_file_location("v1ds_sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _v1_agent = mod.agent
    return _v1_agent


def agent(obs, config=None):
    start = time.perf_counter()
    timeout = _read(config, "actTimeout", 1.0) if config else 1.0
    deadline = start + min(0.85, max(0.5, timeout * 0.85))

    player_id = _read(obs, "player", 0)
    try:
        world = build_world_from_obs(obs, player_id, num_training_agents=4)
    except Exception:
        return _load_v1_fallback()(obs, config)

    if not [p for p in world.planet_list if p.owner == player_id]:
        return []

    macros = generate_candidates_b(world, player_id)
    if not macros:
        return []

    remaining = deadline - time.perf_counter()
    if remaining < 0.05:
        return macro_to_env(roi_fallback_macro(macros))

    n_sim = max(24, min(SUBMIT_MCTS_SIMULATIONS, int(remaining * 100)))
    try:
        net = _load_net()
        mcts = MCTSB(net, num_simulations=n_sim)
        macro, probs = mcts.run(world, macros, obs_for_deepseek=obs, config=config)
    except Exception:
        macro, probs = None, None

    low_conf = macro is None or (probs is not None and probs.max() < CONF_THRESHOLD)
    if low_conf and HYBRID_V1_ON_LOW_CONF and time.perf_counter() < deadline - 0.12:
        try:
            return _load_v1_fallback()(obs, config) or []
        except Exception:
            pass

    if low_conf:
        if time.perf_counter() < deadline - 0.05:
            try:
                ev_mac = _evaluate_pick(world, macros, config)
                if ev_mac:
                    macro = ev_mac
            except Exception:
                pass
        if macro is None:
            macro = roi_fallback_macro(macros)
        if macro is None and time.perf_counter() < deadline - 0.02:
            try:
                return _load_v1_fallback()(obs, config) or []
            except Exception:
                pass

    return macro_to_env(macro) if macro is not None else []
