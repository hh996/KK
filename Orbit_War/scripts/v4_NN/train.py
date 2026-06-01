"""路线 B 训练：macro-index IL + 自博弈 + 中断续训（含 replay 持久化）。"""
import gc
import glob
import os
import pickle
import shutil
import signal
import random
import warnings
import multiprocessing as mp
from collections import deque

import numpy as np
import torch
from kaggle_environments import make

warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
)

from value_util import env_terminal_value
from config import *
from physics import Planet, Fleet, WorldState, MAX_SPEED, SUN_R, BOARD_SIZE as PHYS_BOARD_SIZE
from features import encode_state
from network import PolicyValueNetworkB
from mcts_b import MCTSB
from world_enrich import enrich_world
from candidates_b import (
    generate_candidates_b, match_env_action_to_macro, inject_teacher_macro,
    macro_key, roi_fallback_macro,
)
from atoms_v1 import macro_to_env
from load_v3_backbone import load_v3_backbone_into


def _read(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_world_from_obs(
    obs, player_id, num_training_agents, episode_seed=None,
    comet_speed=4.0, ship_speed=None, sun_radius=None, board_size=None,
):
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    initial_planets = _read(obs, "initial_planets", []) or []
    step = int(_read(obs, "step", 0) or 0)
    base_omega = float(_read(obs, "angular_velocity", 0.0) or 0.0)
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])
    player_ids_found = set()
    for p in raw_planets:
        if p[1] != -1:
            player_ids_found.add(p[1])
    for f in raw_fleets:
        player_ids_found.add(f[1])
    planets = [Planet(p[0], p[1], p[2], p[3], p[4], p[5], p[6]) for p in raw_planets]
    fleets = [Fleet(f[0], f[1], f[2], f[3], f[4], f[5], f[6]) for f in raw_fleets]
    ss = float(MAX_SPEED if ship_speed is None else ship_speed)
    sr = float(SUN_R if sun_radius is None else sun_radius)
    bd = float(PHYS_BOARD_SIZE if board_size is None else board_size)
    world = WorldState(
        planets, fleets, initial_planets, step, base_omega, comets, comet_ids,
        sorted(player_ids_found), player_id, num_training_agents=num_training_agents,
        episode_seed=episode_seed, comet_speed=comet_speed, ship_speed=ss,
        sun_radius=sr, board_size=bd,
    )
    enrich_world(world, player_id)
    return world


def build_macro_pi(macros, probs):
    pi = np.zeros(MAX_MACRO_SLOTS, dtype=np.float32)
    n = min(len(macros), MAX_MACRO_SLOTS)
    p = np.asarray(probs[:n], dtype=np.float32)
    if p.sum() <= 0:
        p = np.ones(n, dtype=np.float32) / max(1, n)
    else:
        p = p / p.sum()
    pi[:n] = p
    return pi


_deepseek_agent_fn = None


def _load_deepseek_agent():
    global _deepseek_agent_fn
    if _deepseek_agent_fn is not None:
        return _deepseek_agent_fn
    import importlib.util
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "v1_rule", "v1_deepseek", "main.py"))
    spec = importlib.util.spec_from_file_location("v1_deepseek_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _deepseek_agent_fn = mod.agent
    return _deepseek_agent_fn


def play_one_game(net, opponent=None, n_simulations=TRAIN_MCTS_SIMULATIONS):
    n_agents = 2 if random.random() < TRAIN_TWO_PLAYER_PROB else 4
    env = make("orbit_wars", debug=False, configuration=orbit_wars_config())
    game_history = []

    if opponent is None:
        hero_slots = set(range(n_agents))
    else:
        if n_agents == 2:
            hero_slots = {random.randrange(2)}
        else:
            start = random.randrange(2)
            hero_slots = {start, start + 2}

    def make_net_agent(player_id, train_net, collect_data):
        def agent(obs, config):
            info = getattr(env, "info", None) or {}
            episode_seed = info.get("seed") or 0
            cs = float(_read(config, "cometSpeed", 4.0) or 4.0)
            sp = float(_read(config, "shipSpeed", MAX_SPEED) or MAX_SPEED)
            su = float(_read(config, "sunRadius", SUN_R) or SUN_R)
            bd = float(_read(config, "boardSize", PHYS_BOARD_SIZE) or PHYS_BOARD_SIZE)
            world = build_world_from_obs(
                obs, player_id, n_agents, episode_seed=episode_seed, comet_speed=cs,
                ship_speed=sp, sun_radius=su, board_size=bd,
            )
            if world.is_terminal():
                return []
            macros = generate_candidates_b(world, player_id)
            if not macros:
                return []
            if collect_data:
                state_np = encode_state(
                    world, perspective_player=player_id, device="cpu",
                ).numpy().astype(np.float16)
                mcts = MCTSB(train_net, num_simulations=n_simulations)
                best, probs = mcts.run(world, macros, obs_for_deepseek=obs, config=config)
                if best is None:
                    return []
                macro_pi = build_macro_pi(macros, probs).astype(np.float16)
                game_history.append((player_id, state_np, macro_pi, len(macros)))
                step_num = int(_read(obs, "step", 0) or 0)
                if step_num < TEMPERATURE_STEPS and len(macros) > 1 and probs.sum() > 0:
                    idx = np.random.choice(len(macros), p=probs / probs.sum())
                    chosen = macros[idx]
                else:
                    chosen = best
                return macro_to_env(chosen)
            mcts = MCTSB(train_net, num_simulations=n_simulations)
            best, _ = mcts.run(world, macros, obs_for_deepseek=obs, config=config)
            return macro_to_env(best) if best is not None else []
        return agent

    agents = []
    for i in range(n_agents):
        if i in hero_slots:
            agents.append(make_net_agent(i, net, collect_data=True))
        elif opponent == "deepseek":
            agents.append(_load_deepseek_agent())
        else:
            agents.append(make_net_agent(i, opponent, collect_data=False))
    env.run(agents)

    final_step = env.steps[-1]
    scores = {i: float(s.reward if s.reward is not None else 0.0) for i, s in enumerate(final_step)}
    participant_ids = list(range(n_agents))
    final_obs = final_step[0].observation if hasattr(final_step[0], "observation") else final_step[0].get("observation", {})
    final_ships = {}
    for p in (_read(final_obs, "planets", []) or []):
        if p[1] != -1:
            final_ships[p[1]] = final_ships.get(p[1], 0) + p[5]
    for f in (_read(final_obs, "fleets", []) or []):
        if f[1] != -1:
            final_ships[f[1]] = final_ships.get(f[1], 0) + f[6]

    samples = []
    for pid, state_np, macro_pi, _ in game_history:
        ev = env_terminal_value(scores, pid, participant_ids=participant_ids)
        my_s = float(final_ships.get(pid, 0.0))
        best_other = max((float(final_ships.get(p, 0.0)) for p in participant_ids if p != pid), default=0.0)
        diff = (my_s - best_other) / max(1.0, float(VALUE_TARGET_SCALE))
        target_v = 0.5 * ev + 0.5 * float(np.tanh(diff))
        samples.append((state_np, float(target_v), macro_pi.astype(np.float16)))
    return samples


def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)


def _run_game_worker(args):
    net_sd, opp_info, n_simulations = args
    net = PolicyValueNetworkB(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS, MAX_MACRO_SLOTS)
    net.load_state_dict(net_sd)
    net.eval()
    if isinstance(opp_info, dict):
        opp_net = PolicyValueNetworkB(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS, MAX_MACRO_SLOTS)
        opp_net.load_state_dict(opp_info)
        opp_net.eval()
        opponent = opp_net
    else:
        opponent = opp_info
    return play_one_game(net, opponent=opponent, n_simulations=n_simulations)


_EVAL_SIMS = min(EVAL_MCTS_SIMULATIONS, MCTS_SIMULATIONS)


def _eval_game_worker(args):
    """评估 worker：单局 net vs 对手，返回是否获胜(bool)。4P 时对角线 2 hero 槽位。"""
    import importlib
    import importlib.util as _ilu

    net_sd, opponent_type, hero, num_agents, seed = args

    net = PolicyValueNetworkB(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS, MAX_MACRO_SLOTS)
    net.load_state_dict(net_sd)
    net.eval()

    if opponent_type == "random":
        orb = importlib.import_module("kaggle_environments.envs.orbit_wars.orbit_wars")
        opp_fn = orb.random_agent
    elif opponent_type == "starter":
        orb = importlib.import_module("kaggle_environments.envs.orbit_wars.orbit_wars")
        opp_fn = orb.starter_agent
    else:
        v1_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "v1_rule", "v1_deepseek", "main.py"
        ))
        spec = _ilu.spec_from_file_location("_v1ds_eval", v1_path)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        opp_fn = mod.agent

    env = make("orbit_wars", debug=False, configuration=orbit_wars_config(seed=seed))

    hero_slots = {hero, (hero + 2) % 4} if num_agents == 4 else {hero}

    def _make_net_agent(pid):
        def _agent(obs, config):
            info = getattr(env, "info", None) or {}
            ep_seed = info.get("seed")
            cs = float(_read(config, "cometSpeed", 4.0) or 4.0)
            sp = float(_read(config, "shipSpeed", MAX_SPEED) or MAX_SPEED)
            su = float(_read(config, "sunRadius", SUN_R) or SUN_R)
            bd = float(_read(config, "boardSize", PHYS_BOARD_SIZE) or PHYS_BOARD_SIZE)
            world = build_world_from_obs(
                obs, pid, num_agents, episode_seed=ep_seed,
                comet_speed=cs, ship_speed=sp, sun_radius=su, board_size=bd,
            )
            if world.is_terminal():
                return []
            macros = generate_candidates_b(world, pid)
            if not macros:
                return []
            bm, probs = MCTSB(net, num_simulations=_EVAL_SIMS, c_puct=C_PUCT).run(
                world, macros, obs_for_deepseek=obs, config=config
            )
            if bm is None or (probs is not None and probs.max() < 0.1):
                bm = roi_fallback_macro(macros)
            return macro_to_env(bm) if bm is not None else []
        return _agent

    roster = [_make_net_agent(i) if i in hero_slots else opp_fn for i in range(num_agents)]
    env.run(roster)
    rr = env.steps[-1][hero].reward if hero < len(env.steps[-1]) else 0
    return float(rr) > 0


def _append_csv_log(log_dir, iteration, metrics, eval_results):
    csv_path = os.path.join(log_dir, "train_log.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_header:
            f.write(
                "iteration,loss,value_loss,policy_loss,"
                "wins_random_2p,wins_deepseek_2p,"
                "wins_random_4p,wins_deepseek_4p,wins_starter_2p\n"
            )
        loss = metrics["loss"] if metrics else ""
        vl = metrics["value_loss"] if metrics else ""
        pl = metrics["policy_loss"] if metrics else ""
        f.write(
            f"{iteration},{loss},{vl},{pl},"
            f"{eval_results.get('wins_random_2p', '')},"
            f"{eval_results.get('wins_deepseek_2p', '')},"
            f"{eval_results.get('wins_random_4p', '')},"
            f"{eval_results.get('wins_deepseek_4p', '')},"
            f"{eval_results.get('wins_starter_2p', '')}\n"
        )


def _trainer_eval(net, pool):
    """并行评估：random / deepseek（2P+4P）+ starter（2P）。"""
    cpu_sd = {k: v.cpu() for k, v in net.state_dict().items()}
    rng = random.Random(42)
    ep = GAME_EVAL_EPISODES
    ep4 = max(4, ep // 2)

    tasks = []
    task_labels = []
    for opp in ["random", "deepseek"]:
        for n_agents, label in [(2, "2p"), (4, "4p")]:
            n_ep = ep if n_agents == 2 else ep4
            for _ in range(n_ep):
                hero = rng.randrange(n_agents)
                seed = rng.randint(0, 2 ** 30 - 1)
                tasks.append((cpu_sd, opp, hero, n_agents, seed))
                task_labels.append(f"{opp}_{label}")

    for _ in range(ep):
        hero = rng.randrange(2)
        seed = rng.randint(0, 2 ** 30 - 1)
        tasks.append((cpu_sd, "starter", hero, 2, seed))
        task_labels.append("starter_2p")

    wins_map = {lbl: 0 for lbl in set(task_labels)}
    total_map = {lbl: 0 for lbl in set(task_labels)}
    try:
        outcomes = pool.map(_eval_game_worker, tasks)
    except Exception as ex:
        print(f"  [eval] 并行评估失败: {ex}")
        return {}

    for label, won in zip(task_labels, outcomes):
        total_map[label] += 1
        if won:
            wins_map[label] += 1

    results = {}
    for opp in ["random", "deepseek", "starter"]:
        labels = [(2, "2p"), (4, "4p")] if opp != "starter" else [(2, "2p")]
        for _, label in labels:
            key = f"{opp}_{label}"
            w, t = wins_map.get(key, 0), total_map.get(key, 0)
            if t > 0:
                print(f"  [eval] vs {opp} ({t} ep {label}): {w}/{t}")
                results[f"wins_{key}"] = w
    return results


def _run_one_deepseek_demo_game(deepseek_fn, n_agents=PRETRAIN_NUM_AGENTS):
    """跑一局 deepseek 自博弈，返回 IL 样本列表。"""
    env = make("orbit_wars", debug=False, configuration=orbit_wars_config())
    game_history = []

    def make_demo_agent(player_id):
        def agent(obs, config):
            world = build_world_from_obs(
                obs, player_id, n_agents,
                episode_seed=(getattr(env, "info", None) or {}).get("seed") or 0,
            )
            if world.is_terminal():
                return []
            macros = generate_candidates_b(world, player_id)
            if not macros:
                return []
            ds_action = deepseek_fn(obs, config) or []
            macros, matched, idx = inject_teacher_macro(macros, ds_action, world, player_id)
            if matched is None:
                return ds_action
            probs = np.zeros(MAX_MACRO_SLOTS, dtype=np.float32)
            if 0 <= idx < MAX_MACRO_SLOTS:
                probs[idx] = 1.0
            state_np = encode_state(world, perspective_player=player_id, device="cpu").numpy()
            game_history.append((player_id, state_np, probs))
            return macro_to_env(matched)
        return agent

    env.run([make_demo_agent(i) for i in range(n_agents)])
    final_step = env.steps[-1]
    scores = {i: float(s.reward if s.reward is not None else 0.0) for i, s in enumerate(final_step)}
    participant_ids = list(range(n_agents))
    final_obs = final_step[0].observation if hasattr(final_step[0], "observation") else {}
    final_ships = {}
    for p in (_read(final_obs, "planets", []) or []):
        if p[1] != -1:
            final_ships[p[1]] = final_ships.get(p[1], 0) + p[5]
    for f in (_read(final_obs, "fleets", []) or []):
        if f[1] != -1:
            final_ships[f[1]] = final_ships.get(f[1], 0) + f[6]
    samples = []
    for pid, state_np, macro_pi in game_history:
        ev = env_terminal_value(scores, pid, participant_ids=participant_ids)
        my_s = float(final_ships.get(pid, 0.0))
        best_other = max(
            (float(final_ships.get(p, 0.0)) for p in participant_ids if p != pid), default=0.0
        )
        diff = (my_s - best_other) / max(1.0, float(VALUE_TARGET_SCALE))
        target_v = 0.5 * ev + 0.5 * float(np.tanh(diff))
        samples.append((state_np.astype(np.float16), float(target_v), macro_pi.astype(np.float16)))
    return samples


def generate_deepseek_demos(n_games, start_idx=0):
    deepseek_fn = _load_deepseek_agent()
    all_samples = []
    for game_idx in range(start_idx, n_games):
        all_samples.extend(_run_one_deepseek_demo_game(deepseek_fn))
        if (game_idx + 1) % 10 == 0:
            print(f"  [pretrain] {game_idx + 1}/{n_games}, samples={len(all_samples)}")
    return all_samples


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, value, macro_pi):
        self.buffer.append((state.astype(np.float16), value, macro_pi.astype(np.float16)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, values, macro_pis = zip(*batch)
        states = torch.from_numpy(np.array(states)).float().to(DEVICE)
        values = torch.tensor(values, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        macro_pis = torch.from_numpy(np.array(macro_pis)).float().to(DEVICE)
        return states, values, macro_pis

    def __len__(self):
        return len(self.buffer)

    def to_list(self):
        return list(self.buffer)

    def load_list(self, items):
        self.buffer.clear()
        for item in items:
            self.buffer.append(item)


def _ckpt_path(name):
    return os.path.join(CHECKPOINT_DIR, f"{name}{CHECKPOINT_SUFFIX}.pt")


def _pretrain_reservoir_path():
    return _ckpt_path(PRETRAIN_RESERVOIR_CHECKPOINT)


def _normalize_sample(sample):
    """统一 fp16 存储，降低 reservoir / 存盘峰值内存（旧版 float32 单条约 680KB）。"""
    state, value, macro_pi = sample
    state = np.asarray(state, dtype=np.float16)
    macro_pi = np.asarray(macro_pi, dtype=np.float16)
    return state, float(value), macro_pi


def _save_pretrain_reservoir(game_idx, total_seen, reservoir):
    for i in range(len(reservoir)):
        reservoir[i] = _normalize_sample(reservoir[i])
    gc.collect()
    path = _pretrain_reservoir_path()
    tmp_path = path + ".tmp"
    torch.save(
        {"game_idx": game_idx, "total_seen": total_seen, "reservoir": reservoir},
        tmp_path,
    )
    os.replace(tmp_path, path)


def _load_pretrain_reservoir():
    path = _pretrain_reservoir_path()
    if not os.path.exists(path):
        return 0, 0, []
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, RuntimeError, OSError, pickle.UnpicklingError) as ex:
        print(f"[pretrain] reservoir 检查点损坏（可能上次 OOM 写盘中断）: {ex}")
        print(f"[pretrain] 删除 {path}，将重新迁移/采 demo")
        try:
            os.remove(path)
        except OSError:
            pass
        return 0, 0, []
    game_idx = int(data.get("game_idx", 0))
    total_seen = int(data.get("total_seen", 0))
    reservoir = data.get("reservoir", [])
    print(f"[pretrain] 加载 reservoir 检查点 game={game_idx}, seen={total_seen}, size={len(reservoir)}")
    return game_idx, total_seen, reservoir


def _cleanup_pretrain_reservoir():
    path = _pretrain_reservoir_path()
    if os.path.exists(path):
        os.remove(path)


def _pretrain_samples_dir():
    return os.path.join(CHECKPOINT_DIR, "pretrain_samples")


def _pretrain_completed_games():
    pattern = os.path.join(_pretrain_samples_dir(), "game_*.pt")
    return len(glob.glob(pattern))


def _migrate_legacy_pretrain_game_files(reservoir=None, total_seen=0, start_file_idx=0):
    """流式读旧版 pretrain_samples/game_*.pt → reservoir；每 50 局存盘，支持断点续迁移。"""
    files = sorted(glob.glob(os.path.join(_pretrain_samples_dir(), "game_*.pt")))
    if not files:
        return start_file_idx, total_seen, reservoir or []
    if reservoir is None:
        reservoir = []
    if start_file_idx >= len(files):
        _cleanup_pretrain_samples()
        return len(files), total_seen, reservoir
    if start_file_idx == 0:
        print(f"[pretrain] 迁移旧版逐局文件 {len(files)} 个（仅此一次，reservoir 用 fp16）...")
    else:
        print(f"[pretrain] 续迁移 legacy {start_file_idx}/{len(files)}（已有 reservoir={len(reservoir)}）...")
    reservoir, total_seen = _stream_pretrain_reservoir_from_files(
        files, reservoir=reservoir, total_seen=total_seen, start_file_idx=start_file_idx,
    )
    game_idx = len(files)
    _save_pretrain_reservoir(game_idx, total_seen, reservoir)
    _cleanup_pretrain_samples()
    print(f"[pretrain] 旧版逐局目录已删除，reservoir 已写入 {_pretrain_reservoir_path()}")
    return game_idx, total_seen, reservoir


def _reservoir_add(reservoir, total_seen, sample):
    """Reservoir sampling：从流式样本中均匀抽取固定容量，峰值内存 O(REPLAY_BUFFER_SIZE)。"""
    sample = _normalize_sample(sample)
    if len(reservoir) < REPLAY_BUFFER_SIZE:
        reservoir.append(sample)
    else:
        j = random.randint(0, total_seen - 1)
        if j < REPLAY_BUFFER_SIZE:
            reservoir[j] = sample
    return total_seen + 1


def _stream_pretrain_reservoir_from_files(
    files, reservoir=None, total_seen=0, start_file_idx=0, checkpoint_every=50,
):
    """逐局文件扫描，不把全部样本一次性载入内存；定期存 reservoir 检查点。"""
    if reservoir is None:
        reservoir = []
    for i in range(start_file_idx, len(files)):
        path = files[i]
        game_samples = torch.load(path, map_location="cpu", weights_only=False)
        for sample in game_samples:
            total_seen = _reservoir_add(reservoir, total_seen, sample)
        del game_samples
        processed = i + 1
        if processed % 10 == 0 or processed == len(files):
            print(f"  [pretrain] reservoir 扫描 {processed}/{len(files)} 局，累计 {total_seen} 样本")
        if processed % checkpoint_every == 0:
            _save_pretrain_reservoir(processed, total_seen, reservoir)
            gc.collect()
    return reservoir, total_seen


def _flush_reservoir_to_buffer(reservoir, replay_buffer):
    print(f"[pretrain] reservoir {len(reservoir)} 条样本入 replay buffer")
    for sample in reservoir:
        replay_buffer.push(*sample)


def _cleanup_pretrain_samples():
    d = _pretrain_samples_dir()
    if os.path.isdir(d):
        shutil.rmtree(d)


def _remove_legacy_pretrain_progress():
    legacy = _ckpt_path(PRETRAIN_PROGRESS_CHECKPOINT)
    if os.path.exists(legacy):
        os.remove(legacy)


def _has_compatible_v4_ckpt():
    for n in (LATEST_CHECKPOINT, INTERRUPT_CHECKPOINT, PRETRAINED_CHECKPOINT):
        path = _ckpt_path(n)
        if not os.path.exists(path):
            continue
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if ck.get("version") == CHECKPOINT_VERSION:
                return True
        except Exception:
            pass
    return False


class TrainerB:
    def __init__(self):
        self.net = PolicyValueNetworkB(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS, MAX_MACRO_SLOTS).to(DEVICE)
        if _has_compatible_v4_ckpt():
            print("[load_v3_backbone] 已有兼容 v4 checkpoint，跳过 v3 迁移。")
        else:
            load_v3_backbone_into(self.net, device=DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=MAX_ITERATIONS, eta_min=1e-5
        )
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.iteration = 0
        self.scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
        self._pool = None
        self._stopping = False

    def _handle_interrupt(self, signum, frame):
        if self._stopping:
            raise KeyboardInterrupt()
        self._stopping = True
        print("\n收到中断信号，正在停止 worker...")
        self._shutdown_pool()
        raise KeyboardInterrupt()

    def _ensure_pool(self):
        if self._pool is None:
            n = min(NUM_WORKERS, NUM_PARALLEL_GAMES, mp.cpu_count())
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(processes=n, initializer=_worker_init)
            print(f"[multiprocessing] {n} workers")
        return self._pool

    def _shutdown_pool(self):
        if self._pool is not None:
            self._stopping = True
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def _load_random_opponent_net(self):
        import glob
        paths = glob.glob(os.path.join(CHECKPOINT_DIR, f"iter_*{CHECKPOINT_SUFFIX}.pt"))
        if not paths:
            return None
        path = random.choice(paths)
        opp = PolicyValueNetworkB(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS, MAX_MACRO_SLOTS).to(DEVICE)
        try:
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            opp.load_state_dict(ck["model_state_dict"])
            opp.eval()
            return opp
        except Exception:
            return None

    def _current_n_simulations(self):
        return MCTS_SIM_FULL if self.iteration >= MCTS_SIM_BOOST_ITER else TRAIN_MCTS_SIMULATIONS

    def _pick_opponent(self):
        if self.iteration < DEEPSEEK_START_ITER:
            return None
        if self.iteration < POOL_START_ITER:
            return "deepseek" if random.random() < DEEPSEEK_OPP_PROB else None
        r = random.random()
        if r < 0.30:
            return None
        if r < 0.65:
            return "deepseek"
        return self._load_random_opponent_net() or None

    def save_checkpoint(self, name, save_replay=False):
        path = _ckpt_path(name)
        torch.save({
            "version": CHECKPOINT_VERSION,
            "iteration": self.iteration,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        if save_replay:
            self._save_replay()

    def _save_replay(self):
        path = _ckpt_path(REPLAY_CHECKPOINT)
        torch.save({"buffer": self.replay_buffer.to_list()}, path)

    def _load_replay(self):
        path = _ckpt_path(REPLAY_CHECKPOINT)
        if not os.path.exists(path):
            return
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            buffer = data.get("buffer", [])
            if buffer:
                pi_len = len(buffer[0][2]) if hasattr(buffer[0][2], "__len__") else 0
                if pi_len != MAX_MACRO_SLOTS:
                    print(f"[resume] replay macro_pi={pi_len} != {MAX_MACRO_SLOTS}，跳过旧 replay")
                    return
            self.replay_buffer.load_list(buffer)
            print(f"[resume] replay buffer: {len(self.replay_buffer)} samples")
        except Exception as ex:
            print(f"[resume] replay 加载失败: {ex}")

    def load_checkpoint(self, name):
        path = _ckpt_path(name)
        if not os.path.exists(path):
            return False
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        if ck.get("version") != CHECKPOINT_VERSION:
            print(f"[resume] {name} 版本 {ck.get('version')} != {CHECKPOINT_VERSION}，跳过")
            return False
        try:
            self.net.load_state_dict(ck["model_state_dict"])
            self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        except RuntimeError as err:
            print(f"检查点不兼容: {err}")
            return False
        self.iteration = int(ck.get("iteration", 0))
        if "scheduler_state_dict" in ck:
            self.scheduler.load_state_dict(ck["scheduler_state_dict"])
        return True

    def _checkpoint_iteration(self, name):
        path = _ckpt_path(name)
        if not os.path.exists(path):
            return -1
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            return int(ck.get("iteration", 0))
        except Exception:
            return -1

    def train_step(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return None
        states, values, macro_pis = self.replay_buffer.sample(BATCH_SIZE)
        self.net.train()
        self.optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            logits, pred_v = self.net(states)
            value_loss = torch.nn.functional.mse_loss(pred_v, values)
            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -(macro_pis * log_probs).sum(dim=1).mean()
            loss = VALUE_LOSS_WEIGHT * value_loss + POLICY_LOSS_WEIGHT * policy_loss
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.net.eval()
        return {
            "loss": float(loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
        }

    def pretrain(self):
        path = _ckpt_path(PRETRAINED_CHECKPOINT)
        if FORCE_REPRETRAIN and os.path.exists(path):
            print("[pretrain] FORCE_REPRETRAIN=True，忽略旧 pretrained，重新 IL。")
            for name in (PRETRAINED_CHECKPOINT, PRETRAIN_RESERVOIR_CHECKPOINT):
                p = _ckpt_path(name)
                if os.path.exists(p):
                    os.remove(p)
                    print(f"[pretrain] 已删除 {p}")
        elif os.path.exists(path):
            print("[pretrain] 已有 pretrained_b.pt，加载后跳过。")
            if self.load_checkpoint(PRETRAINED_CHECKPOINT):
                self.iteration = 0
            return

        _remove_legacy_pretrain_progress()
        game_idx, total_seen, reservoir = _load_pretrain_reservoir()
        legacy_count = _pretrain_completed_games()

        if legacy_count > 0:
            if game_idx < legacy_count:
                game_idx, total_seen, reservoir = _migrate_legacy_pretrain_game_files(
                    reservoir, total_seen, start_file_idx=game_idx,
                )
            else:
                print("[pretrain] legacy 逐局文件已处理，清理遗留目录...")
                _cleanup_pretrain_samples()

        if game_idx < PRETRAIN_GAMES:
            print(
                f"[pretrain] 采 demo {game_idx}/{PRETRAIN_GAMES}（2P，"
                f"每 {PRETRAIN_RESERVOIR_SAVE_EVERY} 局存 reservoir，不写逐局大文件）..."
            )
            deepseek_fn = _load_deepseek_agent()
            while game_idx < PRETRAIN_GAMES:
                samples = _run_one_deepseek_demo_game(deepseek_fn)
                for sample in samples:
                    total_seen = _reservoir_add(reservoir, total_seen, sample)
                del samples
                game_idx += 1
                if game_idx % 10 == 0:
                    print(f"  [pretrain] {game_idx}/{PRETRAIN_GAMES}, reservoir={len(reservoir)}")
                if game_idx % PRETRAIN_RESERVOIR_SAVE_EVERY == 0:
                    _save_pretrain_reservoir(game_idx, total_seen, reservoir)
        else:
            print(f"[pretrain] demo 已达 {game_idx} 局（目标 {PRETRAIN_GAMES}），跳过采 demo")

        if game_idx > 0 and game_idx % PRETRAIN_RESERVOIR_SAVE_EVERY != 0:
            _save_pretrain_reservoir(game_idx, total_seen, reservoir)

        print(f"[pretrain] 共见 {total_seen} 样本，reservoir 容量 {len(reservoir)}")
        _flush_reservoir_to_buffer(reservoir, self.replay_buffer)
        del reservoir

        print(f"[pretrain] {PRETRAIN_TRAIN_STEPS} 步梯度...")
        for step in range(PRETRAIN_TRAIN_STEPS):
            m = self.train_step()
            if m and (step + 1) % 100 == 0:
                print(f"  [pretrain] {step+1}: loss={m['loss']:.4f}")
        self.save_checkpoint(PRETRAINED_CHECKPOINT, save_replay=True)
        self.iteration = 0
        _cleanup_pretrain_reservoir()
        _cleanup_pretrain_samples()

    def run(self):
        loaded = False
        int_iter = self._checkpoint_iteration(INTERRUPT_CHECKPOINT)
        lat_iter = self._checkpoint_iteration(LATEST_CHECKPOINT)
        if int_iter >= lat_iter:
            loaded = self.load_checkpoint(INTERRUPT_CHECKPOINT)
            if not loaded:
                loaded = self.load_checkpoint(LATEST_CHECKPOINT)
        else:
            loaded = self.load_checkpoint(LATEST_CHECKPOINT)
        if loaded:
            self._load_replay()
        elif FORCE_REPRETRAIN:
            print("[run] 无兼容 checkpoint 或 FORCE_REPRETRAIN，从 v3 backbone + 新 IL 开始。")
            if not _has_compatible_v4_ckpt():
                load_v3_backbone_into(self.net, device=DEVICE)
            self.iteration = 0
        self.net.eval()

        if self.iteration == 0 and PRETRAIN_GAMES > 0:
            self.pretrain()

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        print(f"开始训练 v4 路线 B，iter={self.iteration}")
        try:
            while self.iteration < MAX_ITERATIONS:
                print(f"迭代 {self.iteration + 1}: 自博弈...")
                n_sim = self._current_n_simulations()
                cpu_sd = {k: v.cpu() for k, v in self.net.state_dict().items()}
                tasks = []
                for _ in range(NUM_PARALLEL_GAMES):
                    opp = self._pick_opponent()
                    opp_info = {k: v.cpu() for k, v in opp.state_dict().items()} if isinstance(opp, torch.nn.Module) else opp
                    tasks.append((cpu_sd, opp_info, n_sim))
                pool = self._ensure_pool()
                for samples in pool.imap(_run_game_worker, tasks):
                    for item in samples:
                        self.replay_buffer.push(*item)
                self._stopping = False

                metrics = None
                for _ in range(TRAIN_EPOCHS_PER_ITER):
                    metrics = self.train_step()

                self.iteration += 1
                self.scheduler.step()

                if self.iteration % SAVE_EVERY_ITERS == 0:
                    self.save_checkpoint(LATEST_CHECKPOINT, save_replay=True)
                    self.save_checkpoint(f"iter_{self.iteration}")  # 仅模型，不写 replay

                if metrics:
                    print(f"buf={len(self.replay_buffer)} loss={metrics['loss']:.4f} "
                          f"v={metrics['value_loss']:.4f} p={metrics['policy_loss']:.4f}")
                else:
                    print(f"buf={len(self.replay_buffer)}（batch 过小未训练 step）")

                eval_results = {}
                if EVAL_EVERY_ITERS > 0 and self.iteration % EVAL_EVERY_ITERS == 0:
                    eval_results = _trainer_eval(self.net, pool)

                _append_csv_log(LOG_DIR, self.iteration, metrics, eval_results)
        except KeyboardInterrupt:
            print("\n保存 interrupt_b + replay_b...")
            self.save_checkpoint(INTERRUPT_CHECKPOINT, save_replay=True)
        finally:
            self._shutdown_pool()


if __name__ == "__main__":
    mp.freeze_support()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    TrainerB().run()
