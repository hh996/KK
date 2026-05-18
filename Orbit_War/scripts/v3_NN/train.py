import os
import signal
import random
import warnings
import multiprocessing as mp
from collections import deque

import numpy as np
import torch
from kaggle_environments import make

# 压掉 PyTorch 已知误报：scheduler 在 optimizer 之后调用但内部计数判断有误
warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
)

from value_util import env_terminal_value
from config import *
from physics import Planet, Fleet, WorldState, MAX_SPEED, SUN_R, BOARD_SIZE as PHYS_BOARD_SIZE
from features import encode_state
from network import PolicyValueNetwork
from mcts import MCTS, ship_bucket_idx


def _read(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def macro_to_env_moves(macro):
    """宏动作转为环境接受的 [[planet_id, angle, ships], ...]"""
    return [[atom[0], atom[3], atom[2]] for atom in macro]


def build_world_from_obs(
    obs,
    player_id,
    num_training_agents,
    episode_seed=None,
    comet_speed=4.0,
    ship_speed=None,
    sun_radius=None,
    board_size=None,
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
    bs = float(PHYS_BOARD_SIZE if board_size is None else board_size)
    return WorldState(
        planets,
        fleets,
        initial_planets,
        step,
        base_omega,
        comets,
        comet_ids,
        sorted(player_ids_found),
        player_id,
        num_training_agents=num_training_agents,
        episode_seed=episode_seed,
        comet_speed=comet_speed,
        ship_speed=ss,
        sun_radius=sr,
        board_size=bs,
    )


def build_policy_targets(world, legal_macros, probs):
    src_target = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    tgt_target = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    ship_target = np.zeros(SHIP_BUCKET_COUNT, dtype=np.float32)

    if not legal_macros:
        src_target.fill(1.0 / (BOARD_SIZE * BOARD_SIZE))
        tgt_target.fill(1.0 / (BOARD_SIZE * BOARD_SIZE))
        ship_target.fill(1.0 / SHIP_BUCKET_COUNT)
        return src_target, tgt_target, ship_target

    probs = np.asarray(probs, dtype=np.float32)
    den = probs.sum()
    if den <= 0:
        probs = np.ones(len(legal_macros), dtype=np.float32) / len(legal_macros)
    else:
        probs = probs / den

    for macro, p in zip(legal_macros, probs):
        w = float(p) / max(1, len(macro))
        for atom in macro:
            src_id, tgt_id, ships, _, _ = atom
            src = world.planets.get(src_id)
            tgt = world.planets.get(tgt_id)
            if src is None or tgt is None:
                continue
            sx = max(0, min(BOARD_SIZE - 1, int(src.x)))
            sy = max(0, min(BOARD_SIZE - 1, int(src.y)))
            tx = max(0, min(BOARD_SIZE - 1, int(tgt.x)))
            ty = max(0, min(BOARD_SIZE - 1, int(tgt.y)))
            src_target[sy, sx] += w
            tgt_target[ty, tx] += w
            bi = ship_bucket_idx(ships)
            ship_target[bi] += w

    ssum = src_target.sum()
    tsum = tgt_target.sum()
    hsum = ship_target.sum()
    if ssum > 0:
        src_target /= ssum
    if tsum > 0:
        tgt_target /= tsum
    if hsum > 0:
        ship_target /= hsum
    return src_target, tgt_target, ship_target


def _load_deepseek_agent():
    """懒加载 v1_deepseek agent 函数，缓存到模块变量避免重复 import。"""
    import importlib.util, os
    global _deepseek_agent_fn
    if _deepseek_agent_fn is not None:
        return _deepseek_agent_fn
    path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "v1_rule", "v1_deepseek", "main.py"
    ))
    spec = importlib.util.spec_from_file_location("v1_deepseek_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _deepseek_agent_fn = mod.agent
    return _deepseek_agent_fn

_deepseek_agent_fn = None   # 模块级缓存


def play_one_game(net, opponent=None, n_simulations=TRAIN_MCTS_SIMULATIONS):
    """自博弈一局，返回训练样本列表。

    opponent:
      None                  → 纯自博弈（所有玩家均用 net+MCTS）
      "deepseek"            → 随机选一个槽位作为 hero，其余用 v1_deepseek
      PolicyValueNetwork    → 其余玩家用该旧 checkpoint net+MCTS
    """
    n_agents = 2 if random.random() < TRAIN_TWO_PLAYER_PROB else 4
    env = make("orbit_wars", debug=False, configuration={"episodeSteps": MAX_GAME_STEPS})
    game_history = []

    # 决定哪些槽位是"hero"（用 net 训练）
    if opponent is None:
        hero_slots = set(range(n_agents))       # 纯自博弈：所有玩家都是 hero
    else:
        # 有对手时：2P 选 1 个 hero，4P 选 2 个 hero（对角槽位，位置对称）
        if n_agents == 2:
            hero_slots = {random.randrange(2)}
        else:
            start = random.randrange(2)         # 0 或 1
            hero_slots = {start, start + 2}     # 对角线两个槽位：(0,2) 或 (1,3)

    def make_net_agent(player_id, train_net, collect_data):
        """collect_data=True 时把 MCTS priors 写入 game_history。"""
        def agent(obs, config):
            info = getattr(env, "info", None) or {}
            episode_seed = info.get("seed") or 0
            cs = float(_read(config, "cometSpeed", 4.0) or 4.0)
            sp = float(_read(config, "shipSpeed", MAX_SPEED) or MAX_SPEED)
            su = float(_read(config, "sunRadius", SUN_R) or SUN_R)
            bd = float(_read(config, "boardSize", PHYS_BOARD_SIZE) or PHYS_BOARD_SIZE)
            world = build_world_from_obs(
                obs, player_id, n_agents,
                episode_seed=episode_seed, comet_speed=cs,
                ship_speed=sp, sun_radius=su, board_size=bd,
            )
            if world.is_terminal():
                return []
            legal_macros = world.get_legal_macro_actions(player_id)
            if not legal_macros:
                return []

            if collect_data:
                state_np = encode_state(world, perspective_player=player_id, device="cpu").numpy()
                mcts_inst = MCTS(train_net, c_puct=C_PUCT, num_simulations=n_simulations)
                best_macro, probs = mcts_inst.run(world, legal_macros)
                if best_macro is None:
                    return []
                src_pi, tgt_pi, ship_pi = build_policy_targets(world, legal_macros, probs)
                game_history.append((player_id, state_np, src_pi, tgt_pi, ship_pi))
                return macro_to_env_moves(best_macro)
            else:
                # 对手网络：同样用 MCTS 但不收集数据
                mcts_inst = MCTS(train_net, c_puct=C_PUCT, num_simulations=n_simulations)
                best_macro, _ = mcts_inst.run(world, legal_macros)
                return macro_to_env_moves(best_macro) if best_macro is not None else []
        return agent

    agents = []
    for i in range(n_agents):
        if i in hero_slots:
            agents.append(make_net_agent(i, net, collect_data=True))
        elif opponent == "deepseek":
            agents.append(_load_deepseek_agent())
        else:
            # opponent 是旧 checkpoint net
            agents.append(make_net_agent(i, opponent, collect_data=False))
    env.run(agents)

    final_step = env.steps[-1]
    scores = {}
    for i, s in enumerate(final_step):
        r = s.reward
        scores[i] = float(r if r is not None else 0.0)
    participant_ids = list(range(n_agents))

    # 从最终 obs 取真实船数，计算与 MCTS._terminal_value 相同的混合目标值
    final_obs = final_step[0].observation if hasattr(final_step[0], "observation") \
        else (final_step[0].get("observation", {}) if isinstance(final_step[0], dict) else {})
    final_ships = {}
    for p in (_read(final_obs, "planets", []) or []):
        if p[1] != -1:
            final_ships[p[1]] = final_ships.get(p[1], 0) + p[5]
    for f in (_read(final_obs, "fleets", []) or []):
        if f[1] != -1:
            final_ships[f[1]] = final_ships.get(f[1], 0) + f[6]

    samples = []
    for pid, state_np, src_pi, tgt_pi, ship_pi in game_history:
        ev = env_terminal_value(scores, pid, participant_ids=participant_ids)
        my_s = float(final_ships.get(pid, 0.0))
        best_other = max(
            (float(final_ships.get(p, 0.0)) for p in participant_ids if p != pid),
            default=0.0,
        )
        diff = (my_s - best_other) / max(1.0, float(VALUE_TARGET_SCALE))
        target_v = 0.5 * ev + 0.5 * float(np.tanh(diff))
        samples.append((state_np, float(target_v), src_pi, tgt_pi, ship_pi))
    print(f"  Game finished ({n_agents}p), {len(samples)} samples.")
    return samples


def _worker_init():
    """限制每个 worker 的 PyTorch 内部线程数；忽略 SIGINT 以便主进程统一处理 Ctrl+C。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)


def _run_game_worker(args):
    """多进程工作函数：在 CPU 上重建网络，跑一局，返回 samples。必须为顶层函数以支持 pickle。"""
    net_sd, opp_info, n_simulations = args
    net = PolicyValueNetwork(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS)
    net.load_state_dict(net_sd)
    net.eval()
    if isinstance(opp_info, dict):
        opp_net = PolicyValueNetwork(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS)
        opp_net.load_state_dict(opp_info)
        opp_net.eval()
        opponent = opp_net
    else:
        opponent = opp_info  # None 或 "deepseek"
    return play_one_game(net, opponent=opponent, n_simulations=n_simulations)


_EVAL_SIMS = min(48, MCTS_SIMULATIONS)


def _eval_game_worker(args):
    """评估 worker：单局 net vs 对手，返回是否获胜(bool)。"""
    import importlib
    import importlib.util as _ilu
    net_sd, opponent_type, hero, num_agents, seed = args

    net = PolicyValueNetwork(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS)
    net.load_state_dict(net_sd)
    net.eval()

    if opponent_type == "random":
        orb = importlib.import_module("kaggle_environments.envs.orbit_wars.orbit_wars")
        opp_fn = orb.random_agent
    else:
        v1_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "v1_rule", "v1_deepseek", "main.py"
        ))
        spec = _ilu.spec_from_file_location("_v1ds_eval", v1_path)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        opp_fn = mod.agent

    env = make("orbit_wars", debug=False,
               configuration={"episodeSteps": MAX_GAME_STEPS, "seed": seed})

    def _net_agent(obs, config):
        info = getattr(env, "info", None) or {}
        ep_seed = info.get("seed")
        cs = float(_read(config, "cometSpeed", 4.0) or 4.0)
        sp = float(_read(config, "shipSpeed", MAX_SPEED) or MAX_SPEED)
        su = float(_read(config, "sunRadius", SUN_R) or SUN_R)
        bd = float(_read(config, "boardSize", PHYS_BOARD_SIZE) or PHYS_BOARD_SIZE)
        world = build_world_from_obs(obs, hero, num_agents, episode_seed=ep_seed,
                                     comet_speed=cs, ship_speed=sp, sun_radius=su, board_size=bd)
        if world.is_terminal():
            return []
        macs = world.get_legal_macro_actions(hero)
        if not macs:
            return []
        bm, _ = MCTS(net, num_simulations=_EVAL_SIMS, c_puct=C_PUCT).run(world, macs)
        return macro_to_env_moves(bm) if bm is not None else []

    roster = [_net_agent if i == hero else opp_fn for i in range(num_agents)]
    env.run(roster)
    rr = env.steps[-1][hero].reward if hero < len(env.steps[-1]) else 0
    return float(rr) > 0


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, value, src_pi, tgt_pi, ship_pi):
        self.buffer.append((state.astype(np.float16), value, src_pi, tgt_pi, ship_pi))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, values, src_pis, tgt_pis, ship_pis = zip(*batch)
        states = torch.from_numpy(np.array(states)).float().to(DEVICE)
        values = torch.tensor(values, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        src_pis = torch.from_numpy(np.array(src_pis)).float().to(DEVICE).view(len(batch), -1)
        tgt_pis = torch.from_numpy(np.array(tgt_pis)).float().to(DEVICE).view(len(batch), -1)
        ship_pis = torch.from_numpy(np.array(ship_pis)).float().to(DEVICE)
        return states, values, src_pis, tgt_pis, ship_pis

    def __len__(self):
        return len(self.buffer)


def _append_csv_log(log_dir, iteration, metrics, eval_results):
    csv_path = os.path.join(log_dir, "train_log.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_header:
            f.write("iteration,loss,value_loss,policy_loss,"
                    "wins_random_2p,wins_deepseek_2p,"
                    "wins_random_4p,wins_deepseek_4p\n")
        loss = metrics["loss"] if metrics else ""
        vl   = metrics["value_loss"] if metrics else ""
        pl   = metrics["policy_loss"] if metrics else ""
        f.write(f"{iteration},{loss},{vl},{pl},"
                f"{eval_results.get('wins_random_2p','')},"
                f"{eval_results.get('wins_deepseek_2p','')},"
                f"{eval_results.get('wins_random_4p','')},"
                f"{eval_results.get('wins_deepseek_4p','')}\n")


def _trainer_eval(net, pool):
    """并行评估：所有对局同时提交给进程池，比串行快 N 倍。"""
    cpu_sd = {k: v.cpu() for k, v in net.state_dict().items()}
    rng = random.Random(42)
    ep  = min(8, GAME_EVAL_EPISODES)
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

    wins_map  = {lbl: 0 for lbl in set(task_labels)}
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
    for opp in ["random", "deepseek"]:
        for _, label in [(2, "2p"), (4, "4p")]:
            key = f"{opp}_{label}"
            w, t = wins_map.get(key, 0), total_map.get(key, 0)
            print(f"  [eval] vs {opp} ({t} ep {label}): {w}/{t}")
            results[f"wins_{key}"] = w
    return results


class Trainer:
    def __init__(self):
        self.net = PolicyValueNetwork(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS).to(DEVICE)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=MAX_ITERATIONS, eta_min=1e-5
        )
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.iteration = 0
        self.scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
        self._pool = None
        self._stopping = False

    def _handle_interrupt(self, signum, frame):
        """Ctrl+C / kill：立刻 terminate 进程池，避免 pool.map 一直等到 worker 跑完。"""
        if self._stopping:
            raise KeyboardInterrupt()
        self._stopping = True
        print("\n收到中断信号，正在停止 worker 进程...")
        self._shutdown_pool()
        raise KeyboardInterrupt()

    def _ensure_pool(self):
        if self._pool is None:
            n = min(NUM_WORKERS, NUM_PARALLEL_GAMES, mp.cpu_count())
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(processes=n, initializer=_worker_init)
            print(f"[multiprocessing] 进程池启动：{n} workers")
        return self._pool

    def _shutdown_pool(self):
        if self._pool is not None:
            self._stopping = True
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def _load_random_opponent_net(self):
        """从已保存的 iter_*.pt 里随机加载一个旧版本作为对手网络。"""
        import glob
        paths = glob.glob(os.path.join(CHECKPOINT_DIR, "iter_*.pt"))
        if not paths:
            return None
        path = random.choice(paths)
        opp_net = PolicyValueNetwork(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS).to(DEVICE)
        try:
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            opp_net.load_state_dict(ck["model_state_dict"])
            opp_net.eval()
            return opp_net
        except Exception:
            return None

    def _current_n_simulations(self):
        if self.iteration >= MCTS_SIM_BOOST_ITER:
            return MCTS_SIM_FULL
        return TRAIN_MCTS_SIMULATIONS

    def _pick_opponent(self):
        """根据当前迭代数决定训练对手：None / 'deepseek' / 旧 net。"""
        if self.iteration < DEEPSEEK_START_ITER:
            return None
        if self.iteration < POOL_START_ITER:
            return "deepseek" if random.random() < DEEPSEEK_OPP_PROB else None
        # 第三阶段：30% 自博弈 / 35% deepseek / 35% 旧 checkpoint
        r = random.random()
        if r < 0.30:
            return None
        if r < 0.65:
            return "deepseek"
        return self._load_random_opponent_net() or None   # 加载失败退回自博弈

    def save_checkpoint(self, name):
        path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
        torch.save(
            {
                "iteration": self.iteration,
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, name):
        path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
        if not os.path.exists(path):
            return False
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        try:
            self.net.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except RuntimeError as err:
            print(
                f"检查点与当前 PolicyValueNetwork 结构不兼容，已跳过: {path}\n"
                f"  原因多半是旧版 CHANNELS/Ship policy 与新代码不一致。\n"
                f"  将从头训练；若需清空可删除 {CHECKPOINT_DIR} 下过期的 .pt 文件。\n"
                f"  详情: {err}"
            )
            return False
        self.iteration = checkpoint["iteration"]
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # replay buffer 不再持久化，恢复后用自博弈重新填充
        return True

    def train_step(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return None
        states, values, src_pis, tgt_pis, ship_pis = self.replay_buffer.sample(BATCH_SIZE)
        self.net.train()
        self.optimizer.zero_grad()

        # 自动混合精度上下文
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            src_logits, tgt_logits, ship_logits, pred_v = self.net(states)
            value_loss = torch.nn.functional.mse_loss(pred_v, values)
            src_log_probs = torch.log_softmax(src_logits, dim=1)
            tgt_log_probs = torch.log_softmax(tgt_logits, dim=1)
            ship_log_probs = torch.log_softmax(ship_logits, dim=1)
            src_policy_loss = -(src_pis * src_log_probs).sum(dim=1).mean()
            tgt_policy_loss = -(tgt_pis * tgt_log_probs).sum(dim=1).mean()
            ship_policy_loss = -(ship_pis * ship_log_probs).sum(dim=1).mean()
            policy_loss = src_policy_loss + tgt_policy_loss + ship_policy_loss
            loss = VALUE_LOSS_WEIGHT * value_loss + POLICY_LOSS_WEIGHT * policy_loss

        # 缩放损失并反向传播
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

    def _checkpoint_iteration(self, name):
        """读取 checkpoint 里的 iteration 字段，文件不存在返回 -1。"""
        path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
        if not os.path.exists(path):
            return -1
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            return int(ck.get("iteration", 0))
        except Exception:
            return -1

    def run(self):
        int_iter = self._checkpoint_iteration(INTERRUPT_CHECKPOINT)
        lat_iter = self._checkpoint_iteration("latest")
        if int_iter >= lat_iter:
            loaded = self.load_checkpoint(INTERRUPT_CHECKPOINT)
            if not loaded:
                self.load_checkpoint("latest")
        else:
            print(f"[resume] latest({lat_iter}) > interrupt({int_iter})，加载 latest")
            self.load_checkpoint("latest")
        self.net.eval()

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        print(f"开始训练，当前迭代：{self.iteration}")
        try:
            while self.iteration < MAX_ITERATIONS:
                print(f"迭代 {self.iteration+1}: 生成对局数据...")
                n_sim = self._current_n_simulations()
                cpu_sd = {k: v.cpu() for k, v in self.net.state_dict().items()}
                tasks = []
                for _ in range(NUM_PARALLEL_GAMES):
                    opponent = self._pick_opponent()
                    if isinstance(opponent, torch.nn.Module):
                        opp_info = {k: v.cpu() for k, v in opponent.state_dict().items()}
                    else:
                        opp_info = opponent  # None 或 "deepseek"
                    tasks.append((cpu_sd, opp_info, n_sim))
                pool = self._ensure_pool()
                try:
                    for samples in pool.imap(_run_game_worker, tasks):
                        for item in samples:
                            self.replay_buffer.push(*item)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    if self._stopping:
                        raise KeyboardInterrupt() from exc
                    raise
                self._stopping = False

                print(f"训练 {TRAIN_EPOCHS_PER_ITER} 个 epoch...")
                metrics = None
                for _ in range(TRAIN_EPOCHS_PER_ITER):
                    metrics = self.train_step()

                self.iteration += 1
                self.scheduler.step()

                if self.iteration % SAVE_EVERY_ITERS == 0:
                    self.save_checkpoint(f"iter_{self.iteration}")
                    self.save_checkpoint("latest")
                    print(f"检查点已保存 (iter {self.iteration})")

                if metrics is not None:
                    print(
                        f"当前缓冲区大小：{len(self.replay_buffer)} | loss={metrics['loss']:.4f} "
                        f"value={metrics['value_loss']:.4f} policy={metrics['policy_loss']:.4f}"
                    )
                else:
                    print(f"当前缓冲区大小：{len(self.replay_buffer)}（batch 过小未训练 step）")

                eval_results = {}
                if EVAL_EVERY_ITERS > 0 and self.iteration % EVAL_EVERY_ITERS == 0:
                    eval_results = _trainer_eval(self.net, pool)

                _append_csv_log(LOG_DIR, self.iteration, metrics, eval_results)

            print("训练达到最大迭代次数。")

        except KeyboardInterrupt:
            print("\n检测到中断，保存检查点...")
            self.save_checkpoint(INTERRUPT_CHECKPOINT)
        finally:
            self._shutdown_pool()


if __name__ == "__main__":
    mp.freeze_support()   # Windows spawn 必须
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    trainer = Trainer()
    trainer.run()
