"""轻量评估：带 MCTS 的 net_agent 对阵内置 random_agent。"""

import importlib
import random

from kaggle_environments import make

from config import C_PUCT, MCTS_SIMULATIONS, MAX_GAME_STEPS
from mcts import MCTS


def macro_to_env_moves(macro):
    return [[atom[0], atom[3], atom[2]] for atom in macro]


def evaluate_net_vs_random(net, episodes=12, num_agents=2):
    import train as tm

    net.eval()

    orb_mod = importlib.import_module(
        "kaggle_environments.envs.orbit_wars.orbit_wars"
    )
    random_agent = orb_mod.random_agent
    rng_master = random.Random(42)

    def make_agent(player_id, env):
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

            mc = MCTS(net, num_simulations=min(48, MCTS_SIMULATIONS), c_puct=C_PUCT)

            bm, _ = mc.run(world, macs)

            if bm is None:
                return []

            return macro_to_env_moves(bm)

        return agent

    wins = 0
    for ti in range(episodes):
        hero = rng_master.randrange(num_agents)

        env = make(
            "orbit_wars",

            debug=False,

            configuration={"episodeSteps": MAX_GAME_STEPS, "seed": rng_master.randint(0, 2**30 - 1)},
        )

        roster = []
        for i in range(num_agents):

            roster.append(make_agent(i, env) if i == hero else random_agent)

        env.run(roster)



        rew = env.steps[-1]


        rr = rew[hero].reward if hero < len(rew) else 0

        if rr is not None and float(rr) > 0:
            wins += 1



    return wins, episodes



