import torch

from config import CHANNELS, BOARD_SIZE, DEVICE
from network import PolicyValueNetwork
from mcts import MCTS
from physics import Planet, Fleet, WorldState
from value_util import env_terminal_value


def _build_simple_world():
    planets = [
        Planet(0, 0, 20.0, 20.0, 3.0, 40, 2),
        Planet(1, 1, 80.0, 80.0, 3.0, 35, 2),
        Planet(2, -1, 40.0, 40.0, 2.0, 10, 1),
    ]
    fleets = [Fleet(0, 0, 22.0, 20.0, 0.0, 0, 8)]
    initial_planets = [
        (0, 0, 20.0, 20.0, 3.0, 40, 2),
        (1, 1, 80.0, 80.0, 3.0, 35, 2),
        (2, -1, 40.0, 40.0, 2.0, 10, 1),
    ]
    return WorldState(
        planets,
        fleets,
        initial_planets,
        step=0,
        base_omega=0.0,
        comets=[],
        comet_ids=set(),
        player_ids=[0, 1],
        my_id=0,
        num_training_agents=2,
    )


def _build_comet_world():
    planets = [
        Planet(10, 0, 10.0, 10.0, 3.0, 20, 1),
        Planet(11, 1, 90.0, 90.0, 3.0, 20, 1),
        Planet(12, -1, 15.0, 15.0, 2.0, 5, 1),
    ]
    initial_planets = [
        (10, 0, 10.0, 10.0, 3.0, 20, 1),
        (11, 1, 90.0, 90.0, 3.0, 20, 1),
        (12, -1, 15.0, 15.0, 2.0, 5, 1),
    ]
    comets = [{
        "planet_ids": [12],
        "paths": [[(15.0, 15.0), (16.0, 16.0), (17.0, 17.0)]],
        "path_index": 0,
    }]
    return WorldState(
        planets,
        fleets=[],
        initial_planets=initial_planets,
        step=0,
        base_omega=0.0,
        comets=comets,
        comet_ids={12},
        player_ids=[0, 1],
        my_id=0,
        num_training_agents=2,
    )


def run_smoke_checks():
    net = PolicyValueNetwork(CHANNELS, BOARD_SIZE).to(DEVICE)
    net.eval()
    mcts = MCTS(net, num_simulations=5)

    # 1) 终局时 MCTS 应返回空（不再搜索）
    w = _build_simple_world()
    for p in w.planet_list:
        p.owner = 0
    for f in w.fleets:
        f.owner = 0
    best, probs = mcts.run(w, w.get_legal_actions(0))
    assert best is None
    assert probs.size == 0

    # 2) MCTS 在常规局面可返回合法动作
    w2 = _build_simple_world()
    legal2 = w2.get_legal_actions(0)
    best2, probs2 = mcts.run(w2, legal2)
    assert best2 in legal2
    assert probs2.shape[0] == len(legal2)
    assert abs(float(probs2.sum()) - 1.0) < 1e-5

    # 3) 终局检测：单一势力占据所有行星和舰队
    w3 = _build_simple_world()
    for p in w3.planet_list:
        p.owner = 0
    for f in w3.fleets:
        f.owner = 0
    assert w3.is_terminal()

    # 4) 彗星 path_index 推进
    wc = _build_comet_world()
    before = wc.comets[0]["path_index"]
    wc.step({0: [], 1: []})
    after = wc.comets[0]["path_index"]
    assert after == before + 1

    assert env_terminal_value({0: 10.0, 1: 5.0, 2: 3.0}, 0, [0, 1, 2]) > 0
    assert env_terminal_value({0: 2.0, 1: 5.0, 2: 5.0}, 0, [0, 1, 2]) < 0

    print("smoke checks passed")


if __name__ == "__main__":
    torch.manual_seed(0)
    run_smoke_checks()
