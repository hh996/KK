"""路线 B MCTS：macro-index prior + 混合对手 rollout。"""
import math
import random
import numpy as np
import torch

from physics import WorldState
from features import encode_state
from value_util import env_terminal_value
from world_enrich import enrich_world
from candidates_b import generate_candidates_b, match_env_action_to_macro, macro_key
from atoms_v1 import macro_to_env
from config import (
    C_PUCT,
    MCTS_SIMULATIONS,
    DIRICHLET_ALPHA,
    DIRICHLET_EPSILON,
    VALUE_TARGET_SCALE,
    OPPONENT_ROLLOUT_DEEPSEEK_PROB,
    MAX_MACRO_SLOTS,
    CONF_THRESHOLD,
)

_deepseek_fn = None


def _load_deepseek():
    global _deepseek_fn
    if _deepseek_fn is not None:
        return _deepseek_fn
    import importlib.util
    import os
    path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "v1_rule", "v1_deepseek", "main.py")
    )
    spec = importlib.util.spec_from_file_location("v1_deepseek_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _deepseek_fn = mod.agent
    return _deepseek_fn


def _macro_to_actions_dict(macro, player_id):
    if not macro:
        return []
    return [
        (player_id, atom.target_id, atom.eta, atom.ships)
        for atom in macro
    ]


def atoms_to_step_dict(world, macro, player_id):
    """宏 → apply_action 格式 (src, tgt, ships, angle, eta)。"""
    from atoms_v1 import atom_tuple
    acts = []
    for atom in macro:
        acts.append(atom_tuple(atom))
    return acts


class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.cached_value = 0.0
        self.is_terminal = False

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def u_value(self, c_puct):
        if self.parent is None:
            return 0.0
        return (
            c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        )


class MCTSB:
    def __init__(
        self,
        network,
        c_puct=C_PUCT,
        num_simulations=MCTS_SIMULATIONS,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
        opponent_deepseek_prob=OPPONENT_ROLLOUT_DEEPSEEK_PROB,
    ):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.opponent_deepseek_prob = opponent_deepseek_prob

    def run(self, root_state: WorldState, legal_macros=None, obs_for_deepseek=None, config=None):
        if root_state.is_terminal():
            return None, np.array([])
        root_state = root_state.clone()
        enrich_world(root_state, root_state.my_id)
        if legal_macros is None:
            legal_macros = generate_candidates_b(root_state, root_state.my_id)
        if not legal_macros:
            return None, np.array([])
        root = MCTSNode(root_state)
        self._obs = obs_for_deepseek
        self._config = config
        self._expand(root, forced_legal_macros=legal_macros, add_root_noise=True)

        for _ in range(self.num_simulations):
            node = root
            path = [node]
            while node.is_expanded and node.children:
                node = self._select_child(node)
                path.append(node)
            if not node.is_expanded:
                v = self._expand(node)
            else:
                v = node.cached_value
            self._backup(path, v)

        visits = [
            root.children[macro_key(m)].visit_count
            if macro_key(m) in root.children else 0
            for m in legal_macros
        ]
        probs = np.array(visits, dtype=np.float32)
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(len(legal_macros), dtype=np.float32) / len(legal_macros)
        best_macro = legal_macros[int(np.argmax(probs))]
        return best_macro, probs

    def _infer(self, state, perspective_player, n_legal):
        state_t = encode_state(
            state, perspective_player=perspective_player,
            device=next(self.network.parameters()).device,
        ).unsqueeze(0)
        mask = torch.zeros(1, MAX_MACRO_SLOTS, device=state_t.device)
        mask[0, :n_legal] = 1.0
        with torch.no_grad():
            logits, value = self.network(state_t, macro_mask=mask)
        logits_np = logits.squeeze(0).detach().cpu().numpy()[:n_legal]
        return logits_np, float(value.item())

    def _softmax_priors(self, scores, n):
        if n == 0:
            return np.array([])
        s = np.asarray(scores[:n], dtype=np.float32)
        s = s - s.max()
        exp_s = np.exp(np.clip(s, -80, 80))
        d = exp_s.sum()
        if d <= 0:
            return np.ones(n, dtype=np.float32) / n
        return exp_s / d

    def _sample_opponent_macro(self, state, opponent_id):
        macros = generate_candidates_b(state, opponent_id)
        if not macros:
            return None
        if random.random() < self.opponent_deepseek_prob and self._obs is not None:
            try:
                ds = _load_deepseek()
                ds_obs = self._obs if opponent_id == state.my_id else self._obs
                action = ds(ds_obs, self._config)
                mac, _ = match_env_action_to_macro(action, macros)
                return mac
            except Exception:
                pass
        logits, _ = self._infer(state, opponent_id, min(len(macros), MAX_MACRO_SLOTS))
        priors = self._softmax_priors(logits, len(macros))
        idx = int(np.random.choice(len(macros), p=priors))
        return macros[idx]

    def _transition(self, parent_state, macro_action):
        new_state = parent_state.clone()
        enrich_world(new_state, new_state.my_id)
        actions_dict = {pid: [] for pid in new_state.player_ids}
        pid = new_state.my_id
        for atom in macro_action:
            actions_dict[pid].append(
                (atom.src_id, atom.target_id, atom.ships, atom.angle, atom.eta)
            )
        for opp in new_state.player_ids:
            if opp == pid:
                continue
            opp_view = new_state.clone()
            opp_view.my_id = opp
            enrich_world(opp_view, opp)
            mac = self._sample_opponent_macro(opp_view, opp)
            if mac:
                for atom in mac:
                    actions_dict[opp].append(
                        (atom.src_id, atom.target_id, atom.ships, atom.angle, atom.eta)
                    )
        new_state.step(actions_dict)
        enrich_world(new_state, new_state.my_id)
        return new_state

    def _terminal_value(self, state):
        scores_map = state.get_scores()
        ev = env_terminal_value(scores_map, state.my_id, list(state.player_ids))
        best_other = max(
            (float(scores_map.get(p, 0.0)) for p in state.player_ids if p != state.my_id),
            default=0.0,
        )
        my_s = float(scores_map.get(state.my_id, 0.0))
        diff = (my_s - best_other) / max(1.0, float(VALUE_TARGET_SCALE))
        return 0.5 * ev + 0.5 * float(np.tanh(diff))

    def _expand(self, node, forced_legal_macros=None, add_root_noise=False):
        if node.state is None:
            node.state = self._transition(node.parent.state, node.action)

        if node.state.is_terminal():
            node.is_terminal = True
            node.cached_value = self._terminal_value(node.state)
            node.is_expanded = True
            return node.cached_value

        macros = forced_legal_macros
        if macros is None:
            macros = generate_candidates_b(node.state, node.state.my_id)
        n = len(macros)
        logits, net_v = self._infer(node.state, node.state.my_id, min(n, MAX_MACRO_SLOTS))
        priors = self._softmax_priors(logits, n)
        if add_root_noise and n > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * n)
            priors = (1.0 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise

        node.is_expanded = True
        node.cached_value = net_v
        for macro, prior in zip(macros, priors):
            key = macro_key(macro)
            node.children[key] = MCTSNode(None, parent=node, action=macro, prior=float(prior))
        return node.cached_value

    def _select_child(self, node):
        return max(node.children.values(), key=lambda c: c.q_value() + c.u_value(self.c_puct))

    def _backup(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value


def pick_macro_with_fallback(net, world, macros, obs, config, n_simulations, use_eval_fallback=False):
    from candidates_b import roi_fallback_macro

    mcts = MCTSB(net, num_simulations=n_simulations)
    macro, probs = mcts.run(world, macros, obs_for_deepseek=obs, config=config)
    if macro is None:
        return roi_fallback_macro(macros), probs
    if probs.max() >= CONF_THRESHOLD:
        return macro, probs
    if use_eval_fallback:
        macro = _evaluate_pick(world, macros, config)
        if macro is not None:
            return macro, probs
    return roi_fallback_macro(macros) or macro, probs


def _evaluate_pick(world, macros, config):
    import importlib.util
    import os
    import time
    path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "v1_rule", "v1_deepseek", "main.py")
    )
    spec = importlib.util.spec_from_file_location("v1ds_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    deadline = time.perf_counter() + 0.5
    best, best_score = [], -float("inf")
    try:
        best_score = mod.evaluate(world, [], deadline)
    except Exception:
        pass
    from atoms_v1 import macro_score as ms
    sorted_c = sorted([m for m in macros if m], key=lambda c: -ms(c))
    for cand in sorted_c:
        if time.perf_counter() > deadline - 0.05:
            break
        try:
            score = mod.evaluate(world, list(cand), deadline)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best = cand
    return best if best else None
