"""
在 ``WorldState`` 的 **宏动作** 空间上做单步 PUCT 式 MCTS，用 ``PolicyValueNetwork`` 提供先验与叶价值。

与 ``train`` / ``eval_orbit_wars`` 的衔接：
    调用方传入 **根局面**（``my_id`` 为当前玩家）及本步 **合法宏列表** ``legal_macros``；
    ``run`` 返回 **visit 占比最大的宏**（实际落子）及 **visit 归一化分布**（供 ``build_policy_targets`` 做监督）。

仿真一步（``_transition``）：根玩家走候选宏；其余玩家各从网络先验对合法宏 **多项式采样** 一次宏，
    再 ``WorldState.step(actions_dict)``。这是启发式联机，**不是**联合动作纳什搜索。

注意：``BOARD_SIZE`` 此处来自 ``config``（特征网格边长，与 ``encode_state`` 一致）；若未来世界
``board_size`` 与特征图不一致，格点索引需与 ``features`` 一并调整。
"""
import math
import numpy as np
import torch
from physics import WorldState
from features import encode_state
from value_util import env_terminal_value
from config import (
    BOARD_SIZE,
    C_PUCT,
    MCTS_SIMULATIONS,
    DIRICHLET_ALPHA,
    DIRICHLET_EPSILON,
    VALUE_TARGET_SCALE,
    OPPONENT_SAMPLES,
    SHIP_BUCKET_COUNT,
)


def ship_bucket_idx(ships, n_bucket=SHIP_BUCKET_COUNT):
    s = max(1, int(ships))
    if n_bucket <= 2:
        return 0
    # n_bucket = 11: 1-9 各自对应 0..8，10及以上挤入桶9
    if s <= n_bucket - 2:  # s <= 9
        return s - 1        # 0..8
    return n_bucket - 1     # 9


# ---------------------------------------------------------------------------
# 搜索树节点（每条边 = 一个宏动作；子节点懒展开、state 在首次展开时生成）
# ---------------------------------------------------------------------------
class MCTSNode:
    """PUCT 节点：``children`` 的键为 **宏动作 tuple**（可哈希）。"""

    def __init__(self, state, parent=None, action=None, prior=0.0):
        # state：根为真实 clone；非根在展开前可为 None，见 _expand
        self.state = state
        self.parent = parent
        self.action = action  # 从父节点经哪条宏边到达本节点
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.cached_value = 0.0
        self.is_terminal = False

    def q_value(self):
        """平均回传价值 Q = value_sum / N；未访问过为 0。"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def u_value(self, c_puct):
        """PUCT 探索项 U；根节点无父，返回 0。"""
        if self.parent is None:
            return 0.0
        return (
            c_puct
            * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )


# ---------------------------------------------------------------------------
# MCTS：根噪声 + 模拟 + visit 分布输出
# ---------------------------------------------------------------------------
class MCTS:
    def __init__(
        self,
        network,
        c_puct=C_PUCT,
        num_simulations=MCTS_SIMULATIONS,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
    ):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def run(self, root_state: WorldState, legal_macros=None):
        """
        在当前根状态下搜索，返回 ``(best_macro, probs)``。

        ``probs`` 与 ``legal_macros`` 顺序对齐，为子边 visit 计数归一化分布；
        ``best_macro`` 为 ``argmax(probs)``（非随机，探索靠根 Dirichlet 与模拟次数）。
        """
        if root_state.is_terminal():
            return None, np.array([])
        root_state = root_state.clone()
        if not legal_macros:
            legal_macros = root_state.get_legal_actions(root_state.my_id)
        if not legal_macros:
            return None, np.array([])
        root = MCTSNode(root_state)
        self._expand(root, forced_legal_macros=legal_macros, add_root_noise=True)

        # 每条模拟：从根沿树选子直到叶 → 展开或取缓存 v → 沿路 backup 同一标量 v
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
            root.children[m].visit_count if m in root.children else 0
            for m in legal_macros
        ]
        probs = np.array(visits, dtype=np.float32)
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(len(legal_macros)) / len(legal_macros)
        best_macro = legal_macros[int(np.argmax(probs))]
        return best_macro, probs

    def _infer_policy_value(self, state: WorldState, perspective_player):
        """
        编码 ``state`` 为 ``perspective_player`` 视角张量，前向网络。

        返回展平的 ``src_logits``、``tgt_logits``、``ships_logits``（numpy）及标量 ``value``。
        兼容旧三返回值网络（无 ship 头时补零）。
        """
        state_t = encode_state(
            state,
            perspective_player=perspective_player,
            device=next(self.network.parameters()).device,
        ).unsqueeze(0)
        with torch.no_grad():
            out = self.network(state_t)
            if len(out) == 4:
                src_logits, tgt_logits, ships_logits, value = out
            else:
                src_logits, tgt_logits, value = out
                ships_logits = torch.zeros(
                    src_logits.shape[0],
                    SHIP_BUCKET_COUNT,
                    device=value.device,
                )
        ships_np = ships_logits.detach().squeeze(0).cpu().numpy()
        return (
            src_logits.squeeze(0).detach().cpu().numpy(),
            tgt_logits.squeeze(0).detach().cpu().numpy(),
            ships_np,
            float(value.item()),
        )

    def _macro_prior_scores(self, state, macros, src_logits, tgt_logits, ships_logits):
        """
        对每个宏：其内各原子在棋盘格上的 src/tgt logit 与 ship 桶 logit **求和后除以原子数**。

        缺行星时该项打极大负分，softmax 后接近 0 概率。
        """
        scores = []
        for macro in macros:
            msum = 0.0
            for atom in macro:
                sid, tid, ships, _, _ = atom
                src = state.planets.get(sid)
                tgt = state.planets.get(tid)
                if src is None or tgt is None:
                    msum -= 1e9
                    continue
                sx = max(0, min(BOARD_SIZE - 1, int(src.x)))
                sy = max(0, min(BOARD_SIZE - 1, int(src.y)))
                tx = max(0, min(BOARD_SIZE - 1, int(tgt.x)))
                ty = max(0, min(BOARD_SIZE - 1, int(tgt.y)))
                src_idx = sy * BOARD_SIZE + sx
                tgt_idx = ty * BOARD_SIZE + tx
                bi = ship_bucket_idx(ships)
                msum += (
                    float(src_logits[src_idx])
                    + float(tgt_logits[tgt_idx])
                    + float(ships_logits[bi])
                )
            scores.append(msum / max(1.0, len(macro)))
        return np.array(scores, dtype=np.float32)

    def _softmax_priors(self, scores):
        """数值稳定 softmax；空或退化时均匀分布。"""
        if scores.size == 0:
            return scores
        scores = scores - scores.max()
        exp_s = np.exp(np.clip(scores, -80, 80))
        d = exp_s.sum()
        if d <= 0:
            return np.ones(len(scores), dtype=np.float32) / len(scores)
        return exp_s / d

    def _sample_opponent_macro(self, state, opponent_id):
        """
        用网络在 **对手视角** 下对 ``opponent_id`` 的合法宏算先验，再 ``np.random.choice``。

        重复 ``OPPONENT_SAMPLES`` 次，取 **最后一次** 样本，增大方差。
        """
        macros = state.get_legal_macro_actions(opponent_id)
        if not macros:
            return None
        (
            src_logits,
            tgt_logits,
            ships_logits,
            _,
        ) = self._infer_policy_value(state, perspective_player=opponent_id)
        priors = self._softmax_priors(
            self._macro_prior_scores(
                state, macros, src_logits, tgt_logits, ships_logits
            )
        )
        sampled = macros[0]
        for _ in range(max(1, int(OPPONENT_SAMPLES))):
            sampled = macros[int(np.random.choice(len(macros), p=priors))]
        return sampled

    def _transition(self, parent_state, macro_action):
        """
        根玩家执行 ``macro_action``；其他 pid 各采样一宏；``step`` 推进一整个环境 tick。

        ``macro_action`` 为原子元组序列，如 ``(atom1,)`` 或 ``(atom1, atom2)``。
        """
        new_state = parent_state.clone()
        actions_dict = {pid: [] for pid in new_state.player_ids}
        actions_dict[new_state.my_id] = list(macro_action)
        for pid in new_state.player_ids:
            if pid == new_state.my_id:
                continue
            opp = self._sample_opponent_macro(new_state, pid)
            if opp is not None:
                actions_dict[pid] = list(opp)
        new_state.step(actions_dict)
        return new_state

    def _terminal_value(self, state):
        """
        终局叶子的标量价值：``0.5 * env_terminal_value``（与 Kaggle ±1 语义对齐的胜负）
        + ``0.5 * tanh((我方总分 - 最强对手) / VALUE_TARGET_SCALE)`` 平滑船差信号。

        非终局叶用网络 ``net_v``，见 ``_expand``。
        """
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
        """
        若子节点尚无 ``state``，则 ``_transition(parent, action)`` 生成；若终局则算 ``_terminal_value``。

        否则用网络得到 ``src/tgt/ship`` logits 与 ``net_v``；对 ``forced_legal_macros`` 或当前合法宏
        算先验 softmax；根上可选 **Dirichlet 与先验凸组合**；为每个宏建子节点（子 ``state`` 仍为 None
        直至被选中再展开）。返回写入 ``cached_value`` 的标量（展开用 ``net_v``，终局用终端值）。
        """
        if node.state is None:
            node.state = self._transition(node.parent.state, node.action)

        if node.state.is_terminal():
            node.is_terminal = True
            node.cached_value = self._terminal_value(node.state)
            node.is_expanded = True
            return node.cached_value

        (
            src_logits,
            tgt_logits,
            ships_logits,
            net_v,
        ) = self._infer_policy_value(node.state, perspective_player=node.state.my_id)

        macros = forced_legal_macros
        if macros is None:
            macros = node.state.get_legal_actions(node.state.my_id)

        scores = self._macro_prior_scores(
            node.state, macros, src_logits, tgt_logits, ships_logits
        )
        priors = self._softmax_priors(scores)
        if add_root_noise and len(priors) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
            priors = (
                1.0 - self.dirichlet_epsilon
            ) * priors + self.dirichlet_epsilon * noise

        node.is_expanded = True
        node.cached_value = net_v
        for macro, prior in zip(macros, priors):
            node.children[macro] = MCTSNode(
                None, parent=node, action=macro, prior=float(prior)
            )
        return node.cached_value

    def _select_child(self, node):
        """PUCT：``argmax(Q + U)``；``U`` 与先验及父 visit 有关。"""
        return max(
            node.children.values(),
            key=lambda child: child.q_value() + child.u_value(self.c_puct),
        )

    def _backup(self, path, value):
        """路径上每个节点累加 ``visit_count`` 与 ``value_sum``（当前实现：全路径同一 ``value``）。"""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
