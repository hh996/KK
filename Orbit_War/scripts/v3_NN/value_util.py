"""与 kaggle orbit_wars 终局 reward 对齐：最高分且 max_score>0 的玩家一律 +1，否则 -1（含并列最高分）。"""


def env_terminal_value(scores_map, pid, participant_ids=None):
    """
    scores_map: player_id -> 总舰船数（行星+舰队）
    participant_ids: 参与训练的 agent id 列表（缺省为 scores_map 全部键）
    返回 {-1.0, 1.0} 与环境中 state[i].reward 一致语义。
    """
    if participant_ids is None:
        participant_ids = sorted(scores_map.keys())
    mx = max((float(scores_map.get(p, 0.0)) for p in participant_ids), default=0.0)
    sp = float(scores_map.get(pid, 0.0))
    if mx > 0 and abs(sp - mx) <= 1e-6:
        return 1.0
    return -1.0
