"""与 kaggle orbit_wars 终局 reward 对齐。"""


def env_terminal_value(scores_map, pid, participant_ids=None):
    if participant_ids is None:
        participant_ids = sorted(scores_map.keys())
    mx = max((float(scores_map.get(p, 0.0)) for p in participant_ids), default=0.0)
    sp = float(scores_map.get(pid, 0.0))
    if mx > 0 and abs(sp - mx) <= 1e-6:
        return 1.0
    return -1.0
