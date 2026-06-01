"""从 v3 checkpoint 迁移 backbone + value 头（跳过旧 policy 头）。"""
import torch

from config import V3_BACKBONE_PATH, CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS
from network import PolicyValueNetworkB


def load_v3_backbone_into(net: PolicyValueNetworkB, path=None, device="cpu"):
    path = path or V3_BACKBONE_PATH
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"[load_v3_backbone] 未找到 {path}，跳过迁移。")
        return False
    old_sd = ck.get("model_state_dict", ck)
    new_sd = net.state_dict()
    transferred = 0
    skip_prefixes = ("policy_src", "policy_tgt", "policy_ship", "policy_fc")
    for k, v in old_sd.items():
        if any(k.startswith(p) for p in skip_prefixes):
            continue
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v
            transferred += 1
    net.load_state_dict(new_sd, strict=False)
    print(f"[load_v3_backbone] 从 {path} 迁移 {transferred} 个 tensor。")
    return True
