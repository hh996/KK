import numpy as np
import torch
from physics import WorldState, CENTER_X, CENTER_Y, SUN_R
from config import BOARD_SIZE, CHANNELS, DEVICE


# 太阳 mask 固定不变，模块加载时预计算一次
_sun_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
for _y in range(BOARD_SIZE):
    for _x in range(BOARD_SIZE):
        if (_x - CENTER_X) ** 2 + (_y - CENTER_Y) ** 2 <= SUN_R ** 2:
            _sun_mask[_y, _x] = 1.0


def encode_state(world: WorldState, perspective_player=None) -> torch.Tensor:
    if perspective_player is None:
        perspective_player = world.my_id
    H, W = BOARD_SIZE, BOARD_SIZE
    grid = np.zeros((CHANNELS, H, W), dtype=np.float32)   # CHANNELS 变为 17
    step_norm = min(1.0, world.step_count / 500.0)

    opp_ids = sorted([oid for oid in world.player_ids if oid != perspective_player])
    # 确保最多 3 个敌人
    opp_ids = opp_ids[:3]

    for p in world.planet_list:
        cx, cy = int(p.x), int(p.y)
        r = int(np.ceil(p.radius))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        if p.owner == perspective_player:
                            grid[0, ny, nx] = max(grid[0, ny, nx], p.ships / 100.0)
                            grid[1, ny, nx] = max(grid[1, ny, nx], p.production / 5.0)
                        elif p.owner == -1:
                            grid[2, ny, nx] = max(grid[2, ny, nx], p.ships / 100.0)
                        elif p.owner in opp_ids:
                            idx = opp_ids.index(p.owner)
                            if idx < 3:
                                base_ch = 3 + idx * 2  # 3,5,7
                                grid[base_ch, ny, nx] = max(grid[base_ch, ny, nx], p.ships / 100.0)
                                grid[base_ch + 1, ny, nx] = max(grid[base_ch + 1, ny, nx], p.production / 5.0)

    # 彗星标记 (ch9)
    for pid in world.comet_ids:
        p = world.planets.get(pid)
        if p is not None:
            cx, cy = int(p.x), int(p.y)
            if 0 <= cx < W and 0 <= cy < H:
                grid[9, cy, cx] = 1.0

    # 己方舰队 (ch10)
    for f in world.fleets:
        fx, fy = int(f.x), int(f.y)
        if 0 <= fx < W and 0 <= fy < H:
            density = min(1.0, f.ships / 200.0)
            if f.owner == perspective_player:
                grid[10, fy, fx] = max(grid[10, fy, fx], density)
            else:
                # 敌人舰队分配到 ch11-13
                try:
                    idx = opp_ids.index(f.owner)
                    if idx < 3:
                        grid[11 + idx, fy, fx] = max(grid[11 + idx, fy, fx], density)
                except ValueError:
                    pass  # 未知 owner，忽略

    # 全局量
    grid[14, :, :] = step_norm
    grid[15, :, :] = (float(len(world.player_ids)) - 2.0) / 2.0  # 2→0, 4→1
    grid[16] = _sun_mask

    return torch.from_numpy(grid).to(DEVICE)
