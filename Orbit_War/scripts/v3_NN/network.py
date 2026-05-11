import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SHIP_BUCKET_COUNT


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class PolicyValueNetwork(nn.Module):
    def __init__(self, in_channels, board_size, res_blocks=6, filters=64):
        super().__init__()
        self.board_size = board_size
        self.conv_input = nn.Conv2d(in_channels, filters, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(filters)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(filters) for _ in range(res_blocks)]
        )

        self.policy_src_conv = nn.Conv2d(filters, 1, 1, bias=True)
        self.policy_tgt_conv = nn.Conv2d(filters, 1, 1, bias=True)

        self.policy_ship_mlp = nn.Sequential(
            nn.Linear(filters, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, SHIP_BUCKET_COUNT),
        )

        self.value_conv = nn.Conv2d(filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)

        gap = torch.mean(x, dim=(2, 3))
        ships_logits = self.policy_ship_mlp(gap)

        src_logits = self.policy_src_conv(x).view(x.size(0), -1)
        tgt_logits = self.policy_tgt_conv(x).view(x.size(0), -1)

        v = self.value_conv(x)
        v = F.relu(self.value_bn(v))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return src_logits, tgt_logits, ships_logits, v


class ValueNetwork(PolicyValueNetwork):
    """兼容旧名称。"""

