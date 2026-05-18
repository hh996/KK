"""
手动评估脚本：加载 checkpoint，对阵 random / deepseek / 历史版本。

用法：
  python run_eval.py                          # 自动加载最新 checkpoint
  python run_eval.py --ckpt interrupt         # 加载 interrupt.pt
  python run_eval.py --ckpt iter_50           # 加载 iter_50.pt
  python run_eval.py --episodes 20            # 每项跑 20 局
  python run_eval.py --agents 4              # 4 人局
  python run_eval.py --vs random             # 只测 random
  python run_eval.py --vs deepseek           # 只测 deepseek
  python run_eval.py --vs checkpoint --ckpt2 iter_30  # 当前 vs 旧版本
"""

import argparse
import os
import sys

import torch

# 确保 v3_NN 目录在 path 里
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHECKPOINT_DIR,
    CHANNELS,
    BOARD_SIZE,
    RES_BLOCKS,
    RES_FILTERS,
    DEVICE,
)
from network import PolicyValueNetwork
from eval_orbit_wars import (
    evaluate_net_vs_random,
    evaluate_net_vs_deepseek,
    evaluate_net_vs_checkpoint,
)


def load_net(ckpt_name):
    path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pt")
    if not os.path.exists(path):
        print(f"[错误] 找不到 checkpoint: {path}")
        sys.exit(1)
    net = PolicyValueNetwork(CHANNELS, BOARD_SIZE, RES_BLOCKS, RES_FILTERS).to(DEVICE)
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(ck["model_state_dict"])
    net.eval()
    iteration = ck.get("iteration", "?")
    print(f"已加载 {path}  (iteration={iteration}, device={DEVICE})")
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default="latest",   help="checkpoint 名（不含 .pt）")
    parser.add_argument("--ckpt2",    default=None,       help="对比用的旧 checkpoint（--vs checkpoint 时）")
    parser.add_argument("--episodes", type=int, default=12, help="每项评估的局数")
    parser.add_argument("--agents",   type=int, default=2,  choices=[2, 4], help="玩家数")
    parser.add_argument("--vs",       default="all",
                        choices=["all", "random", "deepseek", "checkpoint"],
                        help="对战对象")
    args = parser.parse_args()

    net = load_net(args.ckpt)
    ep  = args.episodes
    na  = args.agents

    print(f"\n评估配置：{ep} 局 / {na}P\n{'='*40}")

    if args.vs in ("all", "random"):
        wins, total = evaluate_net_vs_random(net, episodes=ep, num_agents=na)
        print(f"vs random   : {wins}/{total}  ({wins/total*100:.0f}%)")

    if args.vs in ("all", "deepseek"):
        wins, total = evaluate_net_vs_deepseek(net, episodes=ep, num_agents=na)
        print(f"vs deepseek : {wins}/{total}  ({wins/total*100:.0f}%)")

    if args.vs == "checkpoint":
        if args.ckpt2 is None:
            print("[错误] --vs checkpoint 需要指定 --ckpt2")
            sys.exit(1)
        ckpt2_path = os.path.join(CHECKPOINT_DIR, f"{args.ckpt2}.pt")
        wins, total = evaluate_net_vs_checkpoint(net, ckpt2_path, episodes=ep, num_agents=na)
        print(f"vs {args.ckpt2} : {wins}/{total}  ({wins/total*100:.0f}%)")

    print("="*40)


if __name__ == "__main__":
    main()
