import os
import torch

# =============================================================================
# 路径：权重与日志目录（相对本文件所在目录 v3_NN/）
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")  # train 保存 *.pt
LOG_DIR = os.path.join(BASE_DIR, "logs")  # 预留；当前 train 主要 print，未写 TensorBoard
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================================
# 棋盘与输入张量（须与 features.encode_state、network 输入一致）
# =============================================================================
BOARD_SIZE = 100  # 局面编码为 100×100 网格，与 physics 中坐标范围一致
CHANNELS = 17  # 特征通道数；各通道含义见 features.py

# =============================================================================
# 网络：残差主体宽度与深度（PolicyValueNetwork）
# =============================================================================
RES_BLOCKS = 6  # 残差块重复次数，越大容量越高、越慢
RES_FILTERS = 64  # 卷积通道数（body 宽度）

# =============================================================================
# MCTS（mcts.py）
# =============================================================================
# 每次「选一步棋」时在搜索树上做多少次模拟；越大越强、越慢。训练可酌情调低。
MCTS_SIMULATIONS = 100
# PUCT 探索系数：Q+U 里 U 项权重。越大越愿意试先验低但尚未访问的分支。
C_PUCT = 1.5
# 虚拟损失：用于多线程 MCTS 时避免多线程同时选同一叶子的技巧强度（本仓库 MCTS 未使用）。
VIRTUAL_LOSS = 3.0
# 根节点先验混合 Dirichlet 噪声：alpha 越小噪声越尖、越大越平。
DIRICHLET_ALPHA = 0.3
# 根先验 = (1-eps)*网络先验 + eps*Dirichlet；增大 eps 探索更强。
DIRICHLET_EPSILON = 0.15

# =============================================================================
# 自我对弈与评估（train.py、eval_orbit_wars.py）
# =============================================================================
# 每个「训练迭代」连续跑几局 self-play 再更新网络；局数越多数据越多、越慢。
NUM_PARALLEL_GAMES = 8
# 传给 kaggle orbit_wars 的 episode 最大步数（与环境默认一致即可）。
MAX_GAME_STEPS = 500
# 以该概率开 2 人局，否则 4 人局；混合可让网络见过两种人数。
TRAIN_TWO_PLAYER_PROB = 0.5
# 单回合「宏动作」里最多几支舰队；须与 Kaggle env 每回合可发舰队数一致。
MAX_FLEETS_PER_TURN = 2
# MCTS 模拟对手时：按网络先验对合法宏采样几次，取最后一次（增大则对手更随机）。
OPPONENT_SAMPLES = 1
# eval_orbit_wars 里 quick eval 的局数上限（train 里还会再 min 一次）。
GAME_EVAL_EPISODES = 12

# =============================================================================
# 训练（train.py）
# =============================================================================
BATCH_SIZE = 128  # 每步梯度用的样本数；受显存与 replay 大小约束
LEARNING_RATE = 1e-3  # Adam 学习率
WEIGHT_DECAY = 1e-4  # Adam L2 权重衰减
REPLAY_BUFFER_SIZE = 20000  # 自博弈样本 FIFO 容量
# 每个迭代里调用 train_step 的次数（缓冲区够大时约等于每轮梯度步数；不是「扫一遍数据集」）。
TRAIN_EPOCHS_PER_ITER = 5
POLICY_LOSS_WEIGHT = 1.0  # 总 loss 里 policy（三头交叉熵之和）的系数
VALUE_LOSS_WEIGHT = 1.0  # 总 loss 里 value MSE 的系数
# MCTS 叶节点非纯终局时：终局价值 = 0.5*env 语义 ±1 + 0.5*tanh((我方总分-最强对手)/该尺度)。
# 调大则 tanh 部分更「钝」、数值更接近 0；与「船数」无直接关系。
VALUE_TARGET_SCALE = 1.0
# 派兵数量离散分桶数：network 的 ship 头维度、build_policy_targets / ship_bucket_idx 须一致。
SHIP_BUCKET_COUNT = 11
SAVE_EVERY_ITERS = 20  # 每多少迭代保存 iter_* 与 latest
EVAL_EVERY_ITERS = 50  # 每多少迭代对 random 评估一次；0 可关闭
MAX_ITERATIONS = 100000  # 外层 while 上限；Ctrl+C 会先存 interrupt 再退出
TRAIN_MCTS_SIMULATIONS = 20   # 初始模拟数；达到 MCTS_SIM_BOOST_ITER 后自动升到 MCTS_SIM_FULL
MCTS_SIM_FULL      = 50   # 提升后的模拟数
MCTS_SIM_BOOST_ITER = 150  # 迭代数达到此值后自动切换（与 DEEPSEEK_START_ITER 对齐）

# =============================================================================
# 训练对手课程（Curriculum）
# =============================================================================
# 第一阶段（iter < DEEPSEEK_START_ITER）：纯自博弈
# 第二阶段（iter < POOL_START_ITER）：自博弈 + 以 DEEPSEEK_OPP_PROB 概率混入 deepseek
# 第三阶段（iter >= POOL_START_ITER）：自博弈 + deepseek + 随机旧 checkpoint
DEEPSEEK_START_ITER = 150   # 开始混入 deepseek 的迭代数
POOL_START_ITER     = 400   # 开始使用对手池（旧 checkpoint）的迭代数
DEEPSEEK_OPP_PROB   = 0.5   # 混入 deepseek 的概率

# =============================================================================
# 中断恢复：save/load 时文件名前缀（实际文件为 {name}.pt）
# =============================================================================
INTERRUPT_CHECKPOINT = "interrupt"

# =============================================================================
# 硬件
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
