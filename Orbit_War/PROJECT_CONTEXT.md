
iter 0–149    MCTS=20，纯自博弈
iter 150–399  MCTS=50，自博弈 + 50% deepseek 对手
iter 400+     MCTS=50，自博弈 + deepseek + 旧 checkpoint 对手池

三个关键阈值全在 config.py 里，想调就改那一个文件：

TRAIN_MCTS_SIMULATIONS = 20   # 初始模拟数
MCTS_SIM_FULL           = 50  # 提升后的模拟数
MCTS_SIM_BOOST_ITER     = 150 # ← 改这个控制何时提升
DEEPSEEK_START_ITER     = 150 # ← 改这个控制何时加 deepseek
POOL_START_ITER         = 400 # ← 改这个控制何时加对手池

后续应该看到什么：
- iter 1-50：loss 快速下降但意义有限（buffer 还小，信噪比低）
- iter 50：第一次出现 win rate 数据，这才是真正的信号
- 正常目标：vs random 胜率 iter 100 左右达到 70%+，vs deepseek 要到 iter 300+ 才有机会上升
- value_loss 最终应稳定在 0.05-0.20，policy_loss 稳定在 5-10