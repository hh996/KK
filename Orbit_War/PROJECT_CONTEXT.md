
  iter 0–149    MCTS=20，纯自博弈
  iter 150–399  MCTS=50，自博弈 + 50% deepseek 对手
  iter 400+     MCTS=50，自博弈 + deepseek + 旧 checkpoint 对手池

  三个关键阈值全在 config.py 里，想调就改那一个文件：

  TRAIN_MCTS_SIMULATIONS = 20   # 初始模拟数
  MCTS_SIM_FULL           = 50  # 提升后的模拟数
  MCTS_SIM_BOOST_ITER     = 150 # ← 改这个控制何时提升
  DEEPSEEK_START_ITER     = 150 # ← 改这个控制何时加 deepseek
  POOL_START_ITER         = 400 # ← 改这个控制何时加对手池