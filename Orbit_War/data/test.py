# test.py
from kaggle_environments import make

# 初始化 Orbit Wars 环境（debug=True 打印详细日志）
env = make("orbit_wars", debug=True)

# 运行游戏：[你的agent文件, 对手agent]
# 可选对手："random"（随机行动）、"none"（无行动）、或其他自定义agent文件
steps = env.run(["main.py", "random"])

# 打印最终结果（各玩家的奖励/状态）
final_step = steps[-1]
for player_id, step in enumerate(final_step):
    print(f"玩家 {player_id}：")
    print(f"  奖励（总舰船数）：{step.reward}")
    print(f"  状态：{step.status}")  # DONE=正常结束, ERROR=代码报错, TIMEOUT=超时

# （可选）在 Notebook 中可视化游戏过程（需 ipython 环境）
# env.render(mode="ipython", width=800, height=600)