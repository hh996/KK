# v1 测试
from kaggle_environments import make

def test_match(agent1, agent2, num_games=5):
    """运行多局对战，统计胜率"""
    wins = {0: 0, 1: 0, 'draw': 0}
    total_rewards = {0: 0, 1: 0}

    for i in range(num_games):
        try:
            env = make("orbit_wars", debug=False)
            steps = env.run([agent1, agent2])

            final_step = steps[-1]
            p0_reward = final_step[0].reward
            p1_reward = final_step[1].reward

            total_rewards[0] += p0_reward
            total_rewards[1] += p1_reward

            if p0_reward > p1_reward:
                wins[0] += 1
            elif p1_reward > p0_reward:
                wins[1] += 1
            else:
                wins['draw'] += 1

            print(f"Game {i+1}: P0={p0_reward:.0f}, P1={p1_reward:.0f}")

        except Exception as e:
            print(f"Game {i+1} error: {e}")
            continue

    print(f"\nResults over {num_games} games:")
    print(f"  {agent1} wins: {wins[0]} ({wins[0]/num_games*100:.1f}%)")
    print(f"  {agent2} wins: {wins[1]} ({wins[1]/num_games*100:.1f}%)")
    print(f"  Draws: {wins['draw']}")
    print(f"  Avg reward P0: {total_rewards[0]/num_games:.1f}")
    print(f"  Avg reward P1: {total_rewards[1]/num_games:.1f}")

    return wins

if __name__ == "__main__":
    # v1 测试, 测试 main.py vs random
    print("\n" + "=" * 50)
    print("Testing: main.py vs random")
    print("=" * 50)
    test_match("Orbit_War/data/main.py", "random", num_games=5)
