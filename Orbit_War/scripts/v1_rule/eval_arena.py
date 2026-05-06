"""ELO arena for systematic A/B comparison of Orbit Wars agents.

Usage:
    from eval_arena import Arena
    arena = Arena({"v1_new": agent_a, "v1_old": agent_b, "random": "random"})
    report = arena.round_robin(n_episodes=10)
    arena.print_summary()

The Arena runs round-robin matches between every pair of agents, tracks
ELO ratings, win rates, average score deltas, and per-turn p95 timing.

Pass "random" or "reaction" as a string for built-in Kaggle baseline agents.
"""

import itertools
import statistics
import time
from collections import defaultdict

from kaggle_environments import make


def _is_builtin(agent_ref):
    return isinstance(agent_ref, str)


class Arena:
    def __init__(self, agents):
        self.agents = agents                       # {name: callable or "random"}
        self.elo = {n: 1500.0 for n in agents}
        self.results = []                          # raw match log
        self.match_summary = {}                    # {(a,b): summary dict}

    # ---------- core match loop ----------

    def play_match(self, name_a, name_b, n_episodes=10, verbose=False):
        wins_a = 0
        draws = 0
        deltas = []
        time_a, time_b = [], []

        for ep in range(n_episodes):
            agent_a = self._wrap_timed(self.agents[name_a], time_a)
            agent_b = self._wrap_timed(self.agents[name_b], time_b)
            env = make("orbit_wars")
            env.run([agent_a, agent_b])
            r_a = env.steps[-1][0]["reward"]
            r_b = env.steps[-1][1]["reward"]
            score_a = self._final_score(env, 0)
            score_b = self._final_score(env, 1)
            deltas.append(score_a - score_b)
            if r_a is None: r_a = 0
            if r_b is None: r_b = 0
            if r_a > r_b:
                wins_a += 1
                outcome = "A"
            elif r_a < r_b:
                outcome = "B"
            else:
                draws += 1
                outcome = "D"
            self.results.append({
                "a": name_a, "b": name_b, "ep": ep,
                "outcome": outcome, "r_a": r_a, "r_b": r_b,
                "score_a": score_a, "score_b": score_b,
                "steps": len(env.steps),
            })
            if verbose:
                print(f"    ep{ep+1}: {outcome}  "
                      f"score {score_a:>4} vs {score_b:>4}  "
                      f"steps {len(env.steps)}")

        wr_a = (wins_a + 0.5 * draws) / n_episodes
        return {
            "n": n_episodes,
            "wr_a": wr_a,
            "wins_a": wins_a,
            "wins_b": n_episodes - wins_a - draws,
            "draws": draws,
            "delta_mean": statistics.mean(deltas) if deltas else 0.0,
            "delta_stdev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "time_a_mean_ms": 1000 * statistics.mean(time_a) if time_a else 0.0,
            "time_b_mean_ms": 1000 * statistics.mean(time_b) if time_b else 0.0,
            "time_a_p95_ms": self._p95_ms(time_a),
            "time_b_p95_ms": self._p95_ms(time_b),
            "time_a_max_ms": 1000 * max(time_a) if time_a else 0.0,
            "time_b_max_ms": 1000 * max(time_b) if time_b else 0.0,
        }

    def round_robin(self, n_episodes=10, verbose=False):
        for a, b in itertools.combinations(self.agents.keys(), 2):
            print(f"  {a} vs {b}: ", end="", flush=True)
            res = self.play_match(a, b, n_episodes, verbose=verbose)
            self.update_elo(a, b, res["wr_a"], n=n_episodes)
            self.match_summary[(a, b)] = res
            print(
                f"WR={res['wr_a']*100:>5.1f}%  "
                f"({res['wins_a']}-{res['wins_b']}-{res['draws']})  "
                f"delta={res['delta_mean']:+6.0f}  "
                f"a_p95={res['time_a_p95_ms']:>5.0f}ms  "
                f"b_p95={res['time_b_p95_ms']:>5.0f}ms"
            )
        return self.match_summary

    # ---------- ELO ----------

    def update_elo(self, a, b, wr_a, n=1, K=32):
        ra, rb = self.elo[a], self.elo[b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
        # Apply K once per episode (sum of per-game updates ~ K * n * (wr - ea))
        self.elo[a] += K * n * (wr_a - ea)
        self.elo[b] += K * n * ((1 - wr_a) - (1 - ea))

    def print_summary(self):
        print("\n=== ELO ranking ===")
        for name, elo in sorted(self.elo.items(), key=lambda x: -x[1]):
            print(f"  {name:<14} {elo:>7.1f}")

        print("\n=== Win-rate matrix (rows = agent A, cols = agent B, value = A's win rate) ===")
        names = list(self.agents.keys())
        header = "                 " + " ".join(f"{n:>10}" for n in names)
        print(header)
        for a in names:
            row = [f"{a:<14}"]
            for b in names:
                if a == b:
                    row.append(f"{'-':>10}")
                elif (a, b) in self.match_summary:
                    row.append(f"{self.match_summary[(a, b)]['wr_a']*100:>9.1f}%")
                elif (b, a) in self.match_summary:
                    row.append(f"{(1 - self.match_summary[(b, a)]['wr_a'])*100:>9.1f}%")
                else:
                    row.append(f"{'?':>10}")
            print("  " + " ".join(row))

    # ---------- helpers ----------

    def _wrap_timed(self, agent_ref, store):
        if _is_builtin(agent_ref):
            return agent_ref
        def wrapper(obs, config=None):
            t0 = time.perf_counter()
            try:
                out = agent_ref(obs, config)
            except TypeError:
                out = agent_ref(obs)
            store.append(time.perf_counter() - t0)
            return out
        return wrapper

    @staticmethod
    def _final_score(env, idx):
        obs = env.steps[-1][idx]["observation"]
        s = 0
        for p in obs.get("planets", []):
            if p[1] == idx:
                s += p[5]
        for f in obs.get("fleets", []):
            if f[1] == idx:
                s += f[6]
        return int(s)

    @staticmethod
    def _p95_ms(times):
        if not times:
            return 0.0
        if len(times) < 20:
            return 1000 * max(times)
        return 1000 * statistics.quantiles(times, n=20)[18]
