"""Temps de calcul sur l'instance 48 h : coût d'un pas MCTS (par budget) contre
le coût de la PD complète.

Lancement :  python experiments/run_timing.py [n_repeat]
"""

import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.environment import EnergyStorageEnv, make_synthetic_data
from core.baselines import policy_threshold, dp_optimal
from core.mcts import MCTSPlanner

BUDGETS = [100, 300, 1000]


def main(n_repeat=3):
    prices, production, demand = make_synthetic_data(days=2, seed=1)
    env = EnergyStorageEnv(prices, production, demand=demand, p_consume=70.0,
                           capacity=120.0, eta_charge=0.95, eta_discharge=0.95,
                           max_charge=30.0, max_discharge=30.0,
                           soc0=0.0, n_actions=11)
    thr = policy_threshold(env)

    print("Materiel : %s | Python %s"
          % (platform.processor() or platform.machine(),
             platform.python_version()))
    print("Instance : H = %d, %d actions\n" % (env.H, env.n_actions))

    dp_times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        dp_optimal(env, n_soc=241)
        dp_times.append(time.perf_counter() - t0)
    print("Programmation dynamique (n_soc=241) : %.3f s (épisode complet)"
          % np.mean(dp_times))

    print("\nMCTS (rollout informe) :")
    print("%-10s %14s %16s" % ("budget", "s / pas", "s / épisode (H pas)"))
    for b in BUDGETS:
        planner = MCTSPlanner(env, n_simulations=b, c=1.0,
                              rollout_policy=thr, seed=0)
        step_times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            planner.plan_action(0, env.soc0)
            step_times.append(time.perf_counter() - t0)
        ep_times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            MCTSPlanner(env, n_simulations=b, c=1.0,
                        rollout_policy=thr, seed=0).run()
            ep_times.append(time.perf_counter() - t0)
        print("%-10d %14.4f %16.3f"
              % (b, np.mean(step_times), np.mean(ep_times)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
