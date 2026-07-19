"""Sensibilité du MCTS au budget de simulations et à la constante c.

Même instance que run_demo.py. Pour chaque config, le MCTS est joué sur tout
l'horizon, répété sur plusieurs graines (gain d'arbitrage moyen +/- écart-type).

Lancement :  python experiments/run_sensitivity.py [n_seeds]
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIGDIR = ROOT / "figures"

from core.environment import EnergyStorageEnv, make_synthetic_data
from core.baselines import policy_always_sell, policy_threshold, dp_optimal
from core.mcts import MCTSPlanner

BUDGETS = [30, 100, 300, 1000, 3000]
C_VALUES = [0.25, 0.5, 1.0, 2.0]
C_REF = 1.0        
BUDGET_REF = 300     


def arbitrage(profit, base, opt):
    return 100.0 * (profit - base) / (opt - base)


def run_batch(env, n_sim, c, rollout, seeds):
    """Profits du MCTS pour chaque graine."""
    out = []
    for sd in seeds:
        p, _, _ = MCTSPlanner(env, n_simulations=n_sim, c=c,
                              rollout_policy=rollout, seed=sd).run()
        out.append(p)
    return np.array(out)


def main(n_seeds=5):
    t0 = time.time()
    seeds = list(range(n_seeds))
    prices, production, demand = make_synthetic_data(days=2, seed=1)
    env = EnergyStorageEnv(prices, production, demand=demand, p_consume=70.0,
                           capacity=120.0, eta_charge=0.95, eta_discharge=0.95,
                           max_charge=30.0, max_discharge=30.0,
                           soc0=0.0, n_actions=11)

    base, _, _ = env.rollout_policy(policy_always_sell(env))
    dp_policy, _ = dp_optimal(env, n_soc=241)
    opt, _, _ = env.rollout_policy(dp_policy)
    thr = policy_threshold(env)
    thr_profit, _, _ = env.rollout_policy(thr)
    print("H = %d | sans stockage %.1f | seuil %.1f (%.1f%%) | optimum %.1f"
          % (env.H, base, thr_profit, arbitrage(thr_profit, base, opt), opt))

    rollouts = [("rollout aleatoire", None), ("rollout seuil", thr)]

    # étude 1 : budget de simulations
    curves = {}
    print("\n[1] Gain d'arbitrage (%%) vs budget (c = %.2f, %d graines)" % (C_REF, n_seeds))
    print("%-18s" % "budget" + "".join("%14d" % b for b in BUDGETS))
    for name, ro in rollouts:
        means, stds = [], []
        for b in BUDGETS:
            profs = run_batch(env, b, C_REF, ro, seeds)
            arbs = arbitrage(profs, base, opt)
            means.append(arbs.mean())
            stds.append(arbs.std())
        curves[name] = (np.array(means), np.array(stds))
        print("%-18s" % name
              + "".join("%7.1f +-%4.1f" % (m, s) for m, s in zip(means, stds)))

    # étude 2 : constante d'exploration c
    curves_c = {}
    print("\n[2] Gain d'arbitrage (%%) vs constante c (budget = %d)" % BUDGET_REF)
    print("%-18s" % "c" + "".join("%14.2f" % c for c in C_VALUES))
    for name, ro in rollouts:
        means, stds = [], []
        for c in C_VALUES:
            profs = run_batch(env, BUDGET_REF, c, ro, seeds)
            arbs = arbitrage(profs, base, opt)
            means.append(arbs.mean())
            stds.append(arbs.std())
        curves_c[name] = (np.array(means), np.array(stds))
        print("%-18s" % name
              + "".join("%7.1f +-%4.1f" % (m, s) for m, s in zip(means, stds)))

    print("\nDuree totale : %.0f s" % (time.time() - t0))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

        for name, (m, s) in curves.items():
            ax[0].errorbar(BUDGETS, m, yerr=s, marker="o", capsize=3, label="MCTS " + name)
        ax[0].axhline(100, color="green", ls="--", lw=1, label="Optimum (PD)")
        ax[0].axhline(arbitrage(thr_profit, base, opt), color="grey", ls=":",
                      lw=1, label="Heuristique seuil")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Simulations par pas (log)")
        ax[0].set_ylabel("% de la valeur d'arbitrage")
        ax[0].set_title("Convergence avec le budget (c = %.2f)" % C_REF)
        ax[0].legend(fontsize=8)

        for name, (m, s) in curves_c.items():
            ax[1].errorbar(C_VALUES, m, yerr=s, marker="o", capsize=3, label="MCTS " + name)
        ax[1].axhline(100, color="green", ls="--", lw=1)
        ax[1].set_xlabel("Constante d'exploration c")
        ax[1].set_title("Sensibilité à c (budget = %d)" % BUDGET_REF)
        ax[1].legend(fontsize=8)

        fig.tight_layout()
        FIGDIR.mkdir(exist_ok=True)
        out = FIGDIR / "sensibilite.png"
        fig.savefig(out, dpi=120)
        print("Figure enregistrée : %s" % out)
    except Exception as exc:
        print("(Figure non générée : %s)" % exc)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[0]) if args else 5)
