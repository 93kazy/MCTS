"""Demonstration sur VRAIES donnees France (prix day-ahead + solaire).

Cadre deterministe a prevision parfaite (la semaine reelle est connue) : la PD
donne l'optimum exact, contre lequel on mesure heuristiques et MCTS. C'est le
pendant de run_demo.py, mais sur donnees reelles.

Lancement :  python run_demo_real.py [debut] [fin]   (defaut : une semaine 2024)
"""

import sys
import numpy as np

from data_loader import build_real_env
from baselines import (policy_always_sell, policy_threshold,
                       policy_greedy_myopic, dp_optimal)
from mcts import MCTSPlanner


def main(start="2024-06-10", end="2024-06-16"):
    env, ts, prices, production = build_real_env(
        start, end, capacity_frac=0.6, power_frac=0.3, n_actions=11)
    H = env.H
    print("Donnees France %s -> %s : %d heures" % (start, end, H))
    print("Prix €/MWh : min %.1f  moy %.1f  max %.1f  (%d h negatives)"
          % (prices.min(), prices.mean(), prices.max(), int((prices < 0).sum())))
    print("Solaire MW : pic %.0f | capacite stockage %.0f MWh\n"
          % (production.max(), env.capacity))

    results, socs = {}, {}
    threshold = policy_threshold(env)
    for name, pol in [("Tout vendre", policy_always_sell(env)),
                      ("Glouton myope", policy_greedy_myopic(env)),
                      ("Seuil (mediane)", threshold)]:
        results[name], socs[name], _ = env.rollout_policy(pol)

    dp_policy, v0 = dp_optimal(env, n_soc=241)
    results["Optimum (PD)"], socs["Optimum (PD)"], _ = env.rollout_policy(dp_policy)

    p, socs["MCTS rollout seuil"], _ = MCTSPlanner(
        env, n_simulations=300, c=1.0, rollout_policy=threshold, seed=1).run()
    results["MCTS rollout seuil"] = p

    base = results["Tout vendre"]
    opt = results["Optimum (PD)"]
    storage_value = opt - base

    print("=" * 60)
    print("%-22s %12s %8s %11s" % ("Strategie", "Profit €", "%opt", "%arbitrage"))
    print("-" * 60)
    for name in ["Tout vendre", "Glouton myope", "Seuil (mediane)",
                 "MCTS rollout seuil", "Optimum (PD)"]:
        pr = results[name]
        gain = 100.0 * (pr - base) / storage_value if storage_value else 0.0
        print("%-22s %12.0f %7.1f%% %10.1f%%"
              % (name, pr, 100.0 * pr / opt if opt else 0, gain))
    print("=" * 60)
    print("Valeur de l'arbitrage de stockage : %.0f €" % storage_value)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hours = np.arange(H)
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax[0].plot(hours, prices, color="tab:red", label="Prix day-ahead")
        ax[0].axhline(0, color="grey", lw=0.8)
        ax[0].fill_between(hours, prices, 0, where=(prices < 0),
                           color="tab:red", alpha=0.25, label="Prix negatif")
        ax0b = ax[0].twinx()
        ax0b.fill_between(hours, production, color="tab:orange", alpha=0.2)
        ax0b.set_ylabel("Solaire (MW)")
        ax[0].set_ylabel("Prix €/MWh")
        ax[0].set_title("France %s -> %s (donnees reelles)" % (start, end))
        ax[0].legend(loc="upper left", fontsize=8)
        for name in ["Seuil (mediane)", "MCTS rollout seuil", "Optimum (PD)"]:
            ax[1].step(np.arange(H + 1), socs[name], where="post", label=name)
        ax[1].set_ylabel("Stockage (MWh)"); ax[1].set_xlabel("Heure (UTC)")
        ax[1].set_title("Trajectoires de stockage")
        ax[1].legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig("comparaison_reelle.png", dpi=120)
        print("Figure enregistree : comparaison_reelle.png")
    except Exception as exc:
        print("(Figure non generee : %s)" % exc)


if __name__ == "__main__":
    args = sys.argv[1:]
    s = args[0] if len(args) > 0 else "2024-06-10"
    e = args[1] if len(args) > 1 else "2024-06-16"
    main(s, e)
