"""Demonstration : compare heuristiques, optimum (PD) et MCTS sur donnees
synthetiques. A remplacer par les vraies series EPEX SPOT France ensuite.

Metrique cle : le GAIN D'ARBITRAGE = part de la valeur du stockage captee,
  gain = (profit - profit_tout_vendre) / (optimum - profit_tout_vendre)
"Tout vendre" (n'utilise pas le stockage) vaut 0 % ; l'optimum (PD) vaut 100 %.
Cette metrique isole ce que la strategie apporte VRAIMENT via le stockage, au
lieu d'etre noyee dans le revenu de base de la vente de production.

Lancement :  python run_demo.py
"""

import numpy as np

from environment import EnergyStorageEnv, make_synthetic_data
from baselines import (policy_always_sell, policy_threshold,
                       policy_greedy_myopic, policy_random, dp_optimal)
from mcts import MCTSPlanner


def main():
    # 1) Donnees + environnement -------------------------------------------------
    prices, production = make_synthetic_data(days=2, seed=1)
    env = EnergyStorageEnv(prices, production,
                           capacity=120.0, eta_charge=0.95, eta_discharge=0.95,
                           max_charge=30.0, max_discharge=30.0,
                           soc0=0.0, n_actions=11)

    results, socs = {}, {}

    # 2) Heuristiques (phase 4) --------------------------------------------------
    threshold = policy_threshold(env)
    for name, pol in [("Aleatoire", policy_random(env, seed=0)),
                      ("Tout vendre", policy_always_sell(env)),
                      ("Glouton myope", policy_greedy_myopic(env)),
                      ("Seuil (mediane)", threshold)]:
        profit, soc_traj, _ = env.rollout_policy(pol)
        results[name] = profit
        socs[name] = soc_traj

    # 3) Optimum par programmation dynamique (phase 4) ---------------------------
    dp_policy, v0 = dp_optimal(env, n_soc=241)
    results["Optimum (PD)"], socs["Optimum (PD)"], _ = env.rollout_policy(dp_policy)

    # 4) MCTS (phase 5) : rollout aleatoire vs rollout informe -------------------
    p, socs["MCTS rollout aleatoire"], _ = MCTSPlanner(
        env, n_simulations=1000, c=1.0, rollout_policy=None, seed=42).run()
    results["MCTS rollout aleatoire"] = p

    p, socs["MCTS rollout seuil"], _ = MCTSPlanner(
        env, n_simulations=1000, c=1.0, rollout_policy=threshold, seed=42).run()
    results["MCTS rollout seuil"] = p

    # 5) Tableau comparatif ------------------------------------------------------
    base = results["Tout vendre"]
    opt = results["Optimum (PD)"]
    storage_value = opt - base

    order = ["Aleatoire", "Tout vendre", "Glouton myope", "Seuil (mediane)",
             "MCTS rollout aleatoire", "MCTS rollout seuil", "Optimum (PD)"]
    print("\nValeur optimale theorique V0 (PD) : %.1f" % v0)
    print("Valeur de l'arbitrage de stockage  : %.1f" % storage_value)
    print("=" * 60)
    print("%-26s %10s %8s %12s" % ("Strategie", "Profit", "%opt", "%arbitrage"))
    print("-" * 60)
    for name in order:
        prof = results[name]
        gain = 100.0 * (prof - base) / storage_value
        print("%-26s %10.1f %7.1f%% %11.1f%%" % (name, prof, 100.0 * prof / opt, gain))
    print("=" * 60)

    # 6) Figure des trajectoires -------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        H = env.H
        hours = np.arange(H)
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax[0].plot(hours, prices, color="tab:red", label="Prix spot")
        ax0b = ax[0].twinx()
        ax0b.fill_between(hours, production, color="tab:orange", alpha=0.25)
        ax[0].set_ylabel("Prix"); ax0b.set_ylabel("Production")
        ax[0].set_title("Donnees synthetiques (prix + production)")
        ax[0].legend(loc="upper left")

        for name in ["Seuil (mediane)", "MCTS rollout seuil", "Optimum (PD)"]:
            ax[1].step(np.arange(H + 1), socs[name], where="post", label=name)
        ax[1].set_ylabel("Stockage (SoC)"); ax[1].set_xlabel("Heure")
        ax[1].set_title("Trajectoires de stockage")
        ax[1].legend(loc="upper left")

        fig.tight_layout()
        fig.savefig("comparaison.png", dpi=120)
        print("\nFigure enregistree : comparaison.png")
    except Exception as exc:
        print("\n(Figure non generee : %s)" % exc)


if __name__ == "__main__":
    main()
