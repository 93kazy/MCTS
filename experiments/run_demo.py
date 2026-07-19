"""Comparaison heuristiques / PD / MCTS sur données synthétiques.

Métrique : le gain d'arbitrage = (profit - sans_stockage) / (optimum - sans_stockage),
qui vaut 0 % sans stockage et 100 % a l'optimum. Le MCTS est moyenne sur plusieurs
graines. On teste aussi le mode "simple3" (les 3 actions du sujet).

Lancement :  python experiments/run_demo.py [n_seeds] [n_simulations]
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIGDIR = ROOT / "figures"

from core.environment import EnergyStorageEnv, make_synthetic_data
from core.baselines import (policy_always_sell, policy_threshold,
                            policy_greedy_myopic, policy_random, dp_optimal)
from core.mcts import MCTSPlanner


def eval_mcts(env, n_simulations, rollout, seeds, c=1.0):
    """MCTS sur plusieurs graines ; renvoie (profits, trajectoire de la 1re)."""
    profits, soc_first = [], None
    for sd in seeds:
        p, soc, _ = MCTSPlanner(env, n_simulations=n_simulations, c=c,
                                rollout_policy=rollout, seed=sd).run()
        profits.append(p)
        if soc_first is None:
            soc_first = soc
    return np.array(profits), soc_first


def main(n_seeds=5, n_sim=1000):
    seeds = list(range(n_seeds))

    # environnement en mode grid (11 debits de stockage)
    prices, production, demand = make_synthetic_data(days=2, seed=1)
    env = EnergyStorageEnv(prices, production, demand=demand, p_consume=70.0,
                           capacity=120.0, eta_charge=0.95, eta_discharge=0.95,
                           max_charge=30.0, max_discharge=30.0,
                           soc0=0.0, n_actions=11)
    print("Espace de recherche (mode grid)    : %d^%d sequences"
          % (env.n_actions, env.H))

    results, spread, socs = {}, {}, {}

    # heuristiques
    threshold = policy_threshold(env)
    for name, pol in [("Aleatoire", policy_random(env, seed=0)),
                      ("Sans stockage (u=0)", policy_always_sell(env)),
                      ("Glouton myope", policy_greedy_myopic(env)),
                      ("Seuil (mediane)", threshold)]:
        profit, soc_traj, _ = env.rollout_policy(pol)
        results[name] = profit
        socs[name] = soc_traj

    # optimum par programmation dynamique
    dp_policy, v0 = dp_optimal(env, n_soc=241)
    results["Optimum (PD)"], socs["Optimum (PD)"], _ = env.rollout_policy(dp_policy)

    # MCTS : rollout aleatoire vs rollout informe
    profs, socs["MCTS rollout aleatoire"] = eval_mcts(env, n_sim, None, seeds)
    results["MCTS rollout aleatoire"] = float(profs.mean())
    spread["MCTS rollout aleatoire"] = float(profs.std())

    profs, socs["MCTS rollout seuil"] = eval_mcts(env, n_sim, threshold, seeds)
    results["MCTS rollout seuil"] = float(profs.mean())
    spread["MCTS rollout seuil"] = float(profs.std())

    # tableau comparatif
    base = results["Sans stockage (u=0)"]
    opt = results["Optimum (PD)"]
    storage_value = opt - base

    order = ["Aleatoire", "Sans stockage (u=0)", "Glouton myope", "Seuil (mediane)",
             "MCTS rollout aleatoire", "MCTS rollout seuil", "Optimum (PD)"]
    print("\nValeur optimale theorique V0 (PD) : %.1f" % v0)
    print("Valeur de l'arbitrage de stockage  : %.1f" % storage_value)
    print("MCTS : %d simulations/pas, %d graines (moyenne +/- ecart-type)"
          % (n_sim, n_seeds))
    print("=" * 72)
    print("%-26s %16s %8s %12s" % ("Strategie", "Profit", "%opt", "%arbitrage"))
    print("-" * 72)
    for name in order:
        prof = results[name]
        gain = 100.0 * (prof - base) / storage_value
        if name in spread:
            prof_str = "%8.1f +-%5.1f" % (prof, spread[name])
        else:
            prof_str = "%16.1f" % prof
        print("%-26s %16s %7.1f%% %11.1f%%"
              % (name, prof_str, 100.0 * prof / opt, gain))
    print("=" * 72)

    # mode "simple3" : les 3 actions du sujet
    env3 = EnergyStorageEnv(prices, production, demand=demand, p_consume=70.0,
                            capacity=120.0, eta_charge=0.95, eta_discharge=0.95,
                            max_charge=30.0, max_discharge=30.0,
                            soc0=0.0, action_mode="simple3")
    print("\n--- Mode simple3 : VENDRE / STOCKER / CONSOMMER (arbre 3^%d) ---"
          % env3.H)
    dp3_policy, v0_3 = dp_optimal(env3, n_soc=241)
    opt3, _, _ = env3.rollout_policy(dp3_policy)
    base3, _, _ = env3.rollout_policy(policy_always_sell(env3))
    profs3, _ = eval_mcts(env3, n_sim, policy_threshold(env3), seeds)
    arb3 = 100.0 * (profs3.mean() - base3) / (opt3 - base3)
    print("Tout vendre : %.1f | Optimum (PD) : %.1f" % (base3, opt3))
    print("MCTS (rollout seuil) : %.1f +- %.1f  soit %.1f%% de l'arbitrage"
          % (profs3.mean(), profs3.std(), arb3))
    print("(Le mode grid ci-dessus, plus fin, atteint un optimum de %.1f : "
          "la grille de debits domine les 3 actions tout-ou-rien.)" % opt)

    # figure des trajectoires
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        H = env.H
        hours = np.arange(H)
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax[0].plot(hours, prices, color="tab:red", label="Prix spot")
        ax[0].axhline(70.0, color="tab:purple", lw=0.8, ls=":",
                      label="Prix d'evitement (conso)")
        ax0b = ax[0].twinx()
        ax0b.fill_between(hours, production, color="tab:orange", alpha=0.25)
        ax0b.plot(hours, demand, color="tab:green", lw=1.0, label="Demande interne")
        ax[0].set_ylabel("Prix"); ax0b.set_ylabel("Production / demande")
        ax[0].set_title("Donnees synthetiques (prix, production, demande)")
        ax[0].legend(loc="upper left", fontsize=8)
        ax0b.legend(loc="upper right", fontsize=8)

        for name in ["Seuil (mediane)", "MCTS rollout seuil", "Optimum (PD)"]:
            ax[1].step(np.arange(H + 1), socs[name], where="post", label=name)
        ax[1].set_ylabel("Stockage (SoC)"); ax[1].set_xlabel("Heure")
        ax[1].set_title("Trajectoires de stockage")
        ax[1].legend(loc="upper left")

        fig.tight_layout()
        FIGDIR.mkdir(exist_ok=True)
        out = FIGDIR / "comparaison.png"
        fig.savefig(out, dpi=120)
        print("\nFigure enregistree : %s" % out)
    except Exception as exc:
        print("\n(Figure non generee : %s)" % exc)


if __name__ == "__main__":
    args = sys.argv[1:]
    n_seeds = int(args[0]) if len(args) > 0 else 5
    n_sim = int(args[1]) if len(args) > 1 else 1000
    main(n_seeds, n_sim)
