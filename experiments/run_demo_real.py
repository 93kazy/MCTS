"""Meme comparaison que run_demo.py mais sur de vraies donnees France (prix
day-ahead + solaire). La semaine est connue, donc la PD reste l'optimum exact.
Une demande interne (15 % du pic solaire, prix d'evitement 90 €/MWh) active la
consommation : aux heures a prix negatif, consommer vaut mieux que vendre.

Lancement :  python experiments/run_demo_real.py [debut] [fin] [n_seeds]
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIGDIR = ROOT / "figures"

from core.data_loader import build_real_env
from core.baselines import (policy_always_sell, policy_threshold,
                            policy_greedy_myopic, policy_random, dp_optimal)
from core.mcts import MCTSPlanner


def main(start="2024-06-10", end="2024-06-16", n_seeds=3):
    env, ts, prices, production = build_real_env(
        start, end, capacity_frac=0.6, power_frac=0.3, n_actions=11,
        demand_frac=0.15, p_consume=90.0)
    H = env.H
    span_h = int(round((ts[-1] - ts[0]) / 3600)) + 1
    print("Donnees France %s -> %s : %d heures alignees (fenetre %d h ; "
          "%d h sans prix ou solaire commun)"
          % (start, end, H, span_h, span_h - H))
    print("Prix €/MWh : min %.1f  moy %.1f  max %.1f  (%d h negatives)"
          % (prices.min(), prices.mean(), prices.max(), int((prices < 0).sum())))
    print("Solaire MW : pic %.0f | stockage %.0f MWh | demande interne %.0f MW "
          "(evitement 90 €/MWh)"
          % (production.max(), env.capacity, env.demand[0]))
    print("Espace de recherche : %d^%d sequences\n" % (env.n_actions, H))

    results, spread, socs = {}, {}, {}
    threshold = policy_threshold(env)
    for name, pol in [("Aleatoire", policy_random(env, seed=0)),
                      ("Sans stockage (u=0)", policy_always_sell(env)),
                      ("Glouton myope", policy_greedy_myopic(env)),
                      ("Seuil (mediane)", threshold)]:
        results[name], socs[name], _ = env.rollout_policy(pol)

    dp_policy, v0 = dp_optimal(env, n_soc=241)
    results["Optimum (PD)"], socs["Optimum (PD)"], _ = env.rollout_policy(dp_policy)

    for label, rollout in [("MCTS rollout aleatoire", None),
                           ("MCTS rollout seuil", threshold)]:
        profs = []
        for sd in range(n_seeds):
            p, soc, _ = MCTSPlanner(env, n_simulations=300, c=1.0,
                                    rollout_policy=rollout, seed=sd).run()
            profs.append(p)
            if sd == 0:
                socs[label] = soc
        profs = np.array(profs)
        results[label] = float(profs.mean())
        spread[label] = float(profs.std())

    base = results["Sans stockage (u=0)"]
    opt = results["Optimum (PD)"]
    storage_value = opt - base

    print("=" * 68)
    print("%-22s %18s %8s %11s" % ("Strategie", "Profit €", "%opt", "%arbitrage"))
    print("-" * 68)
    for name in ["Aleatoire", "Sans stockage (u=0)", "Glouton myope",
                 "Seuil (mediane)", "MCTS rollout aleatoire",
                 "MCTS rollout seuil", "Optimum (PD)"]:
        pr = results[name]
        gain = 100.0 * (pr - base) / storage_value if storage_value else 0.0
        if name in spread:
            pr_str = "%10.0f +-%4.0f" % (pr, spread[name])
        else:
            pr_str = "%18.0f" % pr
        print("%-22s %18s %7.1f%% %10.1f%%"
              % (name, pr_str, 100.0 * pr / opt if opt else 0, gain))
    print("=" * 68)
    print("Valeur de l'arbitrage de stockage : %.0f €" % storage_value)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hours = np.arange(H)
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax[0].plot(hours, prices, color="tab:red", label="Prix day-ahead")
        ax[0].axhline(0, color="grey", lw=0.8)
        ax[0].axhline(90, color="tab:purple", lw=0.8, ls=":",
                      label="Prix d'evitement (conso)")
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
        FIGDIR.mkdir(exist_ok=True)
        out = FIGDIR / "comparaison_reelle.png"
        fig.savefig(out, dpi=120)
        print("Figure enregistree : %s" % out)
    except Exception as exc:
        print("(Figure non generee : %s)" % exc)


if __name__ == "__main__":
    args = sys.argv[1:]
    s = args[0] if len(args) > 0 else "2024-06-10"
    e = args[1] if len(args) > 1 else "2024-06-16"
    n = int(args[2]) if len(args) > 2 else 3
    main(s, e, n)
