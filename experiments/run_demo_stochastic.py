"""Evaluation Monte Carlo du cas stochastique.

On tire beaucoup de chemins de prix dans le modèle de Markov. Chaque politique
est jouée en boucle fermée (elle ne voit pas le futur) sur les mêmes chemins,
le clairvoyant, lui, voit le chemin realise et donne la borne haute. On regarde
si le MCTS (qui n'utilise qu'un modèle génératif) rejoint l'optimum causal (SDP).

Lancement :  python experiments/run_demo_stochastic.py [n_eval] [n_sim_mcts]
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIGDIR = ROOT / "figures"

from core.environment import EnergyStorageEnv
from core.price_model import MarkovPriceModel, make_seasonal_profiles
from core.stochastic import solve_sdp, make_ce_mpc_policy, clairvoyant_value
from core.mcts_stochastic import StochasticMCTSPlanner, stochastic_threshold_policy


def rollout_causal(env, model, policy, state_path, return_soc=False):
    soc = env.soc0
    total = 0.0
    soc_traj = [soc]
    for t in range(env.H):
        s = int(state_path[t])
        a = policy(t, soc, s)
        soc_n, out, forced = env.transition(soc, t, a)
        total += env.revenue(out, forced, t, model.price(s, t))
        soc = soc_n
        soc_traj.append(soc)
    return (total, np.array(soc_traj)) if return_soc else total


def main(n_eval=40, n_sim_mcts=800):
    t_start = time.time()

    mu, production = make_seasonal_profiles(days=1)
    model = MarkovPriceModel(mu, rho=0.7, sigma=6.0, spike_prob=0.05)
    env = EnergyStorageEnv(mu, production, capacity=120.0,
                           eta_charge=0.95, eta_discharge=0.95,
                           max_charge=30.0, max_discharge=30.0,
                           soc0=0.0, n_actions=9)

    threshold = float(np.median(mu))
    idx_sell = int(np.argmin(np.abs(env.actions)))

    sdp_policy, v0_sdp = solve_sdp(env, model, n_soc=81)
    ce_policy = make_ce_mpc_policy(env, model, n_soc=41)
    thr_policy = stochastic_threshold_policy(env, model, threshold)

    def always_sell(t, soc, s):
        return idx_sell

    mcts_rand = StochasticMCTSPlanner(env, model, n_simulations=n_sim_mcts,
                                      c=1.0, rollout_policy=None, seed=1)
    mcts_heur = StochasticMCTSPlanner(env, model, n_simulations=n_sim_mcts,
                                      c=1.0, rollout_policy=thr_policy, seed=1)

    causal = {"Tout vendre": always_sell,
              "Seuil (mediane)": thr_policy,
              "Equiv.-certain (MPC)": ce_policy,
              "SDP (optimum causal)": sdp_policy}

    totals = {name: [] for name in causal}
    totals["MCTS rollout aleatoire"] = []
    totals["MCTS rollout seuil"] = []
    clair = []

    rng = np.random.default_rng(2024)
    paths = [model.sample_path(rng) for _ in range(n_eval)]
    for sp in paths:
        realized = model.price_array(sp)
        clair.append(clairvoyant_value(env, realized, n_soc=81))
        for name, pol in causal.items():
            totals[name].append(rollout_causal(env, model, pol, sp))
        totals["MCTS rollout aleatoire"].append(mcts_rand.run_on_path(sp))
        totals["MCTS rollout seuil"].append(mcts_heur.run_on_path(sp))

    mean = {n: float(np.mean(v)) for n, v in totals.items()}
    std = {n: float(np.std(v)) for n, v in totals.items()}
    mean_clair = float(np.mean(clair))
    base = mean["Tout vendre"]
    opt = mean["SDP (optimum causal)"]

    order = ["Tout vendre", "Seuil (mediane)", "Equiv.-certain (MPC)",
             "MCTS rollout aleatoire", "MCTS rollout seuil", "SDP (optimum causal)"]

    print("\n%d scenarios | MCTS : %d simulations/pas | %.0fs"
          % (n_eval, n_sim_mcts, time.time() - t_start))
    print("Borne clairvoyante (prevision parfaite) : %.1f" % mean_clair)
    print("=" * 72)
    print("%-24s %10s %8s %9s %11s"
          % ("Strategie", "Profit", "ecart", "%clairv.", "%arbitrage"))
    print("-" * 72)
    for name in order:
        m = mean[name]
        pa = 100.0 * (m - base) / (opt - base) if opt > base else 0.0
        print("%-24s %10.1f %8.1f %8.1f%% %10.1f%%"
              % (name, m, std[name], 100.0 * m / mean_clair, pa))
    print("=" * 72)
    print("Cout de l'incertitude (clairvoyant - SDP) : %.1f  (%.1f%%)"
          % (mean_clair - opt, 100.0 * (mean_clair - opt) / mean_clair))

    # figure : le scenario avec le plus gros pic + les performances
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k = int(np.argmax([model.price_array(sp).max() for sp in paths]))
        sp = paths[k]
        realized = model.price_array(sp)
        H = env.H
        _, soc_sdp = rollout_causal(env, model, sdp_policy, sp, return_soc=True)
        _, soc_ce = rollout_causal(env, model, ce_policy, sp, return_soc=True)

        names = ["Tout vendre", "Seuil (mediane)", "Equiv.-certain (MPC)",
                 "MCTS rollout seuil", "SDP (optimum causal)"]
        arb = [100.0 * (mean[n] - base) / (opt - base) for n in names]

        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].bar(range(len(names)), arb, color="tab:blue")
        ax[0].axhline(100, color="green", ls="--", label="SDP (optimum)")
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax[0].set_ylabel("% de la valeur d'arbitrage (SDP=100)")
        ax[0].set_title("Performance moyenne (%d scenarios)" % n_eval)
        ax[0].legend(fontsize=8)

        ax2 = ax[1]
        ax2.plot(range(H), realized, color="tab:red", label="Prix realise")
        ax2.set_ylabel("Prix"); ax2.set_xlabel("Heure")
        ax2b = ax2.twinx()
        ax2b.step(range(H + 1), soc_sdp, where="post", color="tab:green", label="SoC SDP")
        ax2b.step(range(H + 1), soc_ce, where="post", color="tab:orange",
                  ls="--", label="SoC Equiv.-certain")
        ax2b.set_ylabel("Stockage (SoC)")
        ax2.set_title("Un scenario avec pic de prix")
        ax2.legend(loc="upper left", fontsize=8)
        ax2b.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        FIGDIR.mkdir(exist_ok=True)
        out = FIGDIR / "comparaison_stochastique.png"
        fig.savefig(out, dpi=120)
        print("Figure enregistree : %s" % out)
    except Exception as exc:
        print("(Figure non generee : %s)" % exc)


if __name__ == "__main__":
    args = sys.argv[1:]
    n_eval = int(args[0]) if len(args) > 0 else 40
    n_sim = int(args[1]) if len(args) > 1 else 800
    main(n_eval, n_sim)
