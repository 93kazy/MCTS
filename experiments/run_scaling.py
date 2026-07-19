"""Ou le MCTS devient plus rapide que la PD.

On passe à B batteries de rendements différents (impossible de les fusionner) :
l'état devient le vecteur des B niveaux de charge. Le coût de la PD croit comme
n_soc^B, alors que le MCTS ne visite que les états
atteints et reste ~independant de B. On mesure les deux temps en fonction de B.

Lancement :  python experiments/run_scaling.py [dp_max_B] [n_sim_mcts]
"""

import itertools
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIGDIR = ROOT / "figures"

from core.mcts import MCTSPlanner


class MultiStorageEnv:
    """B batteries de rendements différents, production et prix communs.

    Actions (n_actions = 2B+1, croit lineairement en B) :
      0       : tout vendre ;
      2b+1    : charger la batterie b au max ;
      2b+2    : decharger la batterie b au max.
    Les rendements distincts empechent de fusionner les batteries : l'état garde
    ses B dimensions, ce qui fait exploser le coût de la PD.
    """

    def __init__(self, prices, production, B=1, capacity=60.0, seed=0):
        self.prices = np.asarray(prices, dtype=float)
        self.production = np.asarray(production, dtype=float)
        self.H = len(self.prices)
        self.B = B
        base = np.linspace(0.90, 0.98, B)
        self.eta_c = base.copy()
        self.eta_d = base[::-1].copy()
        self.capacity = np.full(B, float(capacity))
        self.max_rate = np.full(B, capacity * 0.4)
        self.soc0 = np.zeros(B)
        self.n_actions = 2 * B + 1

    def apply(self, soc, t, a):
        soc = np.asarray(soc, dtype=float).copy()
        prod = self.production[t]
        price = self.prices[t]

        if a == 0:
            sold = prod
        else:
            b = (a - 1) // 2
            if (a - 1) % 2 == 0:
                room = (self.capacity[b] - soc[b]) / self.eta_c[b]
                c = max(min(self.max_rate[b], prod, room), 0.0)
                soc[b] += self.eta_c[b] * c
                sold = prod - c
            else:
                d = max(min(self.max_rate[b], soc[b]), 0.0)
                soc[b] -= d
                sold = prod + self.eta_d[b] * d
        np.clip(soc, 0.0, self.capacity, out=soc)
        return soc, sold * price

    def informed_rollout(self):
        """Rollout : au-dessus du prix médian on décharge la batterie la plus
        pleine, en dessous on charge la plus vide."""
        median = float(np.median(self.prices))

        def pol(t, soc):
            soc = np.asarray(soc)
            if self.prices[t] >= median:
                b = int(np.argmax(soc))
                return 0 if soc[b] <= 1e-9 else 2 * b + 2
            else:
                b = int(np.argmin(soc / self.capacity))
                return 2 * b + 1
        return pol


def dp_multi(env, n_soc=21):
    """PD sur une grille n_soc par batterie (plus proche voisin).
    Coût O(H * n_soc^B * n_actions). Renvoie (V0, temps en secondes)."""
    B, H = env.B, env.H
    grids = [np.linspace(0.0, env.capacity[b], n_soc) for b in range(B)]
    shape = (n_soc,) * B

    def nearest_index(soc):
        idx = []
        for b in range(B):
            j = int(round(soc[b] / env.capacity[b] * (n_soc - 1)))
            idx.append(min(max(j, 0), n_soc - 1))
        return tuple(idx)

    t0 = time.perf_counter()
    V_next = np.zeros(shape)
    all_states = list(itertools.product(range(n_soc), repeat=B))
    for t in range(H - 1, -1, -1):
        V = np.empty(shape)
        for st in all_states:
            soc = np.array([grids[b][st[b]] for b in range(B)])
            best = -1e18
            for a in range(env.n_actions):
                soc_n, r = env.apply(soc, t, a)
                val = r + V_next[nearest_index(soc_n)]
                if val > best:
                    best = val
            V[st] = best
        V_next = V
    v0 = float(V_next[nearest_index(env.soc0)])
    return v0, time.perf_counter() - t0


def make_prices(H=24, seed=1):
    rng = np.random.default_rng(seed)
    hod = np.arange(H) % 24
    price = (50.0 + 30.0 * np.exp(-((hod - 19) ** 2) / 8.0)
             + 20.0 * np.exp(-((hod - 8) ** 2) / 6.0)
             - 15.0 * np.exp(-((hod - 13) ** 2) / 10.0))
    price += rng.normal(0.0, 3.0, H)
    prod = 80.0 * np.exp(-((hod - 13) ** 2) / 12.0) + 20.0
    return np.clip(price, 5.0, None), np.clip(prod, 0.0, None)


def main(dp_max_B=3, n_sim=400, n_soc=21, mcts_max_B=6, n_seeds=3):
    prices, production = make_prices(H=24)
    print("Materiel : %s | Python %s"
          % (platform.processor() or platform.machine(),
             platform.python_version()))
    print("H = %d | grille PD = %d points / batterie | MCTS = %d sim/pas, %d graines\n"
          % (len(prices), n_soc, n_sim, n_seeds))

    header = ("%-4s %10s %14s %14s %12s %10s"
              % ("B", "états PD", "temps PD (s)", "temps MCTS (s)",
                 "profit PD", "profit MCTS"))
    print(header)
    print("-" * len(header))

    rows = []
    for B in range(1, mcts_max_B + 1):
        env = MultiStorageEnv(prices, production, B=B)
        n_states = n_soc ** B

        if B <= dp_max_B:
            v0_dp, t_dp = dp_multi(env, n_soc=n_soc)
        else:
            v0_dp, t_dp = None, None

        rollout = env.informed_rollout()
        prof, t_mc = [], []
        for sd in range(n_seeds):
            t0 = time.perf_counter()
            p, _, _ = MCTSPlanner(env, n_simulations=n_sim, c=1.0,
                                  rollout_policy=rollout, seed=sd).run()
            t_mc.append(time.perf_counter() - t0)
            prof.append(p)
        v_mc, t_mcts = float(np.mean(prof)), float(np.mean(t_mc))

        dp_t_str = "%14.3f" % t_dp if t_dp is not None else "%14s" % "infaisable"
        dp_v_str = "%12.0f" % v0_dp if v0_dp is not None else "%12s" % "-"
        print("%-4d %10d %s %14.3f %s %10.0f"
              % (B, n_states, dp_t_str, t_mcts, dp_v_str, v_mc))
        rows.append((B, n_states, t_dp, t_mcts, v0_dp, v_mc))

    # figure : temps vs nombre de batteries (échelle log)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Bs = [r[0] for r in rows]
        dp_B = [r[0] for r in rows if r[2] is not None]
        dp_t = [r[2] for r in rows if r[2] is not None]
        mc_t = [r[3] for r in rows]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(dp_B, dp_t, "o-", color="tab:red", label="Programmation dynamique")
        ax.plot(Bs, mc_t, "s-", color="tab:blue",
                label="MCTS (%d sim/pas)" % n_sim)
        ax.set_yscale("log")
        ax.set_xlabel("Nombre de batteries B (dimension de l'état)")
        ax.set_ylabel("Temps de calcul par épisode (s, log)")
        ax.set_title("Coût PD vs MCTS quand l'état grandit")
        ax.set_xticks(Bs)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        FIGDIR.mkdir(exist_ok=True)
        out = FIGDIR / "scaling.png"
        fig.savefig(out, dpi=120)
        print("\nFigure enregistrée : %s" % out)
    except Exception as exc:
        print("\n(Figure non générée : %s)" % exc)


if __name__ == "__main__":
    args = sys.argv[1:]
    dp_max_B = int(args[0]) if len(args) > 0 else 3
    n_sim = int(args[1]) if len(args) > 1 else 400
    main(dp_max_B=dp_max_B, n_sim=n_sim)
