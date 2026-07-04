"""Strategies de reference en stochastique.

Trois niveaux de comparaison :
  - CLAIRVOYANT (borne haute) : DP deterministe sur le chemin de prix REALISE.
    Voit le futur -> aucune politique causale ne peut faire mieux en esperance.
    L'ecart clairvoyant - optimum_causal mesure le "cout de l'incertitude".
  - SDP (optimum causal) : DP stochastique sur l'etat augmente (t, soc, etat_prix).
    Meilleure politique possible SANS voir le futur, etant donne le modele.
  - EQUIVALENT-CERTAIN (MPC) : a chaque pas, on re-optimise en deterministe sur
    la prevision MOYENNE des prix, on joue la 1re action, on recommence. Ignore
    l'incertitude -> sous-optimal des que la valeur est non lineaire en l'alea
    (contraintes de stockage, pics de prix).
"""

import numpy as np


# ---------------------------------------------------------------------- #
# DP deterministe parametre par une serie de prix arbitraire             #
# (sert pour le clairvoyant et pour l'equivalent-certain)                #
# ---------------------------------------------------------------------- #
def solve_det_dp(env, prices, t0=0, n_soc=61, soc_grid=None):
    H = env.H
    if soc_grid is None:
        soc_grid = np.linspace(0.0, env.capacity, n_soc)
    nS = len(soc_grid)
    V = np.zeros((H + 1, nS))
    pol = np.zeros((H, nS), dtype=int)
    for t in range(H - 1, t0 - 1, -1):
        for i, soc in enumerate(soc_grid):
            best, ba = -1e18, 0
            for a in range(env.n_actions):
                soc_n, sold = env.dispatch(soc, t, a)
                val = sold * prices[t] + np.interp(soc_n, soc_grid, V[t + 1])
                if val > best:
                    best, ba = val, a
            V[t, i] = best
            pol[t, i] = ba
    return V, pol, soc_grid


def clairvoyant_value(env, realized_prices, n_soc=61):
    """Borne haute pour un chemin de prix donne (prevision parfaite)."""
    V, _, soc_grid = solve_det_dp(env, realized_prices, t0=0, n_soc=n_soc)
    return float(np.interp(env.soc0, soc_grid, V[0]))


# ---------------------------------------------------------------------- #
# Equivalent-certain (MPC sur prevision moyenne)                         #
# ---------------------------------------------------------------------- #
def make_ce_mpc_policy(env, model, n_soc=51):
    """Renvoie une politique causale (t, soc, s) -> action.

    A chaque appel : construit la prevision moyenne des prix de t a H-1, resout
    le DP deterministe sur cette prevision, renvoie l'action optimale en t.
    """
    soc_grid = np.linspace(0.0, env.capacity, n_soc)

    def policy(t, soc, s):
        exp_dev = model.expected_future_devs(s, env.H - t)   # offsets 0..H-1-t
        forecast = np.array(env.prices, dtype=float)         # gabarit, recopie
        for k in range(env.H - t):
            forecast[t + k] = max(model.mu[t + k] + exp_dev[k], model.price_floor)
        _, pol, _ = solve_det_dp(env, forecast, t0=t, n_soc=n_soc, soc_grid=soc_grid)
        i = int(np.argmin(np.abs(soc_grid - soc)))
        return int(pol[t, i])

    return policy


# ---------------------------------------------------------------------- #
# Programmation dynamique stochastique (optimum causal)                  #
# ---------------------------------------------------------------------- #
def _interp_row(x, soc_grid, Vnext):
    """V(x, :) par interpolation lineaire en soc. Vnext de forme (nS, S)."""
    if x <= soc_grid[0]:
        return Vnext[0]
    if x >= soc_grid[-1]:
        return Vnext[-1]
    j = int(np.searchsorted(soc_grid, x)) - 1
    frac = (x - soc_grid[j]) / (soc_grid[j + 1] - soc_grid[j])
    return Vnext[j] * (1.0 - frac) + Vnext[j + 1] * frac


def solve_sdp(env, model, n_soc=61):
    """Resout l'optimum causal exact du MDP augmente (t, soc, etat_prix).

      V(t, soc, s) = max_a { sold(soc,a)*prix(s,t) + E_{s'~P[s]} V(t+1, soc', s') }

    Renvoie une politique causale (t, soc, s) -> action et la valeur V0.
    Complexite : O(H * n_soc * n_actions * S) (+ esperance S). La separation
    prix/dynamique (soc' independant de s) permet de vectoriser l'esperance.
    """
    H, S = env.H, model.S
    soc_grid = np.linspace(0.0, env.capacity, n_soc)
    nS = len(soc_grid)
    P = model.P
    V = np.zeros((H + 1, nS, S))
    pol = np.zeros((H, nS, S), dtype=int)

    for t in range(H - 1, -1, -1):
        price_s = np.array([model.price(s, t) for s in range(S)])   # (S,)
        Vnext = V[t + 1]                                            # (nS, S)
        for i, soc in enumerate(soc_grid):
            # soc'/sold ne dependent pas de s -> calcul une fois par (i, a)
            q = np.full((env.n_actions, S), -1e18)
            for a in range(env.n_actions):
                soc_n, sold = env.dispatch(soc, t, a)
                ev = P @ _interp_row(soc_n, soc_grid, Vnext)        # (S,) esperance
                q[a] = sold * price_s + ev                          # (S,)
            best_a = np.argmax(q, axis=0)                           # (S,)
            V[t, i] = q[best_a, np.arange(S)]
            pol[t, i] = best_a

    def policy(t, soc, s):
        i = int(np.argmin(np.abs(soc_grid - soc)))
        return int(pol[t, i, s])

    v0 = float(np.interp(env.soc0, soc_grid, V[0, :, model.start_state]))
    return policy, v0
