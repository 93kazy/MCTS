"""Stratégies de référence en stochastique :
  - clairvoyant   : DP sur le chemin de prix réel (voit le futur, borne haute) ;
  - SDP           : optimum causal exact (DP sur (t, soc, etat_prix)) ;
  - equiv.-certain: à chaque pas, on re-optimise en deterministe sur la prévision
                    moyenne des prix (MPC) et on joue la 1re action.
"""

import numpy as np


def solve_det_dp(env, prices, t0=0, n_soc=61, soc_grid=None):
    """DP déterministe pour une série de prix donnée."""
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
                soc_n, out, forced = env.transition(soc, t, a)
                val = (env.revenue(out, forced, t, prices[t])
                       + np.interp(soc_n, soc_grid, V[t + 1]))
                if val > best:
                    best, ba = val, a
            V[t, i] = best
            pol[t, i] = ba
    return V, pol, soc_grid


def clairvoyant_value(env, realized_prices, n_soc=61):
    V, _, soc_grid = solve_det_dp(env, realized_prices, t0=0, n_soc=n_soc)
    return float(np.interp(env.soc0, soc_grid, V[0]))


def make_ce_mpc_policy(env, model, n_soc=51):
    """Politique equivalent-certain : à chaque pas, DP sur la prévision moyenne
    de t à H-1, on renvoie l'action optimale en t."""
    soc_grid = np.linspace(0.0, env.capacity, n_soc)

    def policy(t, soc, s):
        exp_dev = model.expected_future_devs(s, env.H - t)
        forecast = np.array(env.prices, dtype=float)
        for k in range(env.H - t):
            forecast[t + k] = max(model.mu[t + k] + exp_dev[k], model.price_floor)
        _, pol, _ = solve_det_dp(env, forecast, t0=t, n_soc=n_soc, soc_grid=soc_grid)
        i = int(np.argmin(np.abs(soc_grid - soc)))
        return int(pol[t, i])

    return policy


def _interp_row(x, soc_grid, Vnext):
    """Interpolation linéaire en soc de Vnext (forme (nS, S))."""
    if x <= soc_grid[0]:
        return Vnext[0]
    if x >= soc_grid[-1]:
        return Vnext[-1]
    j = int(np.searchsorted(soc_grid, x)) - 1
    frac = (x - soc_grid[j]) / (soc_grid[j + 1] - soc_grid[j])
    return Vnext[j] * (1.0 - frac) + Vnext[j + 1] * frac


def solve_sdp(env, model, n_soc=61):
    """Optimum causal exact par DP stochastique sur l'etat (t, soc, s) :
        V(t, soc, s) = max_a { revenu + E_{s'~P[s]} V(t+1, soc', s') }.
    Complexité O(H * n_soc * n_actions * S). Comme soc' ne dépend pas de s, on
    calcule la transition une fois par (soc, a) et on valorise sur tous les s."""
    H, S = env.H, model.S
    soc_grid = np.linspace(0.0, env.capacity, n_soc)
    nS = len(soc_grid)
    P = model.P
    V = np.zeros((H + 1, nS, S))
    pol = np.zeros((H, nS, S), dtype=int)

    for t in range(H - 1, -1, -1):
        price_s = np.array([model.price(s, t) for s in range(S)])
        Vnext = V[t + 1]
        for i, soc in enumerate(soc_grid):
            q = np.full((env.n_actions, S), -1e18)
            for a in range(env.n_actions):
                soc_n, out, forced = env.transition(soc, t, a)
                ev = P @ _interp_row(soc_n, soc_grid, Vnext)
                rev = np.array([env.revenue(out, forced, t, p) for p in price_s])
                q[a] = rev + ev
            best_a = np.argmax(q, axis=0)
            V[t, i] = q[best_a, np.arange(S)]
            pol[t, i] = best_a

    def policy(t, soc, s):
        i = int(np.argmin(np.abs(soc_grid - soc)))
        return int(pol[t, i, s])

    v0 = float(np.interp(env.soc0, soc_grid, V[0, :, model.start_state]))
    return policy, v0
