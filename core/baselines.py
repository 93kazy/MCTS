"""Stratégies de référence : quelques heuristiques simples et l'optimum exact
par programmation dynamique (qui sert de plafond pour juger le MCTS)."""

import numpy as np

from .environment import A_SELL, A_STORE


def policy_always_sell(env):
    """Tout vendre, ne jamais utiliser la batterie."""
    if env.action_mode == "simple3":
        idx = A_SELL
    else:
        idx = int(np.argmin(np.abs(env.actions)))

    def policy(t, soc):
        return idx
    return policy


def policy_threshold(env, threshold=None):
    """Vendre au max si le prix dépasse le seuil (mediane par defaut),
    sinon stocker au max. L'arbitrage classique buy low / sell high."""
    if threshold is None:
        threshold = float(np.median(env.prices))
    if env.action_mode == "simple3":
        idx_sell, idx_store = A_SELL, A_STORE
    else:
        idx_sell = 0
        idx_store = env.n_actions - 1

    def policy(t, soc):
        return idx_sell if env.prices[t] >= threshold else idx_store
    return policy


def policy_greedy_myopic(env):
    """A chaque pas, l'action qui rapporte le plus tout de suite.
    Ne stocke jamais : montre le cout de la myopie."""
    def policy(t, soc):
        rewards = [env.apply(soc, t, a)[1] for a in range(env.n_actions)]
        return int(np.argmax(rewards))
    return policy


def policy_random(env, seed=0):
    rng = np.random.default_rng(seed)

    def policy(t, soc):
        return int(rng.integers(env.n_actions))
    return policy


def dp_optimal(env, n_soc=201):
    """Optimum exact du MDP par induction arriere sur une grille de soc.
    Renvoie une politique jouable et la valeur optimale V0 depuis (0, soc0).
    Complexite O(H * n_soc * n_actions)."""
    H = env.H
    soc_grid = np.linspace(0.0, env.capacity, n_soc)
    V = np.zeros((H + 1, n_soc))
    policy_grid = np.zeros((H, n_soc), dtype=int)

    for t in range(H - 1, -1, -1):
        for i, soc in enumerate(soc_grid):
            best_val = -np.inf
            best_a = 0
            for a in range(env.n_actions):
                soc_next, r = env.apply(soc, t, a)
                val = r + np.interp(soc_next, soc_grid, V[t + 1])
                if val > best_val:
                    best_val = val
                    best_a = a
            V[t, i] = best_val
            policy_grid[t, i] = best_a

    def policy(t, soc):
        i = int(np.argmin(np.abs(soc_grid - soc)))
        return int(policy_grid[t, i])

    v0 = float(np.interp(env.soc0, soc_grid, V[0]))
    return policy, v0
