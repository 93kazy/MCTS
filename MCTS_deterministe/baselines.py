"""Phase 4 — Strategies de reference.

Deux familles :
  1. Heuristiques simples (tout vendre, seuil de prix, glouton myope, aleatoire).
  2. Borne optimale par programmation dynamique (PD) sous prevision parfaite :
     c'est le PLAFOND theorique du profit en deterministe. Le MCTS sera evalue
     en pourcentage de cette borne.

Toutes les strategies renvoient une politique `(t, soc) -> action_index`, jouee
ensuite sur le meme environnement -> comparaison strictement equitable (meme
grille d'actions, meme dynamique).
"""

import numpy as np


# ---------------------------------------------------------------------- #
# Heuristiques                                                            #
# ---------------------------------------------------------------------- #
def policy_always_sell(env):
    """Tout vendre immediatement : n'utilise jamais le stockage (action u=0)."""
    idx_zero = int(np.argmin(np.abs(env.actions)))

    def policy(t, soc):
        return idx_zero
    return policy


def policy_threshold(env, threshold=None):
    """Seuil de prix : si prix >= seuil on vend au maximum (decharge max),
    sinon on stocke au maximum (charge max). Heuristique d'arbitrage classique.
    Seuil par defaut = mediane des prix observes."""
    if threshold is None:
        threshold = float(np.median(env.prices))
    idx_max_discharge = 0                       # u le plus negatif
    idx_max_charge = env.n_actions - 1          # u le plus positif

    def policy(t, soc):
        return idx_max_discharge if env.prices[t] >= threshold else idx_max_charge
    return policy


def policy_greedy_myopic(env):
    """Glouton myope : a chaque pas, action qui maximise la recompense immediate.
    Ignore la valeur future du stockage -> ne stocke jamais (vendre rapporte
    tout de suite, stocker rapporte 0 sur l'instant). Sert a illustrer le cout
    de la myopie."""
    def policy(t, soc):
        rewards = [env.apply(soc, t, a)[1] for a in range(env.n_actions)]
        return int(np.argmax(rewards))
    return policy


def policy_random(env, seed=0):
    """Politique aleatoire : plancher de reference."""
    rng = np.random.default_rng(seed)

    def policy(t, soc):
        return int(rng.integers(env.n_actions))
    return policy


# ---------------------------------------------------------------------- #
# Borne optimale : programmation dynamique (induction arriere)            #
# ---------------------------------------------------------------------- #
def dp_optimal(env, n_soc=201):
    """Resout l'optimum exact du MDP discretise par induction arriere.

    L'etat continu (soc) est discretise sur une grille ; la valeur V(t, .) est
    interpolee lineairement en soc. Renvoie :
      - une politique greedy-vs-V jouable sur l'environnement,
      - V0 = valeur optimale theorique depuis (t=0, soc0).

    Complexite : O(H * n_soc * n_actions). Donne la borne superieure servant de
    reference au MCTS. (Alternative possible : formulation en programmation
    lineaire, qui donne l'optimum continu ; la PD est ici auto-suffisante et
    epouse exactement le MDP que le MCTS resout.)
    """
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
                v_next = np.interp(soc_next, soc_grid, V[t + 1])
                val = r + v_next
                if val > best_val:
                    best_val = val
                    best_a = a
            V[t, i] = best_val
            policy_grid[t, i] = best_a

    def policy(t, soc):
        i = int(np.argmin(np.abs(soc_grid - soc)))   # plus proche point de grille
        return int(policy_grid[t, i])

    v0 = float(np.interp(env.soc0, soc_grid, V[0]))
    return policy, v0
