"""Modele de prix stochastique (version stochastique du projet).

Le prix se decompose en :
  - une moyenne saisonniere DETERMINISTE mu[t] (le motif journalier connu),
  - une deviation STOCHASTIQUE e[t] a retour a la moyenne, discretisee en une
    chaine de Markov a S etats, avec une queue haute (pics de prix occasionnels).

prix[t] = max(mu[t] + e[t], plancher).

Ce modele est volontairement markovien : il admet donc un optimum causal exact
calculable par programmation dynamique stochastique (SDP), qui sert de reference
au MCTS. Mais il fournit aussi un MODELE GENERATIF (echantillonnage de scenarios)
dont le MCTS seul a besoin — c'est l'avantage pratique du MCTS : il se passe de
la matrice de transition explicite et passe a l'echelle quand l'etat de prix
devient riche (plusieurs variables, memoire longue), la ou le SDP explose.
"""

import numpy as np


def make_seasonal_profiles(days=1):
    """Profils deterministes : moyenne de prix mu[t] et production[t]."""
    H = 24 * days
    hod = np.arange(H) % 24
    mu = (50.0
          + 30.0 * np.exp(-((hod - 19) ** 2) / 8.0)
          + 20.0 * np.exp(-((hod - 8) ** 2) / 6.0)
          - 15.0 * np.exp(-((hod - 13) ** 2) / 10.0))
    prod = 80.0 * np.exp(-((hod - 13) ** 2) / 12.0)
    prod = np.clip(prod, 0.0, None)
    return mu, prod


class MarkovPriceModel:
    """Chaine de Markov sur la deviation de prix (retour a la moyenne + pics)."""

    def __init__(self, mu, dev_states=None, rho=0.7, sigma=6.0,
                 spike_prob=0.05, price_floor=5.0):
        self.mu = np.asarray(mu, dtype=float)
        self.H = len(self.mu)
        if dev_states is None:
            # Grille asymetrique : queue haute = pics de prix.
            dev_states = np.array([-20, -12, -6, -2, 0, 2, 6, 12, 25, 45], float)
        self.dev = np.asarray(dev_states, dtype=float)
        self.S = len(self.dev)
        self.price_floor = float(price_floor)
        self.spike_idx = self.S - 1
        self.P = self._build_transition(rho, sigma, spike_prob)
        self._cdf = np.cumsum(self.P, axis=1)             # pour un tirage rapide
        self._cdf[:, -1] = 1.0                            # securite (arrondis)
        self.start_state = int(np.argmin(np.abs(self.dev)))   # demarre a e = 0

    def _build_transition(self, rho, sigma, spike_prob):
        """Matrice S x S : centree sur rho*e (retour a la moyenne) + masse de pic."""
        P = np.zeros((self.S, self.S))
        for i, e in enumerate(self.dev):
            target = rho * e
            w = np.exp(-((self.dev - target) ** 2) / (2.0 * sigma ** 2))
            w /= w.sum()
            w = (1.0 - spike_prob) * w
            w[self.spike_idx] += spike_prob       # proba de saut vers le pic
            w /= w.sum()
            P[i] = w
        return P

    # ---- prix ----
    def price(self, s, t):
        return max(self.mu[t] + self.dev[s], self.price_floor)

    def price_array(self, state_path):
        return np.array([self.price(s, t) for t, s in enumerate(state_path)])

    # ---- modele generatif ----
    def sample_next(self, s, rng):
        # Tirage rapide par transformation inverse (bien plus rapide que rng.choice).
        return int(np.searchsorted(self._cdf[s], rng.random()))

    def sample_path(self, rng):
        s = self.start_state
        path = [s]
        for _ in range(self.H - 1):
            s = self.sample_next(s, rng)
            path.append(s)
        return np.array(path)

    # ---- esperance (pour l'equivalent-certain) ----
    def expected_future_devs(self, s, n):
        """Esperance de la deviation pour les n prochains pas (offset 0 = pas
        courant, deviation connue = dev[s])."""
        dist = np.zeros(self.S)
        dist[s] = 1.0
        out = []
        for _ in range(n):
            out.append(float(dist @ self.dev))
            dist = dist @ self.P
        return np.array(out)
