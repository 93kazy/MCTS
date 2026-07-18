"""Modele de prix stochastique.

prix[t] = max(mu[t] + e[t], plancher), ou mu[t] est le motif journalier connu
et e[t] une deviation aleatoire qui revient vers 0 (chaine de Markov, avec de
temps en temps un pic). Le modele donne a la fois la matrice de transition
(pour le SDP) et de quoi echantillonner des scenarios (pour le MCTS)."""

import numpy as np


def make_seasonal_profiles(days=1):
    """Moyenne de prix mu[t] et production[t] (partie deterministe)."""
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

    def __init__(self, mu, dev_states=None, rho=0.7, sigma=6.0,
                 spike_prob=0.05, price_floor=5.0):
        self.mu = np.asarray(mu, dtype=float)
        self.H = len(self.mu)
        if dev_states is None:
            # grille asymetrique : la queue haute represente les pics de prix
            dev_states = np.array([-20, -12, -6, -2, 0, 2, 6, 12, 25, 45], float)
        self.dev = np.asarray(dev_states, dtype=float)
        self.S = len(self.dev)
        self.price_floor = float(price_floor)
        self.spike_idx = self.S - 1
        self.P = self._build_transition(rho, sigma, spike_prob)
        self._cdf = np.cumsum(self.P, axis=1)
        self._cdf[:, -1] = 1.0
        self.start_state = int(np.argmin(np.abs(self.dev)))

    def _build_transition(self, rho, sigma, spike_prob):
        # gaussienne centree sur rho*e (retour a la moyenne) + un peu de masse
        # sur l'etat de pic
        P = np.zeros((self.S, self.S))
        for i, e in enumerate(self.dev):
            target = rho * e
            w = np.exp(-((self.dev - target) ** 2) / (2.0 * sigma ** 2))
            w /= w.sum()
            w = (1.0 - spike_prob) * w
            w[self.spike_idx] += spike_prob
            w /= w.sum()
            P[i] = w
        return P

    def price(self, s, t):
        return max(self.mu[t] + self.dev[s], self.price_floor)

    def price_array(self, state_path):
        return np.array([self.price(s, t) for t, s in enumerate(state_path)])

    def sample_next(self, s, rng):
        # tirage par inversion de la cdf (plus rapide que rng.choice)
        return int(np.searchsorted(self._cdf[s], rng.random()))

    def sample_path(self, rng):
        s = self.start_state
        path = [s]
        for _ in range(self.H - 1):
            s = self.sample_next(s, rng)
            path.append(s)
        return np.array(path)

    def expected_future_devs(self, s, n):
        """Esperance de la deviation sur les n prochains pas (offset 0 = pas
        courant). Sert a l'equivalent-certain."""
        dist = np.zeros(self.S)
        dist[s] = 1.0
        out = []
        for _ in range(n):
            out.append(float(dist @ self.dev))
            dist = dist @ self.P
        return np.array(out)
