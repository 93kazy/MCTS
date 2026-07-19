"""Environnement (MDP) de la centrale avec stockage.

A chaque heure la centrale produit de l'énergie et peut la vendre au prix spot,
la stocker dans la batterie, ou la consommer en interne (valorisée au prix
d'evitement p_consume). Etat = (t, soc). Pas d'achat sur le marche : le stockage
sert juste a decaler dans le temps l'usage de sa propre production.

Deux modes d'action :
  - "simple3" : les 3 actions du sujet (vendre / stocker / consommer), arbre 3^H.
  - "grid"    : un debit de stockage u parmi n_actions valeurs. Le partage
                vente/consommation du reste est fait par le prix (on consomme si
                p_consume > prix), ce qui laisse le stockage comme seule vraie
                décision. Arbre n_actions^H, bien plus grand.
"""

import numpy as np

A_SELL, A_STORE, A_CONSUME = 0, 1, 2
SIMPLE3_NAMES = {A_SELL: "VENDRE", A_STORE: "STOCKER", A_CONSUME: "CONSOMMER"}


class EnergyStorageEnv:

    def __init__(self, prices, production, demand=None, p_consume=0.0,
                 capacity=100.0, eta_charge=0.95, eta_discharge=0.95,
                 max_charge=30.0, max_discharge=30.0, soc0=0.0,
                 n_actions=11, action_mode="grid"):
        self.prices = np.asarray(prices, dtype=float)
        self.production = np.asarray(production, dtype=float)
        assert len(self.prices) == len(self.production)
        self.H = len(self.prices)

        if demand is None:
            demand = np.zeros(self.H)
        self.demand = np.asarray(demand, dtype=float)
        assert len(self.demand) == self.H
        if np.isscalar(p_consume):
            self.p_consume = np.full(self.H, float(p_consume))
        else:
            self.p_consume = np.asarray(p_consume, dtype=float)
        assert len(self.p_consume) == self.H

        self.capacity = float(capacity)
        self.eta_charge = float(eta_charge)
        self.eta_discharge = float(eta_discharge)
        self.max_charge = float(max_charge)
        self.max_discharge = float(max_discharge)
        self.soc0 = float(soc0)

        self.action_mode = action_mode
        if action_mode == "grid":
            k = max(1, (n_actions - 1) // 2)
            charge = np.linspace(0.0, self.max_charge, k + 1)[1:]
            discharge = -np.linspace(0.0, self.max_discharge, k + 1)[1:]
            self.actions = np.concatenate([discharge[::-1], [0.0], charge])
            self.n_actions = len(self.actions)
        elif action_mode == "simple3":
            self.actions = None
            self.n_actions = 3
        else:
            raise ValueError("action_mode doit etre 'grid' ou 'simple3'")

        self._t = 0
        self._soc = self.soc0

    def transition(self, soc, t, action_index):
        """Dynamique sans le prix : renvoie (soc_suivant, énergie à ecouler,
        conso imposée). forced=None en mode grid (le partage vente/conso est
        differe a revenue), une valeur en mode simple3."""
        prod = self.production[t]

        if self.action_mode == "simple3":
            if action_index == A_SELL:
                d = min(self.max_discharge, soc)
                soc_next = soc - d
                out = prod + self.eta_discharge * d
                forced = 0.0
            elif action_index == A_STORE:
                c = min(self.max_charge, prod,
                        (self.capacity - soc) / self.eta_charge)
                c = max(c, 0.0)
                soc_next = soc + self.eta_charge * c
                out = prod - c
                forced = 0.0
            else: 
                soc_next = soc
                out = prod
                forced = min(self.demand[t], prod)
            soc_next = min(max(soc_next, 0.0), self.capacity)
            return soc_next, out, forced

        u = self.actions[action_index]
        if u >= 0.0:
            c = min(u, prod, (self.capacity - soc) / self.eta_charge)
            c = max(c, 0.0)
            soc_next = soc + self.eta_charge * c
            out = prod - c
        else:
            d = min(-u, soc)
            d = max(d, 0.0)
            soc_next = soc - d
            out = prod + self.eta_discharge * d
        soc_next = min(max(soc_next, 0.0), self.capacity)
        return soc_next, out, None

    def revenue(self, out, forced, t, price):
        """Récompense de l'énergie `out` au prix donné. En mode grid, on consomme
        avant de vendre uniquement si c'est plus rentable (p_consume > prix)."""
        if forced is None:
            if self.p_consume[t] > price:
                consumed = min(self.demand[t], out)
            else:
                consumed = 0.0
        else:
            consumed = forced
        return consumed * self.p_consume[t] + (out - consumed) * price

    def apply(self, soc, t, action_index):
        """Transition + revenu aux prix réels de l'environnement."""
        soc_next, out, forced = self.transition(soc, t, action_index)
        return soc_next, self.revenue(out, forced, t, self.prices[t])

    def reset(self):
        self._t = 0
        self._soc = self.soc0
        return (self._t, self._soc)

    def step(self, action_index):
        soc_next, reward = self.apply(self._soc, self._t, action_index)
        self._t += 1
        self._soc = soc_next
        done = self._t >= self.H
        return (self._t, self._soc), reward, done

    def rollout_policy(self, policy):
        """Joue une politique (t, soc) -> action et renvoie
        (profit total, trajectoire de soc, actions)."""
        soc = self.soc0
        total = 0.0
        soc_traj = [soc]
        actions = []
        for t in range(self.H):
            a = policy(t, soc)
            soc, r = self.apply(soc, t, a)
            total += r
            soc_traj.append(soc)
            actions.append(a)
        return total, np.array(soc_traj), np.array(actions)


def make_synthetic_data(days=2, seed=0):
    """Profil journalier synthétique : prix (creux à midi, pics matin/soir),
    production solaire en cloche, demande interne avec une bosse le soir.
    Renvoie (prix, production, demande)."""
    rng = np.random.default_rng(seed)
    H = 24 * days
    hod = np.arange(H) % 24

    price = (50.0
             + 30.0 * np.exp(-((hod - 19) ** 2) / 8.0)
             + 20.0 * np.exp(-((hod - 8) ** 2) / 6.0)
             - 15.0 * np.exp(-((hod - 13) ** 2) / 10.0))
    price += rng.normal(0.0, 3.0, H)
    price = np.clip(price, 5.0, None)

    prod = 80.0 * np.exp(-((hod - 13) ** 2) / 12.0)
    prod += rng.normal(0.0, 2.0, H)
    prod = np.clip(prod, 0.0, None)

    demand = 15.0 + 10.0 * np.exp(-((hod - 20) ** 2) / 8.0)
    demand += rng.normal(0.0, 1.0, H)
    demand = np.clip(demand, 0.0, None)

    return price, prod, demand
