"""Environnement MDP pour la gestion energetique d'une centrale avec stockage.

Modelisation (rappel phase 1) — probleme de PRODUCTEUR :
A chaque pas de temps t, la centrale produit `production[t]` et observe le prix
spot `prices[t]`. Elle choisit un debit de stockage `u` (l'action) :
  - u > 0 : stocker une partie de la production courante (charge),
  - u < 0 : destocker pour vendre davantage (decharge),
  - u = 0 : tout vendre immediatement.

Seule l'energie produite est dispatchee : pas d'achat sur le marche, ce qui est
coherent avec un probleme de producteur (vendre / stocker / consommer). Le
stockage ne sert donc qu'a DECALER dans le temps la vente de sa propre
production (vendre les heures cheres plutot que les heures peu cheres).

Etat        : (t, soc)   ou soc = niveau de stockage.
Exogene     : prices[t], production[t]  (series temporelles).
Recompense  : energie_vendue * prices[t].
Horizon     : H = len(prices), fini, sans actualisation (gamma = 1).

NB extension "consommer" : la consommation interne se modelise en valorisant une
part de l'energie a un prix d'evitement (cout de l'electricite non achetee). Le
mecanisme est identique a la vente, avec un autre signal de prix ; laisse en
extension pour garder l'espace d'action a une seule dimension.
"""

import numpy as np


class EnergyStorageEnv:
    """Simulateur deterministe du probleme de gestion energetique."""

    def __init__(self, prices, production, capacity=100.0, eta_charge=0.95,
                 eta_discharge=0.95, max_charge=30.0, max_discharge=30.0,
                 soc0=0.0, n_actions=11):
        self.prices = np.asarray(prices, dtype=float)
        self.production = np.asarray(production, dtype=float)
        assert len(self.prices) == len(self.production)
        self.H = len(self.prices)

        self.capacity = float(capacity)
        self.eta_charge = float(eta_charge)
        self.eta_discharge = float(eta_discharge)
        self.max_charge = float(max_charge)
        self.max_discharge = float(max_discharge)
        self.soc0 = float(soc0)

        # Grille d'actions : debits de stockage u, symetrique autour de 0.
        # n_actions impair => 0 (tout vendre) est inclus.
        k = max(1, (n_actions - 1) // 2)
        charge = np.linspace(0.0, self.max_charge, k + 1)[1:]
        discharge = -np.linspace(0.0, self.max_discharge, k + 1)[1:]
        self.actions = np.concatenate([discharge[::-1], [0.0], charge])
        self.n_actions = len(self.actions)

        # Etat courant (pour l'interface gym-like reset/step).
        self._t = 0
        self._soc = self.soc0

    # ------------------------------------------------------------------ #
    # Modele generatif : utilise par les heuristiques, la PD et le MCTS.  #
    # Deterministe : (soc, t, action) -> (soc_suivant, recompense).       #
    # ------------------------------------------------------------------ #
    def dispatch(self, soc, t, action_index):
        """Dynamique SANS prix : (soc, t, action) -> (soc_suivant, energie_vendue).

        La mise a jour du stockage ne depend pas du prix ; seul le revenu en
        depend. Cette separation permet aux methodes stochastiques de brancher un
        prix externe (issu du modele de prix) sans dupliquer la dynamique.
        """
        u = self.actions[action_index]
        prod = self.production[t]

        if u >= 0.0:  # charge : on stocke une partie de la production
            # On ne peut charger que ce qu'on produit et ce qui tient en batterie.
            c = min(u, prod, (self.capacity - soc) / self.eta_charge)
            c = max(c, 0.0)
            soc_next = soc + self.eta_charge * c
            sold = prod - c
        else:  # decharge : on vide la batterie pour vendre plus
            d = min(-u, soc)
            d = max(d, 0.0)
            soc_next = soc - d
            sold = prod + self.eta_discharge * d

        soc_next = min(max(soc_next, 0.0), self.capacity)
        return soc_next, sold

    def apply(self, soc, t, action_index):
        """Applique une action depuis (soc, t) avec les prix deterministes de
        l'environnement. Ne modifie pas l'etat interne."""
        soc_next, sold = self.dispatch(soc, t, action_index)
        return soc_next, sold * self.prices[t]

    # ------------------------------------------------------------------ #
    # Interface gym-like (pratique pour rejouer une trajectoire).         #
    # ------------------------------------------------------------------ #
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
        """Joue une politique (fonction (t, soc) -> action_index) et renvoie
        le profit total et la trajectoire de SoC."""
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


# ---------------------------------------------------------------------- #
# Donnees synthetiques (placeholder en attendant EPEX SPOT France).      #
# ---------------------------------------------------------------------- #
def make_synthetic_data(days=2, seed=0):
    """Genere un profil journalier plausible.

    Prix : creux en milieu de journee (afflux solaire -> prix bas, type
    'duck curve'), pics le matin (~8h) et le soir (~19h).
    Production : profil solaire en cloche centre sur midi.
    Ce decalage rend le stockage utile : stocker le solaire de midi (heures
    peu cheres) pour le vendre le soir (heures cheres).
    """
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

    return price, prod
