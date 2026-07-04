"""Environnement MDP pour la gestion energetique d'une centrale avec stockage.

Modelisation — probleme de PRODUCTEUR :
A chaque pas de temps t, la centrale produit `production[t]` et observe le prix
spot `prices[t]`. Conformement au sujet, l'energie produite peut etre :
  - VENDUE    : injectee sur le marche au prix spot prices[t],
  - STOCKEE   : chargee en batterie (rendement eta_charge), pour etre revendue
                ou consommee plus tard (rendement eta_discharge),
  - CONSOMMEE : utilisee pour couvrir une demande interne demand[t], valorisee
                au prix d'evitement p_consume[t] (cout de l'electricite qu'on
                n'a pas eu besoin d'acheter, p.ex. tarif de detail).

Seule l'energie produite (ou destockee) est dispatchee : pas d'achat sur le
marche, coherent avec un probleme de producteur.

Etat        : (t, soc)   ou soc = niveau de stockage.
Exogene     : prices[t], production[t], demand[t]  (series temporelles).
Recompense  : energie_vendue * prices[t] + energie_consommee * p_consume[t].
Horizon     : H = len(prices), fini, sans actualisation (gamma = 1).

Deux espaces d'action (`action_mode`) :

  - "simple3" : les TROIS actions litterales du sujet (VENDRE / STOCKER /
    CONSOMMER), soit exactement l'arbre 3^H de l'enonce (3^24 ~ 2.8e11 pour
    une journee). VENDRE = decharge max + tout vendre ; STOCKER = charge max,
    surplus vendu ; CONSOMMER = couvrir la demande interne, surplus vendu.

  - "grid" (defaut) : generalisation ou l'action est un debit de stockage u
    parmi n_actions valeurs (u > 0 charge, u < 0 decharge, u = 0 rien).
    L'energie restante est repartie entre vente et consommation par DOMINANCE :
    a energie dispatchee donnee, consommer rapporte p_consume[t] et vendre
    prices[t], sans aucun couplage temporel ; l'affectation optimale est donc
    immediate (consommer d'abord si p_consume[t] > prices[t], dans la limite
    de la demande, vendre le reste). La seule decision sequentielle non
    triviale — le stockage — reste seule dans l'espace d'action, qui n'est pas
    pollue d'actions dominees. L'arbre est alors n_actions^H >> 3^H, ce qui
    renforce l'argument combinatoire du sujet.

Interface pour les planificateurs :
  transition(soc, t, a) -> (soc_next, out, forced)   dynamique SANS prix
  revenue(out, forced, t, price) -> recompense       valorisation A prix donne
  apply(soc, t, a) -> (soc_next, recompense)         raccourci deterministe
La separation transition/revenue permet aux methodes stochastiques de brancher
un prix externe (modele de prix) sans dupliquer la dynamique.
"""

import numpy as np

# Indices d'action du mode "simple3" (les trois decisions du sujet).
A_SELL, A_STORE, A_CONSUME = 0, 1, 2
SIMPLE3_NAMES = {A_SELL: "VENDRE", A_STORE: "STOCKER", A_CONSUME: "CONSOMMER"}


class EnergyStorageEnv:
    """Simulateur du probleme de gestion energetique (deterministe par defaut)."""

    def __init__(self, prices, production, demand=None, p_consume=0.0,
                 capacity=100.0, eta_charge=0.95, eta_discharge=0.95,
                 max_charge=30.0, max_discharge=30.0, soc0=0.0,
                 n_actions=11, action_mode="grid"):
        self.prices = np.asarray(prices, dtype=float)
        self.production = np.asarray(production, dtype=float)
        assert len(self.prices) == len(self.production)
        self.H = len(self.prices)

        # Canal consommation : demande interne + prix d'evitement.
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
            # Grille de debits u, symetrique autour de 0 (n_actions impair
            # => 0 est inclus).
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

        # Etat courant (pour l'interface gym-like reset/step).
        self._t = 0
        self._soc = self.soc0

    # ------------------------------------------------------------------ #
    # Dynamique SANS prix.                                                #
    # ------------------------------------------------------------------ #
    def transition(self, soc, t, action_index):
        """(soc, t, action) -> (soc_next, out, forced). Ne modifie rien.

        out    : energie dispatchee ce pas (a repartir entre vente et conso).
        forced : consommation IMPOSEE par l'action (mode simple3), ou None en
                 mode grid — la repartition optimale vente/conso depend alors
                 du prix et est differee a `revenue`.
        """
        prod = self.production[t]

        if self.action_mode == "simple3":
            if action_index == A_SELL:      # decharge max, tout est vendu
                d = min(self.max_discharge, soc)
                soc_next = soc - d
                out = prod + self.eta_discharge * d
                forced = 0.0
            elif action_index == A_STORE:   # charge max, surplus vendu
                c = min(self.max_charge, prod,
                        (self.capacity - soc) / self.eta_charge)
                c = max(c, 0.0)
                soc_next = soc + self.eta_charge * c
                out = prod - c
                forced = 0.0
            else:                           # CONSOMMER : demande d'abord
                soc_next = soc
                out = prod
                forced = min(self.demand[t], prod)
            soc_next = min(max(soc_next, 0.0), self.capacity)
            return soc_next, out, forced

        # ---- mode grid : action = debit de stockage u ----
        u = self.actions[action_index]
        if u >= 0.0:  # charge : on stocke une partie de la production
            c = min(u, prod, (self.capacity - soc) / self.eta_charge)
            c = max(c, 0.0)
            soc_next = soc + self.eta_charge * c
            out = prod - c
        else:  # decharge : on vide la batterie pour dispatcher plus
            d = min(-u, soc)
            d = max(d, 0.0)
            soc_next = soc - d
            out = prod + self.eta_discharge * d
        soc_next = min(max(soc_next, 0.0), self.capacity)
        return soc_next, out, None

    # ------------------------------------------------------------------ #
    # Valorisation a prix donne.                                          #
    # ------------------------------------------------------------------ #
    def revenue(self, out, forced, t, price):
        """Recompense de l'energie dispatchee `out` au prix spot `price`.

        forced=None (grid) : repartition optimale par dominance — on consomme
        (valeur p_consume[t]) avant de vendre (valeur price) ssi c'est plus
        rentable, dans la limite de la demande. Optimal car sans couplage
        temporel : la repartition n'affecte pas soc_next.
        """
        if forced is None:
            if self.p_consume[t] > price:
                consumed = min(self.demand[t], out)
            else:
                consumed = 0.0
        else:
            consumed = forced
        return consumed * self.p_consume[t] + (out - consumed) * price

    # ------------------------------------------------------------------ #
    # Modele generatif deterministe (heuristiques, PD, MCTS).             #
    # ------------------------------------------------------------------ #
    def apply(self, soc, t, action_index):
        """Applique une action depuis (soc, t) avec les prix deterministes de
        l'environnement. Ne modifie pas l'etat interne."""
        soc_next, out, forced = self.transition(soc, t, action_index)
        return soc_next, self.revenue(out, forced, t, self.prices[t])

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
        le profit total, la trajectoire de SoC et les actions."""
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
# Donnees synthetiques (maitrise des parametres experimentaux).           #
# ---------------------------------------------------------------------- #
def make_synthetic_data(days=2, seed=0):
    """Genere un profil journalier plausible : (prix, production, demande).

    Prix : creux en milieu de journee (afflux solaire -> prix bas, type
    'duck curve'), pics le matin (~8h) et le soir (~19h).
    Production : profil solaire en cloche centre sur midi.
    Demande interne : socle constant + bosse du soir (~20h).
    Le decalage prix/production rend le stockage utile ; la demande donne un
    troisieme usage (consommer) arbitre contre le prix spot.
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

    demand = 15.0 + 10.0 * np.exp(-((hod - 20) ** 2) / 8.0)
    demand += rng.normal(0.0, 1.0, H)
    demand = np.clip(demand, 0.0, None)

    return price, prod, demand
