"""MCTS stochastique : MCTS/UCT pour MDP a transitions ALEATOIRES.

Difference avec la version deterministe : apres une action, l'etat de prix
suivant est tire au sort dans le modele de Markov (noeud de hasard). On ne fait
PAS d'arbre exhaustif sur les aleas ; a chaque passage on echantillonne l'etat
suivant s' ~ P[s] et on agrege les retours par action. C'est l'approche standard
"UCT pour MDP" : l'exploration porte sur les actions, le hasard est echantillonne.

Comme la dynamique du stockage est independante du prix (soc' deterministe), la
recompense d'une action en un noeud est connue ; seule la transition de l'etat
de prix est aleatoire. Le MCTS n'utilise que le modele GENERATIF (sample_next),
jamais la matrice de transition explicite : c'est ce qui le rend extensible a
des modeles de prix riches ou le SDP serait intraitable.
"""

import math
import numpy as np


class _Node:
    __slots__ = ("t", "soc", "s", "untried", "astats", "kids", "N")

    def __init__(self, t, soc, s, n_actions):
        self.t = t
        self.soc = soc
        self.s = s                       # etat de prix courant
        self.untried = list(range(n_actions))
        self.astats = {}                 # a -> [N_a, W_a] (retour-a-venir cumule)
        self.kids = {}                   # (a, s') -> _Node
        self.N = 0


class StochasticMCTSPlanner:
    def __init__(self, env, model, n_simulations=300, c=1.0,
                 rollout_policy=None, seed=0):
        self.env = env
        self.model = model
        self.H = env.H
        self.n_actions = env.n_actions
        self.n_simulations = n_simulations
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.rollout_policy = rollout_policy   # None => aleatoire

    def plan_action(self, t, soc, s):
        root = _Node(t, soc, s, self.n_actions)
        for _ in range(self.n_simulations):
            self._simulate(root)
        # Action la plus visitee.
        return max(root.astats, key=lambda a: root.astats[a][0])

    def _step(self, soc, t, a, s):
        """Transition + revenu au prix de l'etat s (dynamique sans prix)."""
        soc_n, out, forced = self.env.transition(soc, t, a)
        r = self.env.revenue(out, forced, t, self.model.price(s, t))
        return soc_n, r

    def _simulate(self, node):
        if node.t >= self.H:
            return 0.0

        if node.untried:  # ----- Expansion -----
            a = node.untried.pop(self.rng.integers(len(node.untried)))
            soc_n, r = self._step(node.soc, node.t, a, node.s)
            s2 = self.model.sample_next(node.s, self.rng)        # noeud de hasard
            child = _Node(node.t + 1, soc_n, s2, self.n_actions)
            node.kids[(a, s2)] = child
            g = r + self._rollout(node.t + 1, soc_n, s2)         # ----- Simulation -----
            node.astats[a] = [1, g]
        else:             # ----- Selection (UCT) + recursion -----
            a = self._uct_select(node)
            soc_n, r = self._step(node.soc, node.t, a, node.s)
            s2 = self.model.sample_next(node.s, self.rng)        # on re-echantillonne
            child = node.kids.get((a, s2))
            if child is None:
                child = _Node(node.t + 1, soc_n, s2, self.n_actions)
                node.kids[(a, s2)] = child
            g = r + self._simulate(child)
            st = node.astats[a]
            st[0] += 1
            st[1] += g

        node.N += 1                                              # ----- Retropropagation -----
        return g

    def _uct_select(self, node):
        log_n = math.log(node.N)
        q = {a: node.astats[a][1] / node.astats[a][0] for a in node.astats}
        q_min, q_max = min(q.values()), max(q.values())
        span = (q_max - q_min) if q_max > q_min else 1.0
        best_score, best_a = -1e18, None
        for a in node.astats:
            n_a = node.astats[a][0]
            score = (q[a] - q_min) / span + self.c * math.sqrt(log_n / n_a)
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _rollout(self, t, soc, s):
        g = 0.0
        while t < self.H:
            if self.rollout_policy is None:
                a = int(self.rng.integers(self.n_actions))
            else:
                a = self.rollout_policy(t, soc, s)
            soc, r = self._step(soc, t, a, s)
            g += r
            s = self.model.sample_next(s, self.rng)   # le prix evolue dans le rollout
            t += 1
        return g

    def run_on_path(self, state_path):
        """Joue le MCTS en boucle fermee sur un chemin de prix REALISE : le
        planificateur ne connait pas le futur, agit, puis le vrai prix se
        revele. Renvoie le profit total realise."""
        soc = self.env.soc0
        total = 0.0
        for t in range(self.H):
            s = int(state_path[t])
            a = self.plan_action(t, soc, s)
            soc, r = self._step(soc, t, a, s)
            total += r
        return total


def stochastic_threshold_policy(env, model, threshold):
    """Heuristique de seuil adaptee au stochastique (seuil sur le prix courant) :
    si prix >= seuil on vend au maximum (decharge), sinon on stocke au maximum.
    Sert d'heuristique simple ET de politique de rollout informee pour le MCTS.
    (Mode d'action "grid" attendu, comme dans les demos stochastiques.)"""
    idx_max_discharge = 0
    idx_max_charge = env.n_actions - 1

    def policy(t, soc, s):
        return idx_max_discharge if model.price(s, t) >= threshold else idx_max_charge
    return policy
