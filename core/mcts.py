"""Monte Carlo Tree Search (UCT) pour le MDP deterministe a horizon fini.

Le MCTS est utilise comme PLANIFICATEUR a horizon glissant (style MPC) : a
chaque pas reel, on construit un arbre depuis l'etat courant (t, soc), on
choisit l'action la plus visitee, on l'execute, on avance, et on recommence.
C'est la maniere naturelle d'employer le MCTS comme politique et celle qui
s'etend directement au cas STOCHASTIQUE (il suffit d'echantillonner les prix
dans les rollouts au lieu de les rejouer a l'identique).

Les quatre phases d'une simulation :
  1. Selection   : on descend l'arbre via UCT tant que le noeud est entierement
                   developpe.
  2. Expansion   : on ajoute un enfant pour une action non encore essayee.
  3. Simulation  : rollout (politique aleatoire par defaut) jusqu'a l'horizon.
  4. Retropropagation : on remonte le retour-a-venir (return-to-go).

Chaque noeud estime la valeur RESTANTE depuis lui-meme (somme des recompenses
futures). La valeur d'une action a depuis un noeud = recompense d'arete
r(noeud, a) + valeur estimee de l'enfant. UCT est normalise localement
(min/max des valeurs d'action du noeud) pour rester robuste a l'echelle des
recompenses. Fonctionne dans les deux modes d'action de l'environnement
("grid" et "simple3").
"""

import math
import numpy as np


class _Node:
    __slots__ = ("t", "soc", "children", "edge_r", "untried", "N", "W")

    def __init__(self, t, soc, n_actions):
        self.t = t
        self.soc = soc
        self.children = {}          # action_index -> _Node
        self.edge_r = {}            # action_index -> recompense immediate
        self.untried = list(range(n_actions))
        self.N = 0                  # nombre de visites
        self.W = 0.0                # somme des retours-a-venir observes


class MCTSPlanner:
    """Planificateur MCTS/UCT a horizon glissant."""

    def __init__(self, env, n_simulations=600, c=1.0, rollout_policy=None, seed=0):
        self.env = env
        self.H = env.H
        self.n_actions = env.n_actions
        self.n_simulations = n_simulations
        self.c = c                                  # constante d'exploration
        self.rng = np.random.default_rng(seed)
        self.rollout_policy = rollout_policy        # None => rollout aleatoire

    # ---- choix de l'action pour l'etat courant ---- #
    def plan_action(self, t, soc):
        root = _Node(t, soc, self.n_actions)
        for _ in range(self.n_simulations):
            self._simulate(root)
        # Action la plus visitee (choix robuste, standard en MCTS) ;
        # egalite departagee par la valeur estimee.
        return max(root.children,
                   key=lambda a: (root.children[a].N,
                                  root.edge_r[a]
                                  + root.children[a].W / root.children[a].N))

    # ---- une simulation complete (recursive) ---- #
    def _simulate(self, node):
        # Renvoie le retour-a-venir depuis `node`.
        if node.t >= self.H:
            # Compter aussi les visites des noeuds TERMINAUX : sans cela, au
            # dernier pas de l'horizon, tous les enfants de la racine gardent
            # N = 1 et "l'action la plus visitee" devient un choix arbitraire.
            node.N += 1
            return 0.0

        if node.untried:  # ----- Expansion -----
            a = node.untried.pop(self.rng.integers(len(node.untried)))
            soc_next, r = self.env.apply(node.soc, node.t, a)
            child = _Node(node.t + 1, soc_next, self.n_actions)
            node.children[a] = child
            node.edge_r[a] = r
            g_child = self._rollout(child.t, child.soc)   # ----- Simulation -----
            child.N += 1
            child.W += g_child
            g = r + g_child
        else:             # ----- Selection (UCT) puis recursion -----
            a = self._uct_select(node)
            child = node.children[a]
            g_child = self._simulate(child)
            g = node.edge_r[a] + g_child

        # ----- Retropropagation -----
        node.N += 1
        node.W += g
        return g

    def _uct_select(self, node):
        log_n = math.log(node.N)
        # Valeur d'action = recompense d'arete + valeur moyenne de l'enfant.
        q = {a: node.edge_r[a] + ch.W / ch.N for a, ch in node.children.items()}
        q_min = min(q.values())
        q_max = max(q.values())
        span = (q_max - q_min) if q_max > q_min else 1.0

        best_score = -1e18
        best_a = None
        for a, ch in node.children.items():
            exploit = (q[a] - q_min) / span                 # normalise dans [0, 1]
            explore = self.c * math.sqrt(log_n / ch.N)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _rollout(self, t, soc):
        g = 0.0
        while t < self.H:
            if self.rollout_policy is None:
                a = int(self.rng.integers(self.n_actions))
            else:
                a = self.rollout_policy(t, soc)
            soc, r = self.env.apply(soc, t, a)
            g += r
            t += 1
        return g

    # ---- jeu de la politique MCTS en horizon glissant ---- #
    def run(self):
        """Joue le MCTS pas a pas sur l'environnement. Renvoie profit total,
        trajectoire de SoC et actions choisies."""
        soc = self.env.soc0
        total = 0.0
        soc_traj = [soc]
        actions = []
        for t in range(self.H):
            a = self.plan_action(t, soc)
            soc, r = self.env.apply(soc, t, a)
            total += r
            soc_traj.append(soc)
            actions.append(a)
        return total, np.array(soc_traj), np.array(actions)
