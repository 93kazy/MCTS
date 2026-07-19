"""MCTS pour le probleme deterministe.

On l'utilise en horizon glissant : à chaque pas on construit un arbre depuis
l'état courant, on joue l'action la plus visitée, puis on recommence au pas
suivant. Chaque noeud estime la somme des récompenses futures.
"""

import math
import numpy as np


class _Node:
    __slots__ = ("t", "soc", "children", "edge_r", "untried", "N", "W")

    def __init__(self, t, soc, n_actions):
        self.t = t
        self.soc = soc
        self.children = {}
        self.edge_r = {}
        self.untried = list(range(n_actions))
        self.N = 0
        self.W = 0.0


class MCTSPlanner:

    def __init__(self, env, n_simulations=600, c=1.0, rollout_policy=None, seed=0):
        self.env = env
        self.H = env.H
        self.n_actions = env.n_actions
        self.n_simulations = n_simulations
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.rollout_policy = rollout_policy  

    def plan_action(self, t, soc):
        root = _Node(t, soc, self.n_actions)
        for _ in range(self.n_simulations):
            self._simulate(root)
        return max(root.children,
                   key=lambda a: (root.children[a].N,
                                  root.edge_r[a]
                                  + root.children[a].W / root.children[a].N))

    def _simulate(self, node):
        if node.t >= self.H:
            node.N += 1
            return 0.0

        if node.untried:
            a = node.untried.pop(self.rng.integers(len(node.untried)))
            soc_next, r = self.env.apply(node.soc, node.t, a)
            child = _Node(node.t + 1, soc_next, self.n_actions)
            node.children[a] = child
            node.edge_r[a] = r
            g_child = self._rollout(child.t, child.soc)
            child.N += 1
            child.W += g_child
            g = r + g_child
        else:
            a = self._uct_select(node)
            child = node.children[a]
            g = node.edge_r[a] + self._simulate(child)

        node.N += 1
        node.W += g
        return g

    def _uct_select(self, node):
        log_n = math.log(node.N)
        q = {a: node.edge_r[a] + ch.W / ch.N for a, ch in node.children.items()}
        q_min, q_max = min(q.values()), max(q.values())
        span = (q_max - q_min) if q_max > q_min else 1.0

        best_score, best_a = -1e18, None
        for a, ch in node.children.items():
            score = (q[a] - q_min) / span + self.c * math.sqrt(log_n / ch.N)
            if score > best_score:
                best_score, best_a = score, a
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

    def run(self):
        """Joue la politique MCTS sur tout l'horizon.
        Renvoie (profit total, trajectoire de soc, actions)."""
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
