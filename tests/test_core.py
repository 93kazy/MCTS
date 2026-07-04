"""Tests unitaires du coeur du projet.

Lancement (depuis la racine) :
    python -m unittest discover tests -v
"""

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.environment import (EnergyStorageEnv, make_synthetic_data,
                              A_SELL, A_STORE, A_CONSUME)
from core.baselines import policy_threshold, dp_optimal
from core.mcts import MCTSPlanner


def small_env(**kw):
    """Petit environnement de test aux parametres simples."""
    defaults = dict(
        prices=[10.0, 50.0, 20.0, 80.0],
        production=[40.0, 40.0, 40.0, 40.0],
        capacity=50.0, eta_charge=0.9, eta_discharge=0.8,
        max_charge=20.0, max_discharge=20.0, soc0=0.0, n_actions=3)
    defaults.update(kw)
    return EnergyStorageEnv(**defaults)


class TestDynamics(unittest.TestCase):
    """Dynamique du stockage (mode grid) : bornes, rendements, comptabilite."""

    def test_action_grid_contains_zero_and_is_symmetric(self):
        env = small_env(n_actions=11)
        self.assertIn(0.0, env.actions)
        np.testing.assert_allclose(env.actions, -env.actions[::-1])

    def test_charge_respects_capacity_and_efficiency(self):
        env = small_env()
        # Charge max depuis un stock presque plein : ne doit pas depasser capacity.
        soc0 = env.capacity - 1.0
        a_charge = env.n_actions - 1                     # u = +max_charge
        soc_next, out, _ = env.transition(soc0, 0, a_charge)
        self.assertLessEqual(soc_next, env.capacity + 1e-9)
        # L'energie prelevee sur la production = (soc_next - soc0) / eta_charge.
        taken = (soc_next - soc0) / env.eta_charge
        self.assertAlmostEqual(out, env.production[0] - taken)

    def test_charge_bounded_by_production(self):
        env = small_env(production=[5.0, 5.0, 5.0, 5.0])
        a_charge = env.n_actions - 1                     # u = 20 > prod = 5
        soc_next, out, _ = env.transition(0.0, 0, a_charge)
        self.assertAlmostEqual(soc_next, env.eta_charge * 5.0)   # tout stocke
        self.assertAlmostEqual(out, 0.0)                          # rien a vendre

    def test_discharge_bounded_by_soc(self):
        env = small_env()
        soc0 = 3.0
        soc_next, out, _ = env.transition(soc0, 0, 0)    # u = -max_discharge
        self.assertAlmostEqual(soc_next, 0.0)
        self.assertAlmostEqual(out, env.production[0] + env.eta_discharge * soc0)

    def test_apply_reward_is_out_times_price_without_demand(self):
        env = small_env()
        idx_zero = int(np.argmin(np.abs(env.actions)))
        _, r = env.apply(0.0, 1, idx_zero)
        self.assertAlmostEqual(r, env.production[1] * env.prices[1])


class TestConsumptionChannel(unittest.TestCase):
    """Canal consommer : repartition par dominance et plafond de demande."""

    def test_consume_when_avoidance_price_higher(self):
        env = small_env(demand=[30.0] * 4, p_consume=100.0)   # p_c > tous les prix
        idx_zero = int(np.argmin(np.abs(env.actions)))
        _, r = env.apply(0.0, 0, idx_zero)
        # 30 consommes a 100, le reste (10) vendu a 10.
        self.assertAlmostEqual(r, 30.0 * 100.0 + 10.0 * 10.0)

    def test_sell_when_spot_price_higher(self):
        env = small_env(demand=[30.0] * 4, p_consume=5.0)     # p_c < tous les prix
        idx_zero = int(np.argmin(np.abs(env.actions)))
        _, r = env.apply(0.0, 0, idx_zero)
        self.assertAlmostEqual(r, 40.0 * 10.0)                # tout vendu

    def test_consumption_capped_by_demand(self):
        env = small_env(demand=[1000.0] * 4, p_consume=100.0)
        idx_zero = int(np.argmin(np.abs(env.actions)))
        _, r = env.apply(0.0, 0, idx_zero)
        self.assertAlmostEqual(r, 40.0 * 100.0)               # borne = out, pas demand


class TestSimple3Mode(unittest.TestCase):
    """Mode 'simple3' : les trois actions litterales du sujet."""

    def make(self):
        return small_env(action_mode="simple3", demand=[15.0] * 4, p_consume=60.0)

    def test_three_actions(self):
        env = self.make()
        self.assertEqual(env.n_actions, 3)

    def test_sell_discharges_and_sells_everything(self):
        env = self.make()
        soc_next, out, forced = env.transition(10.0, 0, A_SELL)
        self.assertAlmostEqual(soc_next, 0.0)
        self.assertAlmostEqual(out, 40.0 + env.eta_discharge * 10.0)
        self.assertAlmostEqual(forced, 0.0)

    def test_store_charges_max_and_sells_surplus(self):
        env = self.make()
        soc_next, out, forced = env.transition(0.0, 0, A_STORE)
        self.assertAlmostEqual(soc_next, env.eta_charge * env.max_charge)
        self.assertAlmostEqual(out, 40.0 - env.max_charge)
        self.assertAlmostEqual(forced, 0.0)

    def test_consume_serves_demand_first(self):
        env = self.make()
        soc_next, out, forced = env.transition(0.0, 0, A_CONSUME)
        self.assertAlmostEqual(soc_next, 0.0)
        self.assertAlmostEqual(out, 40.0)
        self.assertAlmostEqual(forced, 15.0)                  # min(demande, prod)
        r = env.revenue(out, forced, 0, env.prices[0])
        self.assertAlmostEqual(r, 15.0 * 60.0 + 25.0 * 10.0)


class TestPlanners(unittest.TestCase):
    """PD exacte vs enumeration exhaustive, et MCTS vs optimum."""

    def brute_force(self, env):
        """Optimum exact par enumeration de toutes les sequences d'actions."""
        best = -np.inf
        for seq in itertools.product(range(env.n_actions), repeat=env.H):
            soc, total = env.soc0, 0.0
            for t, a in enumerate(seq):
                soc, r = env.apply(soc, t, a)
                total += r
            best = max(best, total)
        return best

    def test_dp_matches_brute_force(self):
        env = small_env()                                     # 3^4 = 81 sequences
        brute = self.brute_force(env)
        dp_policy, v0 = dp_optimal(env, n_soc=401)
        played, _, _ = env.rollout_policy(dp_policy)
        # v0 et le profit joue doivent retrouver l'optimum exact (a la
        # discretisation de la grille de soc pres).
        self.assertAlmostEqual(v0, brute, delta=0.01 * abs(brute))
        self.assertAlmostEqual(played, brute, delta=0.01 * abs(brute))

    def test_mcts_reaches_brute_force_optimum(self):
        # Instance piegeuse : la valeur de "stocker" a t=2 n'apparait qu'un pas
        # plus tard -> il faut assez d'exploration (c) et de budget pour que
        # UCT ne famine pas cette action. Avec c=2 et 5000 simulations, le
        # MCTS retrouve l'optimum exact de l'enumeration.
        env = small_env()
        brute = self.brute_force(env)
        profit, _, _ = MCTSPlanner(env, n_simulations=5000, c=2.0,
                                   rollout_policy=None, seed=0).run()
        self.assertGreaterEqual(profit, 0.995 * brute)

    def test_mcts_informed_beats_or_matches_threshold_on_synthetic(self):
        prices, prod, demand = make_synthetic_data(days=1, seed=3)
        env = EnergyStorageEnv(prices, prod, demand=demand, p_consume=70.0,
                               capacity=120.0, max_charge=30.0,
                               max_discharge=30.0, n_actions=11)
        thr = policy_threshold(env)
        p_thr, _, _ = env.rollout_policy(thr)
        p_mcts, _, _ = MCTSPlanner(env, n_simulations=600, c=1.0,
                                   rollout_policy=thr, seed=0).run()
        # Le MCTS informe par le seuil ne doit pas faire (nettement) moins bien
        # que le seuil seul.
        self.assertGreaterEqual(p_mcts, 0.99 * p_thr)


if __name__ == "__main__":
    unittest.main()
