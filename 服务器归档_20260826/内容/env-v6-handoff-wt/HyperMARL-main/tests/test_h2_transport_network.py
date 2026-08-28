import unittest

import numpy as np

from envs.microgrid.h2_transport import H2TransportNetwork


class H2TransportNetworkTest(unittest.TestCase):
    def _network(self, **overrides):
        config = {
            "num_agents": 4,
            "h2_traffic_min_eta": 4,
            "h2_traffic_max_eta": 6,
            "h2_traffic_truck_capacity_kg": 500.0,
            "h2_traffic_edge_capacity": 8.0,
            "h2_traffic_bpr_alpha": 0.15,
            "h2_traffic_bpr_beta": 4.0,
            "h2_traffic_seed": 20260716,
        }
        config.update(overrides)
        network = H2TransportNetwork(config)
        network.reset(day_index=8, seed=30)
        return network

    def test_complete_directed_graph_and_three_stable_routes(self):
        network = self._network()
        self.assertEqual(len(network.edge_ids), 12)
        routes = network.route_options(seller_id=0, buyer_id=3)
        self.assertEqual(routes[0], (0, 3))
        self.assertEqual(routes[1:], ((0, 1, 3), (0, 2, 3)))
        self.assertEqual(len(set(routes)), 3)

    def test_a6_bins_select_direct_low_and_high_intermediate_routes(self):
        network = self._network()
        for action, expected_rank in ((-1.0, 0), (-1.0 / 3.0, 1), (0.0, 1), (1.0 / 3.0, 2), (1.0, 2)):
            with self.subTest(action=action):
                rank, path = network.choose_route(0, 3, action)
                self.assertEqual(rank, expected_rank)
                self.assertEqual(path, network.route_options(0, 3)[expected_rank])

    def test_seeded_background_has_reproducible_daily_peaks(self):
        first = self._network()
        second = self._network()
        np.testing.assert_allclose(
            list(first.background_utilization(8).values()),
            list(second.background_utilization(8).values()),
            rtol=0.0,
            atol=0.0,
        )
        midnight = np.mean(list(first.background_utilization(0).values()))
        morning = np.mean(list(first.background_utilization(8).values()))
        evening = np.mean(list(first.background_utilization(18).values()))
        self.assertGreater(morning, midnight)
        self.assertGreater(evening, midnight)

        different_day = H2TransportNetwork(first.config)
        different_day.reset(day_index=17, seed=30)
        self.assertFalse(np.allclose(
            list(first.background_utilization(8).values()),
            list(different_day.background_utilization(8).values()),
        ))

    def test_peak_amplitudes_are_configurable_without_random_incidents(self):
        quiet = self._network(
            h2_traffic_morning_peak_amplitude=0.0,
            h2_traffic_evening_peak_amplitude=0.0,
        )
        peaked = self._network(
            h2_traffic_morning_peak_amplitude=1.0,
            h2_traffic_evening_peak_amplitude=1.0,
        )
        quiet_mean = np.mean(list(quiet.background_utilization(8).values()))
        peak_mean = np.mean(list(peaked.background_utilization(8).values()))
        self.assertGreater(peak_mean - quiet_mean, 0.5)

    def test_directional_peak_phases_make_a_detour_strictly_useful(self):
        network = self._network(
            h2_traffic_morning_peak_amplitude=0.85,
            h2_traffic_evening_peak_amplitude=0.95,
        )
        useful = []
        for t in range(24):
            utilization = network.background_utilization(t)
            for seller in range(4):
                for buyer in range(4):
                    if seller == buyer:
                        continue
                    etas = [
                        network._route_eta(path, utilization)[0]
                        for path in network.route_options(seller, buyer)
                    ]
                    useful.append(min(etas[1:]) < etas[0])
        self.assertTrue(any(useful))

    def test_route_features_are_finite_normalized_and_fixed_width(self):
        features = self._network().route_features(buyer_id=2, t=8)
        self.assertEqual(features.shape, (3,))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features >= 0.0))
        self.assertTrue(np.all(features <= 1.0))

    def test_assignment_is_order_independent_and_uses_buyer_action_only(self):
        trades = [
            {"seller_id": 0, "buyer_id": 2, "quantity": 900.0, "price": 0.5},
            {"seller_id": 1, "buyer_id": 3, "quantity": 600.0, "price": 0.6},
        ]
        actions = np.asarray([-1.0, 1.0, 0.0, 1.0], dtype=np.float32)
        first = self._network().assign_shipments(trades, actions, dispatch_t=8)
        second = self._network().assign_shipments(list(reversed(trades)), actions, dispatch_t=8)
        self.assertEqual(first, second)

        changed_seller_actions = actions.copy()
        changed_seller_actions[0] = 1.0
        changed_seller_actions[1] = -1.0
        third = self._network().assign_shipments(trades, changed_seller_actions, dispatch_t=8)
        self.assertEqual(first, third)

    def test_more_same_route_flow_never_reduces_eta_and_eta_is_bounded(self):
        actions = np.full(4, -1.0, dtype=np.float32)
        small = self._network().assign_shipments(
            [{"seller_id": 0, "buyer_id": 3, "quantity": 10.0, "price": 0.5}],
            actions,
            dispatch_t=8,
        )[0]
        large = self._network().assign_shipments(
            [{"seller_id": 0, "buyer_id": 3, "quantity": 20000.0, "price": 0.5}],
            actions,
            dispatch_t=8,
        )[0]
        self.assertGreaterEqual(large["eta"], small["eta"])
        for shipment in (small, large):
            self.assertGreaterEqual(shipment["eta"], 4)
            self.assertLessEqual(shipment["eta"], 6)
            self.assertEqual(shipment["deliver_at"], 8 + shipment["eta"])

    def test_bpr_delay_multiplies_free_flow_time(self):
        network = self._network()
        eta, delay, nominal = network._route_eta((0, 1), {(0, 1): 1.0})
        self.assertEqual(nominal, 4)
        self.assertAlmostEqual(delay, 0.6, delta=1e-9)
        self.assertEqual(eta, 5)

    def test_v1_rejects_eta_bounds_other_than_four_to_six(self):
        for key, value in (("h2_traffic_min_eta", 3), ("h2_traffic_max_eta", 7)):
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, "exactly 4..6"):
                    self._network(**{key: value})

    def test_shipment_tracks_gross_net_and_loss_exactly_once(self):
        shipment = self._network().assign_shipments(
            [{"seller_id": 0, "buyer_id": 1, "quantity": 1000.0, "price": 0.5}],
            np.full(4, -1.0, dtype=np.float32),
            dispatch_t=4,
            transport_loss=0.1,
        )[0]
        self.assertAlmostEqual(shipment["gross_quantity"], 1000.0)
        self.assertAlmostEqual(shipment["loss_quantity"], 100.0)
        self.assertAlmostEqual(shipment["net_quantity"], 900.0)
        self.assertAlmostEqual(
            shipment["gross_quantity"],
            shipment["net_quantity"] + shipment["loss_quantity"],
        )


if __name__ == "__main__":
    unittest.main()
