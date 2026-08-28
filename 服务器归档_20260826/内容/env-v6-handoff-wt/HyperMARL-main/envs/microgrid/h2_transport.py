"""Deterministic traffic assignment for delayed internal hydrogen trades.

The model is deliberately small: four microgrid nodes form a complete
directed road graph.  Each buyer/seller pair has a direct path and two
one-stop alternatives.  Background traffic follows reproducible morning and
evening peaks, while all H2 shipments dispatched in the same environment step
are assigned simultaneously before BPR-style delays are calculated.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


EPS = 1e-9


class H2TransportNetwork:
    """Assign CDA hydrogen trades to routes and calculate bounded dynamic ETA."""

    def __init__(self, config: Mapping[str, object]):
        self.config = dict(config)
        self.num_nodes = int(self.config.get("num_agents", 4))
        if self.num_nodes != 4:
            raise ValueError("H2 traffic v1 requires exactly four microgrid nodes")
        # v2 (设计规格 1a): 可选外部供应站节点 EXT (id = num_agents), 计划性
        # 外购走同一张路网, 与内部交易共享道路容量、共同制造拥堵。
        self.external_node_enable = bool(
            self.config.get("h2_traffic_external_node_enable", False)
        )
        self.external_node_id = self.num_nodes if self.external_node_enable else None
        self.num_route_nodes = self.num_nodes + (1 if self.external_node_enable else 0)

        legacy_min = int(self.config.get("h2_traffic_min_eta", 4))
        legacy_max = int(self.config.get("h2_traffic_max_eta", 6))
        has_v2_eta = (
            "h2_traffic_eta_min" in self.config
            or "h2_traffic_eta_max" in self.config
        )
        self.min_eta = int(self.config.get("h2_traffic_eta_min", legacy_min))
        self.max_eta = int(self.config.get("h2_traffic_eta_max", legacy_max))
        if has_v2_eta or self.external_node_enable:
            # v2 (设计规格 1c): 放宽 ETA 上界, 拥堵/绕行才能真正拉开时距。
            if not 1 <= self.min_eta < self.max_eta <= 24:
                raise ValueError(
                    "H2 traffic v2 ETA bounds must satisfy 1 <= min < max <= 24"
                )
        elif (self.min_eta, self.max_eta) != (4, 6):
            raise ValueError("H2 traffic v1 ETA bounds must be exactly 4..6 hours")
        # v2 (设计规格 1d): 在途损耗按小时累计 (boil-off), 默认 0 = v1 行为。
        self.transit_loss_per_hour = float(
            self.config.get("h2_traffic_transit_loss_per_hour", 0.0)
        )
        if not 0.0 <= self.transit_loss_per_hour < 1.0:
            raise ValueError("h2_traffic_transit_loss_per_hour must be in [0, 1)")

        self.truck_capacity_kg = float(
            self.config.get("h2_traffic_truck_capacity_kg", 500.0)
        )
        self.edge_capacity = float(
            self.config.get("h2_traffic_edge_capacity", 8.0)
        )
        self.bpr_alpha = float(self.config.get("h2_traffic_bpr_alpha", 0.15))
        self.bpr_beta = float(self.config.get("h2_traffic_bpr_beta", 4.0))
        self.background_base_min = float(
            self.config.get("h2_traffic_background_base_min", 0.25)
        )
        self.background_base_max = float(
            self.config.get("h2_traffic_background_base_max", 0.45)
        )
        self.morning_peak_amplitude = float(
            self.config.get("h2_traffic_morning_peak_amplitude", 1.0)
        )
        self.evening_peak_amplitude = float(
            self.config.get("h2_traffic_evening_peak_amplitude", 1.1)
        )
        self.peak_width_hours = float(
            self.config.get("h2_traffic_peak_width_hours", 2.0)
        )
        self.directional_phase_hours = float(
            self.config.get("h2_traffic_directional_phase_hours", 4.0)
        )
        self.lhv_h2 = float(self.config.get("LHV_H2", 33.33))
        if self.truck_capacity_kg <= 0.0 or self.edge_capacity <= 0.0:
            raise ValueError("H2 traffic truck and edge capacities must be positive")
        if (
            self.background_base_min < 0.0
            or self.background_base_max < self.background_base_min
            or self.morning_peak_amplitude < 0.0
            or self.evening_peak_amplitude < 0.0
            or self.peak_width_hours <= 0.0
            or self.directional_phase_hours < 0.0
        ):
            raise ValueError("H2 traffic background profile parameters are invalid")

        self.base_seed = int(self.config.get("h2_traffic_seed", 20260716))
        self.edge_ids = tuple(
            (source, target)
            for source in range(self.num_route_nodes)
            for target in range(self.num_route_nodes)
            if source != target
        )
        self.day_index = 0
        self.seed_value = self.base_seed
        self._edge_base = {}
        self._morning_weight = {}
        self._evening_weight = {}
        self._morning_phase = {
            edge: (
                (((edge[0] * 3 + edge[1] * 5) % 7) - 3)
                / 3.0
                * self.directional_phase_hours
            )
            for edge in self.edge_ids
        }
        self._evening_phase = {
            edge: -self._morning_phase[edge] for edge in self.edge_ids
        }
        self.last_edge_utilization = {edge: 0.0 for edge in self.edge_ids}
        self.last_background_utilization = {edge: 0.0 for edge in self.edge_ids}
        self.reset()

    def reset(self, day_index: int = 0, seed: int | None = None):
        """Rebuild deterministic edge-specific traffic factors for one episode."""

        self.day_index = int(day_index)
        self.seed_value = self.base_seed if seed is None else int(seed)
        mixed_seed = (
            self.base_seed
            + 1009 * self.day_index
            + 9176 * self.seed_value
        ) % (2**32 - 1)
        rng = np.random.RandomState(mixed_seed)
        self._edge_base = {
            edge: float(rng.uniform(self.background_base_min, self.background_base_max))
            for edge in self.edge_ids
        }
        self._morning_weight = {
            edge: float(rng.uniform(0.85, 1.15)) for edge in self.edge_ids
        }
        self._evening_weight = {
            edge: float(rng.uniform(0.85, 1.15)) for edge in self.edge_ids
        }
        self.last_edge_utilization = self.background_utilization(0)
        self.last_background_utilization = dict(self.last_edge_utilization)
        return self

    def route_options(self, seller_id: int, buyer_id: int) -> tuple[tuple[int, ...], ...]:
        """Return direct, lower-index detour, and higher-index detour paths."""

        seller = int(seller_id)
        buyer = int(buyer_id)
        if seller == buyer:
            raise ValueError("A hydrogen shipment cannot have the same buyer and seller")
        if not (0 <= seller < self.num_route_nodes and 0 <= buyer < self.num_route_nodes):
            raise ValueError("H2 traffic buyer/seller node is out of range")
        # 绕行只取序号最小的两个中间节点: 4 节点时与 v1 完全一致; 启用 EXT
        # 节点后, 微电网对之间的绕行仍不经过供应站 (EXT id 最大, 排不进前二)。
        intermediates = sorted(set(range(self.num_route_nodes)) - {seller, buyer})
        return (
            (seller, buyer),
            (seller, intermediates[0], buyer),
            (seller, intermediates[1], buyer),
        )

    def choose_route(
        self, seller_id: int, buyer_id: int, action: float
    ) -> tuple[int, tuple[int, ...]]:
        """Map continuous a6 to one of three stable candidate route ranks."""

        value = float(np.clip(action, -1.0, 1.0))
        if value < -1.0 / 3.0:
            rank = 0
        elif value < 1.0 / 3.0:
            rank = 1
        else:
            rank = 2
        return rank, self.route_options(seller_id, buyer_id)[rank]

    @staticmethod
    def _path_edges(path: Sequence[int]) -> tuple[tuple[int, int], ...]:
        return tuple((int(path[i]), int(path[i + 1])) for i in range(len(path) - 1))

    def background_utilization(self, t: int) -> dict[tuple[int, int], float]:
        """Return edge v/c ratios with deterministic 08:00 and 18:00 peaks."""

        hour = float(int(t) % 24)
        utilization = {}
        for edge in self.edge_ids:
            morning = math.exp(
                -0.5
                * ((hour - (8.0 + self._morning_phase[edge])) / self.peak_width_hours)
                ** 2
            )
            evening = math.exp(
                -0.5
                * ((hour - (18.0 + self._evening_phase[edge])) / self.peak_width_hours)
                ** 2
            )
            utilization[edge] = float(
                np.clip(
                    self._edge_base.get(edge, 0.35)
                    + self.morning_peak_amplitude
                    * self._morning_weight.get(edge, 1.0)
                    * morning
                    + self.evening_peak_amplitude
                    * self._evening_weight.get(edge, 1.0)
                    * evening,
                    0.0,
                    1.5,
                )
            )
        return utilization

    def _route_eta(
        self,
        path: Sequence[int],
        utilization: Mapping[tuple[int, int], float],
    ) -> tuple[int, float, int]:
        edges = self._path_edges(path)
        nominal_eta = self.min_eta if len(edges) == 1 else min(
            self.max_eta, self.min_eta + 1
        )
        edge_free_flow_time = float(nominal_eta) / max(len(edges), 1)
        delay = sum(
            edge_free_flow_time
            * self.bpr_alpha
            * max(0.0, float(utilization.get(edge, 0.0))) ** self.bpr_beta
            for edge in edges
        )
        rounded = int(math.floor(float(nominal_eta) + delay + 0.5))
        eta = int(np.clip(rounded, self.min_eta, self.max_eta))
        return eta, float(delay), int(nominal_eta)

    def route_features(self, buyer_id: int, t: int) -> np.ndarray:
        """Return three current route-rank ETA indicators without future leakage."""

        buyer = int(buyer_id)
        if not 0 <= buyer < self.num_nodes:
            raise ValueError("H2 traffic buyer node is out of range")
        background = self.background_utilization(t)
        sellers = [node for node in range(self.num_route_nodes) if node != buyer]
        mean_eta = []
        for rank in range(3):
            values = [
                self._route_eta(self.route_options(seller, buyer)[rank], background)[0]
                for seller in sellers
            ]
            mean_eta.append(float(np.mean(values)))
        width = max(1.0, float(self.max_eta - self.min_eta))
        return np.clip(
            (np.asarray(mean_eta, dtype=np.float32) - self.min_eta) / width,
            0.0,
            1.0,
        ).astype(np.float32)

    def assign_shipments(
        self,
        trades: Iterable[Mapping[str, object]],
        route_actions: Sequence[float],
        dispatch_t: int,
        transport_loss: float = 0.0,
    ) -> list[dict[str, object]]:
        """Assign all same-step trades simultaneously and return shipment records."""

        actions = np.asarray(route_actions, dtype=np.float64).reshape(-1)
        if actions.size < self.num_nodes:
            raise ValueError("route_actions must provide one scalar for every agent")
        loss_ratio = float(np.clip(transport_loss, 0.0, 1.0))
        normalized_trades = sorted(
            (
                {
                    "seller_id": int(trade["seller_id"]),
                    "buyer_id": int(trade["buyer_id"]),
                    "quantity": max(0.0, float(trade["quantity"])),
                    "price": float(trade.get("price", 0.0)),
                }
                for trade in trades
                if float(trade.get("quantity", 0.0)) > EPS
            ),
            key=lambda item: (
                item["seller_id"],
                item["buyer_id"],
                item["price"],
                item["quantity"],
            ),
        )

        routed = []
        h2_vehicle_flow = {edge: 0.0 for edge in self.edge_ids}
        for trade in normalized_trades:
            buyer = trade["buyer_id"]
            rank, path = self.choose_route(
                trade["seller_id"], buyer, actions[buyer]
            )
            edges = self._path_edges(path)
            vehicle_equivalent = (
                trade["quantity"] / self.lhv_h2 / self.truck_capacity_kg
            )
            for edge in edges:
                h2_vehicle_flow[edge] += vehicle_equivalent
            routed.append((trade, rank, path, edges, vehicle_equivalent))

        background = self.background_utilization(dispatch_t)
        utilization = {
            edge: float(background[edge] + h2_vehicle_flow[edge] / self.edge_capacity)
            for edge in self.edge_ids
        }
        self.last_background_utilization = dict(background)
        self.last_edge_utilization = dict(utilization)

        shipments = []
        for sequence, (trade, rank, path, edges, vehicle_equivalent) in enumerate(routed):
            eta, congestion_delay, nominal_eta = self._route_eta(path, utilization)
            gross = float(trade["quantity"])
            # 平坦损耗 + 按在途小时累计的 boil-off 损耗 (设计规格 1d)。
            effective_loss_ratio = float(
                np.clip(loss_ratio + self.transit_loss_per_hour * float(eta), 0.0, 1.0)
            )
            loss = gross * effective_loss_ratio
            net = gross - loss
            shipments.append(
                {
                    "shipment_id": (
                        f"{int(dispatch_t)}:{trade['seller_id']}:"
                        f"{trade['buyer_id']}:{sequence}"
                    ),
                    "buyer_id": int(trade["buyer_id"]),
                    "seller_id": int(trade["seller_id"]),
                    "gross_quantity": gross,
                    "loss_quantity": loss,
                    "net_quantity": net,
                    "quantity": net,
                    "price": float(trade["price"]),
                    "route_rank": int(rank),
                    "route_id": f"{trade['seller_id']}->{trade['buyer_id']}:r{rank}",
                    "path": [int(node) for node in path],
                    "edge_ids": [f"{source}->{target}" for source, target in edges],
                    "vehicle_equivalent": float(vehicle_equivalent),
                    "nominal_eta": int(nominal_eta),
                    "congestion_delay": float(congestion_delay),
                    "eta": int(eta),
                    "dispatch_t": int(dispatch_t),
                    "deliver_at": int(dispatch_t) + int(eta),
                    "max_edge_utilization": max(
                        (float(utilization[edge]) for edge in edges), default=0.0
                    ),
                    "status": "pending",
                }
            )
        return shipments
