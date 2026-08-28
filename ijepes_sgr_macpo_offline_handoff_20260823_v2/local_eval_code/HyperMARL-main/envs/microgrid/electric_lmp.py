from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


IEEE33_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (15, 16), (16, 17), (1, 18), (18, 19), (19, 20), (20, 21), (2, 22),
    (22, 23), (23, 24), (5, 25), (25, 26), (26, 27), (27, 28), (28, 29),
    (29, 30), (30, 31), (31, 32),
]

IEEE33_R = [
    0.0922, 0.4930, 0.3660, 0.3811, 0.8190, 0.1872, 1.7114, 1.0300,
    1.0440, 0.1966, 0.3744, 1.4680, 0.5416, 0.5910, 0.7463, 1.2890,
    0.7320, 0.1640, 1.5042, 0.4095, 0.7089, 0.4512, 0.8980, 0.8960,
    0.2030, 0.2842, 1.0590, 0.8042, 0.5075, 0.9744, 0.3105, 0.3410,
]

IEEE33_X = [
    0.0470, 0.2511, 0.1864, 0.1941, 0.7070, 0.6188, 1.2351, 0.7400,
    0.7400, 0.0650, 0.1238, 1.1550, 0.7129, 0.5260, 0.5450, 1.7210,
    0.5740, 0.1565, 1.3554, 0.4784, 0.9373, 0.3083, 0.7091, 0.7011,
    0.1034, 0.1447, 0.9337, 0.7006, 0.2585, 0.9630, 0.3619, 0.5302,
]

IEEE33_LOAD_KW = [
    0.0, 100.0, 90.0, 120.0, 60.0, 60.0, 200.0, 200.0, 60.0, 60.0,
    45.0, 60.0, 60.0, 120.0, 60.0, 60.0, 60.0, 90.0, 90.0, 90.0,
    90.0, 90.0, 90.0, 420.0, 420.0, 60.0, 60.0, 60.0, 120.0, 200.0,
    150.0, 210.0, 60.0,
]


def _array(values: Any, size: int, default: float) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=np.float32).reshape(-1)
    if arr.size == size:
        return arr.astype(np.float32)
    if arr.size == 1:
        return np.full(size, float(arr[0]), dtype=np.float32)
    return np.full(size, float(default), dtype=np.float32)


def _network(config: Dict[str, Any], mode: str) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mode == "simple_lmp":
        topology = str(config.get("elec_lmp_simple_topology", "chain"))
        if topology == "star":
            edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
        else:
            edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        bus_count = 5
        slack = 0
        r = np.asarray(config.get("elec_lmp_simple_line_r", [0.08, 0.10, 0.12, 0.14]), dtype=np.float32)
        x = np.asarray(config.get("elec_lmp_simple_line_x", [0.04, 0.05, 0.06, 0.07]), dtype=np.float32)
        return bus_count, slack, np.asarray(edges, dtype=np.int64), _array(r, len(edges), 0.1), _array(x, len(edges), 0.05), np.zeros(bus_count, dtype=np.float32)
    bus_count = int(config.get("elec_lmp_bus_count", 33))
    slack = int(config.get("elec_lmp_slack_bus", 0))
    edges = np.asarray(config.get("elec_lmp_line_edges", IEEE33_EDGES), dtype=np.int64).reshape(-1, 2)
    r = _array(config.get("elec_lmp_line_r", IEEE33_R), edges.shape[0], 0.1)
    x = _array(config.get("elec_lmp_line_x", IEEE33_X), edges.shape[0], 0.05)
    bg = _array(config.get("elec_lmp_background_load_kw", IEEE33_LOAD_KW), bus_count, 0.0)
    return bus_count, slack, edges, r, x, bg


def _agent_buses(config: Dict[str, Any], mode: str, agent_num: int, bus_count: int) -> np.ndarray:
    if mode == "simple_lmp":
        default = [1, 2, 3, 4]
        values = config.get("elec_lmp_simple_agent_bus_indices", default)
    else:
        default = [4, 12, 23, 32]
        values = config.get("elec_lmp_agent_bus_indices", default)
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size != agent_num:
        arr = np.asarray(default[:agent_num], dtype=np.int64)
    if bool(config.get("elec_lmp_agent_bus_one_indexed", False)):
        arr = arr - 1
    return np.clip(arr, 0, bus_count - 1)


def _tree(bus_count: int, slack: int, edges: np.ndarray) -> Tuple[List[List[int]], np.ndarray, np.ndarray, List[List[int]]]:
    children = [[] for _ in range(bus_count)]
    parent = np.full(bus_count, -1, dtype=np.int64)
    parent_line = np.full(bus_count, -1, dtype=np.int64)
    for idx, (u, v) in enumerate(edges):
        u = int(u)
        v = int(v)
        if 0 <= u < bus_count and 0 <= v < bus_count:
            children[u].append(v)
            parent[v] = u
            parent_line[v] = idx
    paths = [[] for _ in range(bus_count)]
    for bus in range(bus_count):
        cur = bus
        seen = set()
        while cur != slack and cur >= 0 and cur not in seen:
            seen.add(cur)
            line = int(parent_line[cur])
            if line < 0:
                break
            paths[bus].append(line)
            cur = int(parent[cur])
        paths[bus] = list(reversed(paths[bus]))
    return children, parent, parent_line, paths


def _flows_from_load(bus_load: np.ndarray, children: List[List[int]], parent_line: np.ndarray, line_count: int) -> np.ndarray:
    flows = np.zeros(line_count, dtype=np.float32)

    def visit(bus: int) -> float:
        subtotal = float(bus_load[bus])
        for child in children[bus]:
            subtotal += visit(child)
        line = int(parent_line[bus])
        if line >= 0:
            flows[line] = subtotal
        return subtotal

    roots = [i for i, line in enumerate(parent_line) if line < 0]
    for root in roots:
        visit(int(root))
    return flows


def _sell_prices(buy: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    mode = str(config.get("elec_lmp_sell_mode", "ratio"))
    floor = float(config.get("elec_lmp_sell_price_min", 0.0))
    if mode == "symmetric":
        sell = buy.copy()
    elif mode == "fixed_spread":
        spread = float(config.get("elec_lmp_sell_spread", 0.15))
        sell = buy - spread
    else:
        ratio = float(config.get("elec_lmp_sell_ratio", 0.55))
        sell = buy * ratio
    return np.maximum(floor, sell).astype(np.float32)


def build_electric_price_tables(config: Dict[str, Any], profiles: Dict[str, np.ndarray], tou_buy: np.ndarray, tou_sell: np.ndarray) -> Dict[str, Any]:
    mode = str(config.get("elec_price_mode", "tou"))
    agent_num = int(config.get("num_agents", profiles["load_e"].shape[0]))
    episode_length = int(config.get("episode_length", profiles["load_e"].shape[1]))
    if mode == "tou":
        agent_buy = np.tile(np.asarray(tou_buy, dtype=np.float32).reshape(-1, 1), (1, agent_num))
        agent_sell = np.tile(np.asarray(tou_sell, dtype=np.float32).reshape(-1, 1), (1, agent_num))
        return {
            "mode": mode,
            "agent_bus_indices": np.arange(agent_num, dtype=np.int64),
            "node_buy_prices": agent_buy.copy(),
            "node_sell_prices": agent_sell.copy(),
            "agent_buy_prices": agent_buy,
            "agent_sell_prices": agent_sell,
            "line_loading": np.zeros((episode_length, 0), dtype=np.float32),
            "line_loading_max": np.zeros(episode_length, dtype=np.float32),
            "congestion_count": np.zeros(episode_length, dtype=np.float32),
            "slack_import": np.zeros(episode_length, dtype=np.float32),
            "price_spread": np.zeros(episode_length, dtype=np.float32),
            "status_code": np.zeros(episode_length, dtype=np.float32),
        }
    bus_count, slack, edges, r, _x, bg = _network(config, mode)
    agent_bus = _agent_buses(config, mode, agent_num, bus_count)
    children, _parent, parent_line, paths = _tree(bus_count, slack, edges)
    line_count = int(edges.shape[0])
    cap_default = float(config.get("elec_lmp_line_capacity_kw", 9000.0 if mode != "simple_lmp" else 12000.0))
    line_capacity = _array(config.get("elec_lmp_line_capacities_kw", None), line_count, cap_default)
    line_capacity = np.maximum(line_capacity, float(config.get("elec_lmp_line_capacity_min_kw", 1000.0)))
    bg_scale = float(config.get("elec_lmp_background_load_scale", 1.0 if mode != "simple_lmp" else 0.0))
    loss_coef = float(config.get("elec_lmp_loss_coef", 0.08 if mode != "simple_lmp" else 0.03))
    congestion_coef = float(config.get("elec_lmp_congestion_coef", 0.35 if mode != "simple_lmp" else 0.08))
    depth_coef = float(config.get("elec_lmp_depth_coef", 0.01 if mode != "simple_lmp" else 0.005))
    gen_credit_coef = float(config.get("elec_lmp_local_generation_credit_coef", 0.05))
    threshold = float(config.get("elec_lmp_congestion_threshold", 0.70))
    price_min = float(config.get("elec_lmp_price_min", 0.10))
    price_max = float(config.get("elec_lmp_price_max", 1.40 if mode != "simple_lmp" else 1.15))
    r_scale = max(float(np.mean(np.maximum(r, 1.0e-6))), 1.0e-6)
    aggregate_load = np.maximum(np.sum(profiles["load_e"], axis=0), 1.0)
    aggregate_ref = max(float(np.mean(aggregate_load)), 1.0)
    node_buy = np.zeros((episode_length, bus_count), dtype=np.float32)
    line_loading = np.zeros((episode_length, line_count), dtype=np.float32)
    line_loading_max = np.zeros(episode_length, dtype=np.float32)
    congestion_count = np.zeros(episode_length, dtype=np.float32)
    slack_import = np.zeros(episode_length, dtype=np.float32)
    status_code = np.ones(episode_length, dtype=np.float32)
    forecast_net = np.asarray(profiles["load_e"] - profiles["pv"] - profiles["wt"], dtype=np.float32)
    for t in range(episode_length):
        load_shape = float(aggregate_load[t] / aggregate_ref)
        bus_load = bg.astype(np.float32) * bg_scale * load_shape
        for agent_id, bus in enumerate(agent_bus):
            bus_load[int(bus)] += float(forecast_net[agent_id, t])
        flows = _flows_from_load(bus_load, children, parent_line, line_count)
        loading = np.abs(flows) / np.maximum(line_capacity, 1.0)
        line_loading[t] = loading.astype(np.float32)
        line_loading_max[t] = float(np.max(loading)) if loading.size else 0.0
        congestion_count[t] = float(np.sum(loading >= threshold))
        slack_import[t] = float(np.sum(bus_load))
        base = float(tou_buy[t])
        for bus in range(bus_count):
            adder = 0.0
            for depth, line in enumerate(paths[bus], start=1):
                load_ratio = float(loading[line])
                resist = float(r[line] / r_scale)
                loss = loss_coef * resist * load_ratio
                congestion = congestion_coef * max(0.0, load_ratio - threshold) / max(1.0e-6, 1.0 - threshold)
                adder += loss + congestion + depth_coef / max(depth, 1)
            if bus_load[bus] < 0.0:
                relief = gen_credit_coef * min(1.0, abs(float(bus_load[bus])) / max(aggregate_ref, 1.0))
                adder -= relief
            node_buy[t, bus] = np.clip(base * (1.0 + adder), price_min, price_max)
    node_sell = _sell_prices(node_buy, config)
    agent_buy = node_buy[:, agent_bus]
    agent_sell = node_sell[:, agent_bus]
    return {
        "mode": mode,
        "agent_bus_indices": agent_bus.astype(np.int64),
        "node_buy_prices": node_buy.astype(np.float32),
        "node_sell_prices": node_sell.astype(np.float32),
        "agent_buy_prices": agent_buy.astype(np.float32),
        "agent_sell_prices": agent_sell.astype(np.float32),
        "line_loading": line_loading.astype(np.float32),
        "line_loading_max": line_loading_max.astype(np.float32),
        "congestion_count": congestion_count.astype(np.float32),
        "slack_import": slack_import.astype(np.float32),
        "price_spread": (np.max(agent_buy, axis=1) - np.min(agent_buy, axis=1)).astype(np.float32),
        "status_code": status_code.astype(np.float32),
    }
