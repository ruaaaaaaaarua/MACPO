from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-6


def _valid_orders(orders: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int]]:
    normalized: List[Dict[str, Any]] = []
    agent_ids = set()
    for sequence, raw in enumerate(orders):
        side = raw.get("side")
        quantity = max(0.0, float(raw.get("quantity", 0.0)))
        if side not in {"buy", "sell"} or quantity <= EPS:
            continue
        agent_id = int(raw["agent_id"])
        agent_ids.add(agent_id)
        normalized.append({
            "agent_id": agent_id,
            "side": side,
            "price": float(raw.get("price", 0.0)),
            "quantity": quantity,
            "remaining": quantity,
            "sequence": int(sequence),
        })
    return normalized, sorted(agent_ids)


def _empty_stats(agent_ids: Sequence[int]) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]:
    buy_matched = {int(agent_id): 0.0 for agent_id in agent_ids}
    sell_matched = {int(agent_id): 0.0 for agent_id in agent_ids}
    buy_cost = {int(agent_id): 0.0 for agent_id in agent_ids}
    sell_revenue = {int(agent_id): 0.0 for agent_id in agent_ids}
    return buy_matched, sell_matched, buy_cost, sell_revenue


def _build_agent_results(agent_ids: Sequence[int], buy_matched: Dict[int, float], sell_matched: Dict[int, float], buy_cost: Dict[int, float], sell_revenue: Dict[int, float]) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for agent_id in sorted(int(i) for i in agent_ids):
        buy_qty = float(buy_matched.get(agent_id, 0.0))
        sell_qty = float(sell_matched.get(agent_id, 0.0))
        if buy_qty > EPS:
            results.append({
                "agent_id": agent_id,
                "side": "buy",
                "matched_quantity": buy_qty,
                "matched_price": float(buy_cost.get(agent_id, 0.0)) / buy_qty,
            })
        if sell_qty > EPS:
            results.append({
                "agent_id": agent_id,
                "side": "sell",
                "matched_quantity": sell_qty,
                "matched_price": float(sell_revenue.get(agent_id, 0.0)) / sell_qty,
            })
    return results


def _book_snapshot(orders: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    return [
        {
            "agent_id": int(order["agent_id"]),
            "price": float(order["price"]),
            "remaining_quantity": float(order["remaining"]),
        }
        for order in orders
        if float(order.get("remaining", 0.0)) > EPS
    ]


def _distance(buyer_id: int, seller_id: int, agent_locations: Optional[Sequence[float]], distance_matrix: Optional[Sequence[Sequence[float]]]) -> float:
    if distance_matrix is not None:
        matrix = np.asarray(distance_matrix, dtype=float)
        if matrix.ndim == 2 and buyer_id < matrix.shape[0] and seller_id < matrix.shape[1]:
            value = float(matrix[buyer_id, seller_id])
            return value if np.isfinite(value) and value >= 0.0 else 0.0
    if agent_locations is not None:
        locations = np.asarray(agent_locations, dtype=float).reshape(-1)
        if buyer_id < locations.size and seller_id < locations.size:
            return abs(float(locations[buyer_id]) - float(locations[seller_id]))
    return 0.0


def _trade_price(buy_price: float, sell_price: float, network_fee: float, price_rule: str) -> float:
    rule = str(price_rule)
    if rule == "buyer_bid":
        return buy_price
    if rule == "seller_ask":
        return sell_price
    if rule == "midpoint":
        return 0.5 * (buy_price + sell_price)
    surplus = max(0.0, buy_price - sell_price - network_fee)
    return sell_price + 0.5 * surplus


def _pair_summary(trades: Iterable[Dict[str, Any]], agent_count: int, agent_locations: Optional[Sequence[float]], distance_matrix: Optional[Sequence[Sequence[float]]], distance_fee_coef: float) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray, np.ndarray, float, float]:
    quantity_matrix = np.zeros((agent_count, agent_count), dtype=np.float32)
    value_matrix = np.zeros((agent_count, agent_count), dtype=np.float32)
    distance_out = np.zeros((agent_count, agent_count), dtype=np.float32)
    pair_acc: Dict[Tuple[int, int], Dict[str, float]] = {}
    total_fee = 0.0
    total_distance_quantity = 0.0
    total_quantity = 0.0
    for trade in trades:
        buyer_id = int(trade["buyer_id"])
        seller_id = int(trade["seller_id"])
        if not (0 <= buyer_id < agent_count and 0 <= seller_id < agent_count):
            continue
        quantity = float(trade.get("quantity", 0.0))
        if quantity <= EPS:
            continue
        price = float(trade.get("price", 0.0))
        distance = float(trade.get("distance", _distance(buyer_id, seller_id, agent_locations, distance_matrix)))
        network_fee = float(trade.get("network_fee", max(0.0, distance_fee_coef) * distance))
        value = price * quantity
        fee_value = network_fee * quantity
        quantity_matrix[seller_id, buyer_id] += quantity
        value_matrix[seller_id, buyer_id] += value
        distance_out[seller_id, buyer_id] = distance
        key = (seller_id, buyer_id)
        acc = pair_acc.setdefault(key, {
            "seller_id": float(seller_id),
            "buyer_id": float(buyer_id),
            "quantity": 0.0,
            "value": 0.0,
            "network_fee_value": 0.0,
            "trade_count": 0.0,
            "distance": distance,
        })
        acc["quantity"] += quantity
        acc["value"] += value
        acc["network_fee_value"] += fee_value
        acc["trade_count"] += 1.0
        acc["distance"] = distance
        total_fee += fee_value
        total_distance_quantity += distance * quantity
        total_quantity += quantity
    summaries: List[Dict[str, float]] = []
    for (seller_id, buyer_id), acc in sorted(pair_acc.items()):
        quantity = max(float(acc["quantity"]), EPS)
        summaries.append({
            "seller_id": int(seller_id),
            "buyer_id": int(buyer_id),
            "quantity": float(acc["quantity"]),
            "value": float(acc["value"]),
            "weighted_average_price": float(acc["value"]) / quantity,
            "network_fee_value": float(acc["network_fee_value"]),
            "average_network_fee": float(acc["network_fee_value"]) / quantity,
            "trade_count": int(acc["trade_count"]),
            "distance": float(acc["distance"]),
        })
    mean_distance = total_distance_quantity / total_quantity if total_quantity > EPS else 0.0
    return summaries, quantity_matrix, value_matrix, distance_out, total_fee, mean_distance


def summarize_pairwise_trades(trades: Iterable[Dict[str, Any]], agent_count: int, agent_locations: Optional[Sequence[float]] = None, distance_matrix: Optional[Sequence[Sequence[float]]] = None, distance_fee_coef: float = 0.0) -> Dict[str, Any]:
    pair_summary, quantity_matrix, value_matrix, distance_out, network_fee_total, mean_distance = _pair_summary(
        trades, agent_count, agent_locations, distance_matrix, distance_fee_coef
    )
    total_quantity = float(np.sum(quantity_matrix))
    total_value = float(np.sum(value_matrix))
    return {
        "pair_summary": pair_summary,
        "pair_quantity_matrix": quantity_matrix.astype(float).tolist(),
        "pair_value_matrix": value_matrix.astype(float).tolist(),
        "pair_distance_matrix": distance_out.astype(float).tolist(),
        "pair_count": len(pair_summary),
        "network_fee_total": float(network_fee_total),
        "mean_trade_distance": float(mean_distance),
        "weighted_average_price": total_value / total_quantity if total_quantity > EPS else 0.0,
    }


def run_bilateral_p2p_market(orders: Iterable[Dict[str, Any]], default_price: float = 0.0, agent_count: Optional[int] = None, agent_locations: Optional[Sequence[float]] = None, distance_matrix: Optional[Sequence[Sequence[float]]] = None, distance_fee_coef: float = 0.0, max_distance: Optional[float] = None, price_rule: str = "split_surplus") -> Dict[str, Any]:
    normalized, agent_ids = _valid_orders(orders)
    if agent_count is None:
        agent_count = max(agent_ids) + 1 if agent_ids else 0
    buy_matched, sell_matched, buy_cost, sell_revenue = _empty_stats(agent_ids)
    buys = [order for order in normalized if order["side"] == "buy"]
    sells = [order for order in normalized if order["side"] == "sell"]
    candidates: List[Tuple[float, float, int, int, Dict[str, Any], Dict[str, Any], float, float, float]] = []
    fee_coef = max(0.0, float(distance_fee_coef))
    max_dist = None if max_distance is None or float(max_distance) <= 0.0 else float(max_distance)
    for buy in buys:
        for sell in sells:
            buyer_id = int(buy["agent_id"])
            seller_id = int(sell["agent_id"])
            distance = _distance(buyer_id, seller_id, agent_locations, distance_matrix)
            if max_dist is not None and distance > max_dist + EPS:
                continue
            network_fee = fee_coef * distance
            gross_spread = float(buy["price"]) - float(sell["price"])
            net_spread = gross_spread - network_fee
            if net_spread + EPS < 0.0:
                continue
            candidates.append((
                -net_spread,
                distance,
                int(buy["sequence"]),
                int(sell["sequence"]),
                buy,
                sell,
                network_fee,
                gross_spread,
                net_spread,
            ))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    trades: List[Dict[str, float]] = []
    clearing_price = float(default_price)
    for _score, distance, _buy_sequence, _sell_sequence, buy, sell, network_fee, gross_spread, net_spread in candidates:
        if buy["remaining"] <= EPS or sell["remaining"] <= EPS:
            continue
        quantity = min(float(buy["remaining"]), float(sell["remaining"]))
        if quantity <= EPS:
            continue
        price = _trade_price(float(buy["price"]), float(sell["price"]), float(network_fee), price_rule)
        buyer_id = int(buy["agent_id"])
        seller_id = int(sell["agent_id"])
        trades.append({
            "buyer_id": buyer_id,
            "seller_id": seller_id,
            "price": float(price),
            "quantity": float(quantity),
            "buyer_bid": float(buy["price"]),
            "seller_ask": float(sell["price"]),
            "distance": float(distance),
            "network_fee": float(network_fee),
            "gross_spread": float(gross_spread),
            "net_spread": float(net_spread),
        })
        buy_matched[buyer_id] += quantity
        sell_matched[seller_id] += quantity
        buy_cost[buyer_id] += price * quantity
        sell_revenue[seller_id] += price * quantity
        clearing_price = float(price)
        buy["remaining"] -= quantity
        sell["remaining"] -= quantity
    open_buy_orders = sorted(_book_snapshot(buys), key=lambda order: (-order["price"], order["agent_id"]))
    open_sell_orders = sorted(_book_snapshot(sells), key=lambda order: (order["price"], order["agent_id"]))
    pair_diag = summarize_pairwise_trades(
        trades,
        int(agent_count),
        agent_locations=agent_locations,
        distance_matrix=distance_matrix,
        distance_fee_coef=fee_coef,
    )
    return {
        "trades": trades,
        "clearing_price": clearing_price,
        "buy_matched": buy_matched,
        "sell_matched": sell_matched,
        "buy_cost": buy_cost,
        "sell_revenue": sell_revenue,
        "agent_results": _build_agent_results(agent_ids, buy_matched, sell_matched, buy_cost, sell_revenue),
        "open_buy_orders": open_buy_orders,
        "open_sell_orders": open_sell_orders,
        "pair_summary": pair_diag["pair_summary"],
        "pair_quantity_matrix": pair_diag["pair_quantity_matrix"],
        "pair_value_matrix": pair_diag["pair_value_matrix"],
        "pair_distance_matrix": pair_diag["pair_distance_matrix"],
        "pair_count": pair_diag["pair_count"],
        "network_fee_total": pair_diag["network_fee_total"],
        "mean_trade_distance": pair_diag["mean_trade_distance"],
        "weighted_average_price": pair_diag["weighted_average_price"],
        "mechanism": "p2p_bilateral",
    }
