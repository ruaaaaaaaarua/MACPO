"""Continuous double auction utilities used by the environment."""

from bisect import insort


EPS = 1e-6


def _empty_stats(agent_ids):
    buy_matched = {agent_id: 0.0 for agent_id in agent_ids}
    sell_matched = {agent_id: 0.0 for agent_id in agent_ids}
    buy_cost = {agent_id: 0.0 for agent_id in agent_ids}
    sell_revenue = {agent_id: 0.0 for agent_id in agent_ids}
    return buy_matched, sell_matched, buy_cost, sell_revenue


def _insert_buy_order(book, order):
    insort(book, (-order["price"], order["sequence"], order))


def _insert_sell_order(book, order):
    insort(book, (order["price"], order["sequence"], order))


def _build_agent_results(agent_ids, buy_matched, sell_matched, buy_cost, sell_revenue):
    results = []
    for agent_id in sorted(agent_ids):
        buy_qty = buy_matched.get(agent_id, 0.0)
        sell_qty = sell_matched.get(agent_id, 0.0)
        if buy_qty > EPS:
            results.append({
                "agent_id": agent_id,
                "side": "buy",
                "matched_quantity": buy_qty,
                "matched_price": buy_cost[agent_id] / buy_qty,
            })
        if sell_qty > EPS:
            results.append({
                "agent_id": agent_id,
                "side": "sell",
                "matched_quantity": sell_qty,
                "matched_price": sell_revenue[agent_id] / sell_qty,
            })
    return results


def _book_snapshot(book):
    return [
        {
            "agent_id": order["agent_id"],
            "price": order["price"],
            "remaining_quantity": order["remaining"],
        }
        for _, _, order in book
        if order["remaining"] > EPS
    ]


def run_continuous_double_auction(orders, default_price=0.0):
    """Run a price-time-priority continuous double auction.

    Args:
        orders: list of dict with keys:
            agent_id: int
            side: "buy" or "sell"
            price: float
            quantity: float
        default_price: fallback price used when no trade occurs.

    Returns:
        dict with trades, last clearing price, aggregated matched quantities,
        costs/revenues, per-agent matched results, and residual order books.
    """
    agent_ids = {order["agent_id"] for order in orders}
    buy_matched, sell_matched, buy_cost, sell_revenue = _empty_stats(agent_ids)

    buy_book = []
    sell_book = []
    trades = []
    clearing_price = float(default_price)

    for sequence, raw_order in enumerate(orders):
        quantity = max(0.0, float(raw_order["quantity"]))
        side = raw_order["side"]
        if side not in {"buy", "sell"} or quantity <= EPS:
            continue

        order = {
            "agent_id": int(raw_order["agent_id"]),
            "side": side,
            "price": float(raw_order["price"]),
            "quantity": quantity,
            "remaining": quantity,
            "sequence": sequence,
        }

        if side == "buy":
            while order["remaining"] > EPS and sell_book:
                _, _, resting_order = sell_book[0]
                if order["price"] + EPS < resting_order["price"]:
                    break

                trade_qty = min(order["remaining"], resting_order["remaining"])
                trade_price = resting_order["price"]
                trades.append({
                    "buyer_id": order["agent_id"],
                    "seller_id": resting_order["agent_id"],
                    "price": trade_price,
                    "quantity": trade_qty,
                })

                buy_matched[order["agent_id"]] += trade_qty
                sell_matched[resting_order["agent_id"]] += trade_qty
                buy_cost[order["agent_id"]] += trade_price * trade_qty
                sell_revenue[resting_order["agent_id"]] += trade_price * trade_qty
                clearing_price = trade_price

                order["remaining"] -= trade_qty
                resting_order["remaining"] -= trade_qty
                if resting_order["remaining"] <= EPS:
                    sell_book.pop(0)

            if order["remaining"] > EPS:
                _insert_buy_order(buy_book, order)
        else:
            while order["remaining"] > EPS and buy_book:
                _, _, resting_order = buy_book[0]
                if resting_order["price"] + EPS < order["price"]:
                    break

                trade_qty = min(order["remaining"], resting_order["remaining"])
                trade_price = resting_order["price"]
                trades.append({
                    "buyer_id": resting_order["agent_id"],
                    "seller_id": order["agent_id"],
                    "price": trade_price,
                    "quantity": trade_qty,
                })

                buy_matched[resting_order["agent_id"]] += trade_qty
                sell_matched[order["agent_id"]] += trade_qty
                buy_cost[resting_order["agent_id"]] += trade_price * trade_qty
                sell_revenue[order["agent_id"]] += trade_price * trade_qty
                clearing_price = trade_price

                order["remaining"] -= trade_qty
                resting_order["remaining"] -= trade_qty
                if resting_order["remaining"] <= EPS:
                    buy_book.pop(0)

            if order["remaining"] > EPS:
                _insert_sell_order(sell_book, order)

    return {
        "trades": trades,
        "clearing_price": clearing_price,
        "buy_matched": buy_matched,
        "sell_matched": sell_matched,
        "buy_cost": buy_cost,
        "sell_revenue": sell_revenue,
        "agent_results": _build_agent_results(
            agent_ids, buy_matched, sell_matched, buy_cost, sell_revenue
        ),
        "open_buy_orders": _book_snapshot(buy_book),
        "open_sell_orders": _book_snapshot(sell_book),
    }


def run_cda_clearing(sellers, buyers, default_price=0.0):
    """Backward-compatible wrapper for the original single-market interface."""
    orders = []
    for seller in sellers:
        orders.append({
            "agent_id": seller["agent_id"],
            "side": "sell",
            "price": seller["price"],
            "quantity": seller["quantity"],
        })
    for buyer in buyers:
        orders.append({
            "agent_id": buyer["agent_id"],
            "side": "buy",
            "price": buyer["price"],
            "quantity": buyer["quantity"],
        })
    return run_continuous_double_auction(orders, default_price=default_price)
