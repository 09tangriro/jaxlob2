"""Module containing all functions to manipulate the orderbook.

To follow the functional programming paradigm of JAX, the functions of the
orderbook are not put into an object but are left standalone.

There are 3 types of datastructures manipulated by the functions: orders, messages, and trades.
Orders are the existing limit orders in the orderbook, messages are the incoming orders, and trades are the executed trades.

Orders = [price, quantity, agent_id, order_id, time_s, time_ns]
Messages = [type, side, price, quantity, agent_id, order_id, time_s, time_ns]
Trades = [order_price, order_quantity, order_agent_id, message_agent_id, time_s, time_ns, order_side]
"""

from functools import partial
from typing import Tuple

import jax
from jax import numpy as jnp

INIT_ORDER_ID = 0
MARKET_AGENT_ID = 0
MAX_INT = 2_147_483_647  # max 32 bit int


class MessageType:
    LIMIT = 1
    CANCEL = 2
    DELETE = 3
    MARKET = 4


class OrderSide:
    ASK = -1
    BID = 1


class Order:
    PRICE_IDX = 0
    QUANTITY_IDX = 1
    AGENT_ID_IDX = 2
    ORDER_ID_IDX = 3
    TIME_S_IDX = 4
    TIME_NS_IDX = 5


class Message:
    TYPE_IDX = 0
    SIDE_IDX = 1
    PRICE_IDX = 2
    QUANTITY_IDX = 3
    AGENT_ID_IDX = 4
    ORDER_ID_IDX = 5
    TIME_S_IDX = 6
    TIME_NS_IDX = 7


class Trade:
    ORDER_PRICE_IDX = 0
    ORDER_QUANTITY_IDX = 1
    ORDER_AGENT_ID_IDX = 2
    MESSAGE_AGENT_ID_IDX = 3
    TIME_S_IDX = 4
    TIME_NS_IDX = 5
    ORDER_SIDE_IDX = 6


############### ADD AND REMOVE ###############
@jax.jit
def _remove_zero_neg_quant(orders: jnp.ndarray) -> jnp.ndarray:
    """Removes orders with zero or negative quantity from the orderbook.

    :param orders: The orderbook side to remove orders from.
    :return: The updated orderbook side.
    """
    return jnp.where(
        (orders[:, Order.QUANTITY_IDX] <= 0).reshape((orders.shape[0], 1)),
        (jnp.ones(orders.shape) * -1).astype(jnp.int32),
        orders,
    )


@jax.jit
def add_order(orders: jnp.ndarray, msg: jnp.ndarray) -> jnp.ndarray:
    """Add a new order to the orderbook.

    :param orders: The orderbook side to add the order to.
    :param msg: The dict message containing the order to add.
    :return: The updated orderbook side.
    """
    emptyidx = jnp.where(orders == -1, size=1, fill_value=-1)[0]
    orders = (
        orders.at[emptyidx, :]
        .set(
            jnp.array(
                [
                    msg[Message.PRICE_IDX],
                    jnp.maximum(0, msg[Message.QUANTITY_IDX]),
                    msg[Message.AGENT_ID_IDX],
                    msg[Message.ORDER_ID_IDX],
                    msg[Message.TIME_S_IDX],
                    msg[Message.TIME_NS_IDX],
                ]
            )
        )
        .astype(jnp.int32)
    )
    return _remove_zero_neg_quant(orders)


@jax.jit
def cancel_order(orders: jnp.ndarray, msg: jnp.ndarray) -> jnp.ndarray:
    """Cancel an order from the orderbook.

    :param orders: The orderbook side to remove the order from.
    :param msg: The dict message containing the order to cancel.
    """

    def get_init_id_match(orders: jnp.ndarray, msg: jnp.ndarray) -> jnp.ndarray:
        init_id_match = (orders[:, Order.PRICE_IDX] == msg[Message.PRICE_IDX]) & (
            orders[:, Order.ORDER_ID_IDX] <= INIT_ORDER_ID
        )
        idx = jnp.where(init_id_match, size=1, fill_value=-1)[0][0]
        return idx

    # TODO: also check for price here?
    oid_match = orders[:, Order.ORDER_ID_IDX] == msg[Message.ORDER_ID_IDX]
    idx = jnp.where(oid_match, size=1, fill_value=-1)[0][0]
    idx = jax.lax.cond(idx == -1, get_init_id_match, lambda a, b: idx, orders, msg)
    orders = orders.at[idx, Order.QUANTITY_IDX].set(
        orders[idx, Order.QUANTITY_IDX] - msg[Message.QUANTITY_IDX]
    )
    return _remove_zero_neg_quant(orders)


############### MATCHING FUNCTIONS ###############
@jax.jit
def _get_top_bid_order_idx(bids: jnp.ndarray) -> int:
    """Return the index of the top bid order in the orderbook."""
    max_price = jnp.max(bids[:, Order.PRICE_IDX], axis=0)
    times = jnp.where(
        bids[:, Order.PRICE_IDX] == max_price, bids[:, Order.TIME_S_IDX], MAX_INT
    )
    min_time_s = jnp.min(times, axis=0)
    times_ns = jnp.where(times == min_time_s, bids[:, Order.TIME_NS_IDX], MAX_INT)
    min_time_ns = jnp.min(times_ns, axis=0)
    return jnp.where(times_ns == min_time_ns, size=1, fill_value=-1)[0]


@jax.jit
def _get_top_ask_order_idx(asks: jnp.ndarray) -> int:
    """Return the index of the top ask order in the orderbook."""
    prices = asks[:, Order.PRICE_IDX]
    prices = jnp.where(prices == -1, MAX_INT, prices)
    min_price = jnp.min(prices)
    times = jnp.where(
        asks[:, Order.PRICE_IDX] == min_price, asks[:, Order.TIME_S_IDX], MAX_INT
    )
    minTime_s = jnp.min(times, axis=0)
    times_ns = jnp.where(times == minTime_s, asks[:, Order.TIME_NS_IDX], MAX_INT)
    minTime_ns = jnp.min(times_ns, axis=0)
    return jnp.where(times_ns == minTime_ns, size=1, fill_value=-1)[0]


@jax.jit
def match_order(
    top_order_idx: int,
    orders: jnp.ndarray,
    quantity: int,
    price: int,
    trades: jnp.ndarray,
    agent_id: int,
    time: int,
    time_ns: int,
    side: int,
) -> Tuple[jnp.ndarray, int, int, jnp.ndarray, int, int, int, int]:
    """Match a given order against the top order in the orderbook.

    :param top_order_idx: The index of the top order in the orderbook.
    :param orders: The orderbook side to match the order against.
    :param quantity: The quantity of the order to match.
    :param price: The price of the order to match.
    :param trades: The trade array to append the trade to.
    :param agent_id: The agent_id ID.
    :param time: The time of the trade.
    :param time_ns: The time in nanoseconds of the trade.
    :param side: The side of the order to match.
    """
    newquant = jnp.maximum(
        0, orders[top_order_idx, Order.QUANTITY_IDX] - quantity
    )  # Could theoretically be removed as an operation because the removeZeroQuand func also removes negatives.
    quantity = quantity - orders[top_order_idx, Order.QUANTITY_IDX]
    quantity = quantity.astype(jnp.int32)
    emptyidx = jnp.where(trades == -1, size=1, fill_value=-1)[0]
    trades = trades.at[emptyidx, :].set(
        jnp.array(
            [
                orders[top_order_idx, Order.PRICE_IDX],
                orders[top_order_idx, Order.QUANTITY_IDX] - newquant,
                orders[top_order_idx, Order.AGENT_ID_IDX],
                [agent_id],
                [time],
                [time_ns],
                [side],
            ]
        ).transpose()
    )
    orders = _remove_zero_neg_quant(orders.at[top_order_idx, 1].set(newquant))
    return (
        orders.astype(jnp.int32),
        jnp.squeeze(quantity),
        price,
        trades,
        agent_id,
        time,
        time_ns,
        side,
    )


@jax.jit
def _match_against_bid_orders(
    bids: jnp.ndarray,
    quantity: int,
    price: int,
    trades: jnp.ndarray,
    agent_id: int,
    time: int,
    time_ns: int,
) -> Tuple[jnp.ndarray, int, jnp.ndarray]:
    def _check_before_matching_bid(data_tuple):
        top_order_idx, bids, quantity, price, _, _, _, _, _ = data_tuple
        res = (
            (bids[top_order_idx, Order.PRICE_IDX] >= price)
            & (quantity > 0)
            & (bids[top_order_idx, Order.PRICE_IDX] != -1)
        )
        return jnp.squeeze(res)

    def _match_bid_order(data_tuple):
        return _get_top_bid_order_idx(data_tuple[1]), *match_order(*data_tuple)

    top_order_idx = _get_top_bid_order_idx(bids)
    top_order_idx, bids, quantity, price, trades, _, _, _, _ = jax.lax.while_loop(
        _check_before_matching_bid,
        _match_bid_order,
        (
            top_order_idx,
            bids,
            quantity,
            price,
            trades,
            agent_id,
            time,
            time_ns,
            OrderSide.BID,
        ),
    )
    return bids, quantity, trades


@jax.jit
def _match_against_ask_orders(
    asks: jnp.ndarray,
    quantity: int,
    price: int,
    trades: jnp.ndarray,
    agent_id: int,
    time: int,
    time_ns: int,
) -> Tuple[jnp.ndarray, int, jnp.ndarray]:
    def _check_before_matching_ask(data_tuple):
        top_order_idx, asks, quantity, price, _, _, _, _, _ = data_tuple
        res = (
            (asks[top_order_idx, Order.PRICE_IDX] <= price)
            & (quantity > 0)
            & (asks[top_order_idx, Order.PRICE_IDX] != -1)
        )
        return jnp.squeeze(res)

    def _match_ask_order(data_tuple):
        return _get_top_ask_order_idx(data_tuple[1]), *match_order(*data_tuple)

    top_order_idx = _get_top_ask_order_idx(asks)
    top_order_idx, asks, quantity, price, trades, _, _, _, _ = jax.lax.while_loop(
        _check_before_matching_ask,
        _match_ask_order,
        (
            top_order_idx,
            asks,
            quantity,
            price,
            trades,
            agent_id,
            time,
            time_ns,
            OrderSide.ASK,
        ),
    )
    return asks, quantity, trades


########Type Functions#############


def do_nothing(
    msg: jnp.ndarray, asks: jnp.ndarray, bids: jnp.ndarray, trades: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return asks, bids, trades


def bid_lim(
    msg: jnp.ndarray, asks: jnp.ndarray, bids: jnp.ndarray, trades: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # match with asks side
    # add remainder to bids side
    asks, leftover_quantity, trades = _match_against_ask_orders(
        asks,
        msg[Message.QUANTITY_IDX],
        msg[Message.PRICE_IDX],
        trades,
        msg[Message.AGENT_ID_IDX],
        msg[Message.TIME_S_IDX],
        msg[Message.TIME_NS_IDX],
    )
    msg = msg.at[Message.QUANTITY_IDX].set(leftover_quantity)
    bids = add_order(bids, msg)
    return asks, bids, trades


def bid_cancel(
    msg: jnp.ndarray, asks: jnp.ndarray, bids: jnp.ndarray, trades: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return asks, cancel_order(bids, msg), trades


def bid_mkt(
    msg: jnp.ndarray, asks: jnp.ndarray, bids: jnp.ndarray, trades: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    msg = msg.at[Message.PRICE_IDX].set(MAX_INT)
    asks, _, trades = _match_against_ask_orders(
        asks,
        msg[Message.QUANTITY_IDX],
        msg[Message.PRICE_IDX],
        trades,
        msg[Message.AGENT_ID_IDX],
        msg[Message.TIME_S_IDX],
        msg[Message.TIME_NS_IDX],
    )
    return asks, bids, trades


def ask_lim(
    msg: jnp.ndarray, asks: jnp.ndarray, bids: jnp.ndarray, trades: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    bids, leftover_quantity, trades = _match_against_bid_orders(
        bids,
        msg[Message.QUANTITY_IDX],
        msg[Message.PRICE_IDX],
        trades,
        msg[Message.AGENT_ID_IDX],
        msg[Message.TIME_S_IDX],
        msg[Message.TIME_NS_IDX],
    )
    msg = msg.at[Message.QUANTITY_IDX].set(leftover_quantity)
    asks = add_order(asks, msg)
    return asks, bids, trades


def ask_cancel(
    msg: jnp.ndarray, asks: jnp.ndarray, bids: jnp.ndarray, trades: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return cancel_order(asks, msg), bids, trades


def ask_mkt(
    msg: jnp.ndarray, asks: jnp.ndarray, bids: jnp.ndarray, trades: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    msg = msg.at[Message.PRICE_IDX].set(0)
    bids, _, trades = _match_against_bid_orders(
        bids,
        msg[Message.QUANTITY_IDX],
        msg[Message.PRICE_IDX],
        trades,
        msg[Message.AGENT_ID_IDX],
        msg[Message.TIME_S_IDX],
        msg[Message.TIME_NS_IDX],
    )
    return asks, bids, trades


############### MAIN BRANCHING FUNCS ###############


@jax.jit
def cond_type_side(carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], x: jnp.ndarray):
    asks, bids, trades = carry
    msg = x
    # jax.debug.breakpoint()
    # jax.debug.print("Askside before is \n {}",asks)
    side = msg[Message.SIDE_IDX]
    msg_type = msg[Message.TYPE_IDX]
    index = (
        (
            ((side == OrderSide.ASK) & (msg_type == MessageType.LIMIT))
            | ((side == OrderSide.BID) & (msg_type == MessageType.MARKET))
        )
        * 0
        + (
            ((side == OrderSide.BID) & (msg_type == MessageType.LIMIT))
            | ((side == OrderSide.ASK) & (msg_type == MessageType.MARKET))
        )
        * 1
        + (
            ((side == OrderSide.ASK) & (msg_type == MessageType.DELETE))
            | ((side == OrderSide.ASK) & (msg_type == MessageType.CANCEL))
        )
        * 2
        + (
            ((side == OrderSide.BID) & (msg_type == MessageType.DELETE))
            | ((side == OrderSide.BID) & (msg_type == MessageType.CANCEL))
        )
        * 3
    )
    # jax.debug.print("msg[side] {}", msg["side"])
    # jax.debug.print("msg[type] {}", msg["type"])
    # jax.debug.print("index is {}", index)
    asks, bids, trades = jax.lax.switch(
        index, (ask_lim, bid_lim, ask_cancel, bid_cancel), msg, asks, bids, trades
    )
    return (asks, bids, trades), 0


@jax.jit
def cond_type_side_save_states(
    carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], x: jnp.ndarray
):
    asks, bids, trades = carry
    msg = x
    # jax.debug.breakpoint()
    # jax.debug.print("Askside before is \n {}",askside)
    side = msg[Message.SIDE_IDX]
    msg_type = msg[Message.TYPE_IDX]
    index = (
        (
            ((side == OrderSide.ASK) & (msg_type == MessageType.LIMIT))
            | ((side == OrderSide.BID) & (msg_type == MessageType.MARKET))
        )
        * 0
        + (
            ((side == OrderSide.BID) & (msg_type == MessageType.LIMIT))
            | ((side == OrderSide.ASK) & (msg_type == MessageType.MARKET))
        )
        * 1
        + (
            ((side == OrderSide.ASK) & (msg_type == MessageType.DELETE))
            | ((side == OrderSide.ASK) & (msg_type == MessageType.CANCEL))
        )
        * 2
        + (
            ((side == OrderSide.BID) & (msg_type == MessageType.DELETE))
            | ((side == OrderSide.BID) & (msg_type == MessageType.CANCEL))
        )
        * 3
    )
    asks, bids, trades = jax.lax.switch(
        index, (ask_lim, bid_lim, ask_cancel, bid_cancel), msg, asks, bids, trades
    )
    # jax.debug.print("Askside after is \n {}",ask)
    return (asks, bids, trades), (asks, bids, trades)


@jax.jit
def cond_type_side_save_bidask(
    carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], x: jnp.ndarray
):
    asks, bids, trades = carry
    msg = x
    # jax.debug.breakpoint()
    # jax.debug.print("Askside before is \n {}",askside)
    side = msg[Message.SIDE_IDX]
    msg_type = msg[Message.TYPE_IDX]
    index = (
        (
            ((side == OrderSide.ASK) & (msg_type == MessageType.LIMIT))
            | ((side == OrderSide.BID) & (msg_type == MessageType.MARKET))
        )
        * 0
        + (
            ((side == OrderSide.BID) & (msg_type == MessageType.LIMIT))
            | ((side == OrderSide.ASK) & (msg_type == MessageType.MARKET))
        )
        * 1
        + (
            ((side == OrderSide.ASK) & (msg_type == MessageType.DELETE))
            | ((side == OrderSide.ASK) & (msg_type == MessageType.CANCEL))
        )
        * 2
        + (
            ((side == OrderSide.BID) & (msg_type == MessageType.DELETE))
            | ((side == OrderSide.BID) & (msg_type == MessageType.CANCEL))
        )
        * 3
    )
    asks, bids, trades = jax.lax.switch(
        index, (ask_lim, bid_lim, ask_cancel, bid_cancel), msg, asks, bids, trades
    )
    # jax.debug.print("Askside after is \n {}",ask)
    # jax.debug.breakpoint()
    return (asks, bids, trades), get_best_bid_ask_incl_quants(asks, bids)


vcond_type_side = jax.vmap(cond_type_side, (0, 0, 0, 0))

############### SCAN FUNCTIONS ###############


def process_messages(
    messages: jnp.ndarray, asks: jnp.ndarray, bids: jnp.ndarray, trades: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Process a batch of messages through the orderbook.

    :param messages: The batch of messages to process.
    :param asks: The ask side of the orderbook.
    :param bids: The bid side of the orderbook.
    :param trades: The trade array.

    :return asks: The updated ask side.
    :return bids: The updated bid side.
    :return trades: The updated trade array.
    """
    (asks, bids, trades), _ = jax.lax.scan(
        cond_type_side, init=(asks, bids, trades), xs=messages
    )
    return asks, bids, trades


def process_messages_save_states(
    messages: jnp.ndarray,
    asks: jnp.ndarray,
    bids: jnp.ndarray,
    trades: jnp.ndarray,
    messages_per_step: int,
):
    """Process a batch of messages through the orderbook, saving the book states between steps.

    :param messages: The batch of messages to process.
    :param asks: The ask side of the orderbook.
    :param bids: The bid side of the orderbook.
    :param trades: The trade array.
    :param messages_per_step: The number of data messages being processed.

    :return last_asks: The updated ask side.
    :return last_bids: The updated bid side.
    :return last_trades: The updated trade array.
    :return all_asks: The ask side at each step.
    :return all_bids: The bid side at each step.
    :return all_trades: The trade array at each step.
    """
    # Will return the states for each of the processed messages, but only those from data to keep array size constant, and enabling variable #of actions (AutoCancel)
    last, all = jax.lax.scan(
        cond_type_side_save_states, init=(asks, bids, trades), xs=messages
    )
    return (all[0][-messages_per_step:], all[1][-messages_per_step:], last[2])


def process_messages_save_bidask(
    messages: jnp.ndarray,
    asks: jnp.ndarray,
    bids: jnp.ndarray,
    trades: jnp.ndarray,
    messages_per_step: int,
):
    """Process a batch of messages through the orderbook, saving the best bids and asks between steps.

    :param messages: The batch of messages to process.
    :param asks: The ask side of the orderbook.
    :param bids: The bid side of the orderbook.
    :param trades: The trade array.
    :param messages_per_step: The number of data messages being processed.

    :return asks: The updated ask side.
    :return bids: The updated bid side.
    :return trades: The updated trade array.
    :return best_asks: The best ask prices and quantities at each step.
    :return best_bids: The best bid prices and quantities at each step.
    """
    # Will return the states for each of the processed messages, but only those from data to keep array size constant, and enabling variable #of actions (AutoCancel)
    last, all = jax.lax.scan(
        cond_type_side_save_bidask, init=(asks, bids, trades), xs=messages
    )
    # jax.debug.breakpoint()
    return (
        last[0],
        last[1],
        last[2],
        all[0][-messages_per_step:],
        all[1][-messages_per_step:],
    )


################ GET CANCEL MESSAGES ################


# Obtain messages to cancel based on a given ID to lookup. Currently only used in the execution environment.
def get_num_agent_orders(orders: jnp.ndarray, agent_id: int) -> int:
    return jnp.sum(jnp.where(orders[:, Message.AGENT_ID_IDX] == agent_id, 1, 0)).astype(
        jnp.int32
    )


@partial(jax.jit, static_argnums=2)
def get_cancel_msgs(
    orders: jnp.ndarray, agent_id: int, size: int, side: int
) -> jnp.ndarray:
    orders = jnp.concatenate([orders, jnp.zeros((1, 6), dtype=jnp.int32)], axis=0)
    indeces_to_cancel = jnp.where(
        orders[:, Order.AGENT_ID_IDX] == agent_id, size=size, fill_value=-1
    )
    cancel_msgs = jnp.concatenate(
        [
            jnp.ones((1, size), dtype=jnp.int32) * 2,
            jnp.ones((1, size), dtype=jnp.int32) * side,
            orders[indeces_to_cancel, Order.PRICE_IDX],
            orders[indeces_to_cancel, Order.QUANTITY_IDX],
            orders[indeces_to_cancel, Order.AGENT_ID_IDX],
            orders[indeces_to_cancel, Order.ORDER_ID_IDX],
            orders[indeces_to_cancel, Order.TIME_S_IDX],
            orders[indeces_to_cancel, Order.TIME_NS_IDX],
        ],
        axis=0,
    ).transpose()
    return cancel_msgs


###### Helper functions for getting information #######


@jax.jit
def get_quant_at_price(orders: jnp.ndarray, price: int) -> int:
    """Return the total quantity at a given price. If there is no order at the given price, return 0."""
    return jnp.sum(
        jnp.where(orders[:, Order.PRICE_IDX] == price, orders[:, Order.QUANTITY_IDX], 0)
    )


get_quant_at_prices = jax.vmap(get_quant_at_price, (None, 0), 0)


@jax.jit
def get_best_bid_ask(asks: jnp.ndarray, bids: jnp.ndarray) -> Tuple[int, int]:
    best_ask = jnp.min(
        jnp.where(asks[:, Order.PRICE_IDX] == -1, MAX_INT, asks[:, Order.PRICE_IDX])
    )
    best_bid = jnp.max(bids[:, Order.PRICE_IDX])
    # jax.debug.print("-----")
    # jax.debug.print("best_bid from [get_best_bid_and_ask] {}", best_bid)
    # jax.debug.print("bids {}", bids)
    return best_ask, best_bid


@jax.jit
def get_best_bid_ask_incl_quants(
    asks: jnp.ndarray, bids: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    best_ask, best_bid = get_best_bid_ask(asks, bids)
    best_ask_Q = jnp.sum(
        jnp.where(asks[:, Order.PRICE_IDX] == best_ask, asks[:, Order.QUANTITY_IDX], 0)
    )
    best_bid_Q = jnp.sum(
        jnp.where(bids[:, Order.PRICE_IDX] == best_bid, bids[:, Order.QUANTITY_IDX], 0)
    )
    best_ask = jnp.array([best_ask, best_ask_Q], dtype=jnp.int32)
    best_bid = jnp.array([best_bid, best_bid_Q], dtype=jnp.int32)
    return best_ask, best_bid


@jax.jit
def get_max_ask(asks: jnp.ndarray) -> int:
    """Return the highest ask price. If there is no ask, return -1."""
    return jnp.max(asks[:, Order.PRICE_IDX])


@partial(jax.jit, static_argnums=0)
def init_orders(num_orders: int = 100):
    return (jnp.ones((num_orders, 6)) * -1).astype(jnp.int32)


@partial(jax.jit, static_argnums=3)
def get_initial_messages(
    book: jnp.ndarray, idx_window: int, time: jnp.ndarray, book_depth: int
):
    data = jnp.array(book[idx_window]).reshape(int(book_depth * 2), 2)
    newarr = jnp.zeros((int(book_depth * 2), 8), dtype=jnp.int32)
    init_msg = (
        newarr.at[:, Message.TYPE_IDX]
        .set(MessageType.LIMIT)  # type = limit order
        .at[0 : book_depth * 4 : 2, Message.SIDE_IDX]
        .set(OrderSide.ASK)  # side = ask
        .at[1 : book_depth * 4 : 2, Message.SIDE_IDX]
        .set(OrderSide.BID)  # side = bid
        .at[:, Message.PRICE_IDX]
        .set(data[:, Order.PRICE_IDX])  # price
        .at[:, Message.QUANTITY_IDX]
        .set(data[:, Order.QUANTITY_IDX])  # quantity
        .at[:, Message.AGENT_ID_IDX]
        .set(MARKET_AGENT_ID)  # agent_id
        .at[:, Message.ORDER_ID_IDX]
        .set(INIT_ORDER_ID - jnp.arange(0, book_depth * 2))  # order_id
        .at[:, Message.TIME_S_IDX]
        .set(time[0])  # time_s
        .at[:, Message.TIME_NS_IDX]
        .set(time[1])  # time_ns
    )
    return init_msg


@jax.jit
def get_initial_time(messages: jnp.ndarray, idx_window: int) -> jnp.ndarray:
    return messages[idx_window, 0, 0, -2:]


@jax.jit
def get_data_messages(
    messages: jnp.ndarray, idx_window: int, step_counter: int
) -> jnp.ndarray:
    messages = messages[idx_window, step_counter, :, :]
    return messages


@jax.jit
def filter_bid_messages(messages: jnp.ndarray) -> jnp.ndarray:
    is_bid = messages[:, Message.SIDE_IDX] == OrderSide.BID
    is_bid = jnp.expand_dims(is_bid, axis=1)
    return jnp.where(is_bid, messages, jnp.zeros_like(messages))


@jax.jit
def filter_limit_bid_messages(messages: jnp.ndarray) -> jnp.ndarray:
    is_limit_bid = (messages[:, Message.TYPE_IDX] == MessageType.LIMIT) & (
        messages[:, Message.SIDE_IDX] == OrderSide.BID
    )
    is_limit_bid = jnp.expand_dims(is_limit_bid, axis=1)
    return jnp.where(is_limit_bid, messages, jnp.zeros_like(messages))


@jax.jit
def filter_ask_messages(messages: jnp.ndarray) -> jnp.ndarray:
    is_ask = messages[:, Message.SIDE_IDX] == OrderSide.ASK
    is_ask = jnp.expand_dims(is_ask, axis=1)
    return jnp.where(is_ask, messages, jnp.zeros_like(messages))


@jax.jit
def filter_limit_ask_messages(messages: jnp.ndarray) -> jnp.ndarray:
    is_limit_ask = (messages[:, Message.TYPE_IDX] == MessageType.LIMIT) & (
        messages[:, Message.SIDE_IDX] == OrderSide.ASK
    )
    is_limit_ask = jnp.expand_dims(is_limit_ask, axis=1)
    return jnp.where(is_limit_ask, messages, jnp.zeros_like(messages))


@jax.jit
def get_message_by_max_price(messages: jnp.ndarray) -> jnp.ndarray:
    return messages[jnp.argmax(messages[:, Message.PRICE_IDX])]


@jax.jit
def get_message_by_min_price(messages: jnp.ndarray) -> jnp.ndarray:
    return messages[jnp.argmin(messages[:, Message.PRICE_IDX])]


# ===================================== #
# ******* Config your own func ******** #
# ===================================== #


@jax.jit
def get_best_ask(asks: jnp.ndarray) -> int:
    """Return the best / lowest ask price. If there is no ask, return -1."""
    min = jnp.min(
        jnp.where(asks[:, Order.PRICE_IDX] == -1, MAX_INT, asks[:, Order.PRICE_IDX])
    )
    return jnp.where(min == MAX_INT, -1, min)


@jax.jit
def get_best_bid(bids: jnp.ndarray) -> int:
    """Return the best / highest bid price. If there is no bid, return -1."""
    return jnp.max(bids[:, Order.PRICE_IDX])


@jax.jit
def get_volume_at_price(orders: jnp.ndarray, price: int) -> jnp.ndarray:
    volume = jnp.sum(
        jnp.where(orders[:, Order.PRICE_IDX] == price, orders[:, Order.QUANTITY_IDX], 0)
    )
    return volume


@jax.jit
def get_volume_between_prices_soft(
    orders: jnp.ndarray, lower_price: int, upper_price: int
) -> jnp.ndarray:
    """Returns the volume of orders between two prices."""
    volume = jnp.sum(
        jnp.where(
            (orders[:, Order.PRICE_IDX] >= lower_price)
            & (orders[:, Order.PRICE_IDX] <= upper_price),
            orders[:, Order.QUANTITY_IDX],
            0,
        )
    )
    return volume


@jax.jit
def get_volume_between_prices_hard(
    orders: jnp.ndarray, lower_price: int, upper_price: int
) -> jnp.ndarray:
    """Returns the volume of orders between two prices."""
    volume = jnp.sum(
        jnp.where(
            (orders[:, Order.PRICE_IDX] > lower_price)
            & (orders[:, Order.PRICE_IDX] < upper_price),
            orders[:, Order.QUANTITY_IDX],
            0,
        )
    )
    return volume


@jax.jit
def get_init_volume_at_price(orders: jnp.ndarray, price: int) -> jnp.ndarray:
    """Returns the size of initial volume (order with INIT_ORDER_ID) at given price."""
    volume = jnp.sum(
        jnp.where(
            (orders[:, Order.PRICE_IDX] == price)
            & (orders[:, Order.ORDER_ID_IDX] <= INIT_ORDER_ID),
            orders[:, Order.QUANTITY_IDX],
            0,
        )
    )
    return volume


@jax.jit
def get_order_by_id(
    orders: jnp.ndarray,
    order_id: int,
) -> jnp.ndarray:
    """Returns all order fields for the first order matching the given order_id.
    CAVE: if the same ID is used multiple times, will only return the first
    (e.g. for INIT_ORDER_ID)
    """
    idx = jnp.where(
        orders[..., Order.ORDER_ID_IDX] == order_id,
        size=1,
        fill_value=-1,
    )
    # return vector of -1 if not found
    return jax.lax.cond(
        idx[0][0] == -1,
        lambda _: -1 * jnp.ones((6,), dtype=jnp.int32),
        lambda i: orders[i][0],
        idx,
    )


@jax.jit
def get_order_by_id_and_price(
    orders: jax.Array,
    order_id: int,
    price: int,
) -> jax.Array:
    """Returns all order fields for the first order matching the given order_id at the given price.
    CAVE: if the same ID is used multiple times at the given price level, will only return the first
    """
    idx = jnp.where(
        (
            (orders[..., Order.ORDER_ID_IDX] == order_id)
            & (orders[..., Order.PRICE_IDX] == price)
        ),
        size=1,
        fill_value=-1,
    )
    # return vector of -1 if not found
    return jax.lax.cond(
        idx == -1,
        lambda i: -1 * jnp.ones((6,), dtype=jnp.int32),
        lambda i: orders[i][0],
        idx,
    )


@jax.jit
def get_order_ids(
    book: jax.Array,
) -> jax.Array:
    """Returns a list of all order ids in the orderbook"""
    return jnp.unique(book[:, Order.ORDER_ID_IDX], size=book.shape[0], fill_value=1)


@partial(jax.jit, static_argnums=(2,))
def get_agent_orders(
    orders: jnp.ndarray,
    agent_id: int,
    size: int,
) -> jax.Array:
    """Returns all orders for the given agent_id."""
    idx = jnp.where(
        orders[..., Order.AGENT_ID_IDX] == agent_id,
        size=size,
        fill_value=-1,
    )

    fill_value = -1 * jnp.ones((size, 6), dtype=jnp.int32)
    condition = jnp.expand_dims(idx[0] != -1, axis=1)
    return jnp.where(condition, orders[:size], fill_value)


@partial(jax.jit, static_argnums=0)
def get_next_executable_order(side: int, orders: jnp.ndarray):
    # best sell order / ask
    if side == OrderSide.ASK:
        idx = _get_top_ask_order_idx(orders)
    # best buy order / bid
    elif side == OrderSide.BID:
        idx = _get_top_bid_order_idx(orders)
    else:
        raise ValueError("Side must be 1 (bid) or -1 (ask).")
    return orders[idx].squeeze()


@partial(jax.jit, static_argnums=2)
def get_L2_state(asks: jnp.ndarray, bids: jnp.ndarray, n_levels: int) -> jnp.ndarray:
    # unique sorts ascending --> negative values to get descending
    bid_prices = -1 * jnp.unique(
        -1 * bids[:, Order.PRICE_IDX], size=n_levels, fill_value=1
    )
    # replace -1 with max 32 bit int in sorting asks before sorting
    ask_prices = jnp.unique(
        jnp.where(asks[:, Order.PRICE_IDX] == -1, MAX_INT, asks[:, Order.PRICE_IDX]),
        size=n_levels,
        fill_value=-1,
    )
    # replace max 32 bit int with -1 after sorting
    ask_prices = jnp.where(ask_prices == MAX_INT, -1, ask_prices)

    bids = jnp.stack((bid_prices, get_quant_at_prices(bids, bid_prices)))
    asks = jnp.stack((ask_prices, get_quant_at_prices(asks, ask_prices)))
    # set negative volumes to 0
    bids = bids.at[Order.QUANTITY_IDX].set(
        jnp.where(bids[Order.QUANTITY_IDX] < 0, 0, bids[Order.QUANTITY_IDX])
    )
    asks = asks.at[Order.QUANTITY_IDX].set(
        jnp.where(asks[Order.QUANTITY_IDX] < 0, 0, asks[Order.QUANTITY_IDX])
    )
    # combine asks and bids in joint representation
    l2_state = jnp.hstack((asks.T, bids.T)).flatten()
    return l2_state


@partial(jax.jit, static_argnums=1)
def get_bid_state(bids: jnp.ndarray, n_levels: int) -> jnp.ndarray:
    bid_prices = -1 * jnp.unique(
        -1 * bids[:, Order.PRICE_IDX], size=n_levels, fill_value=1
    )
    bid_volumes = get_quant_at_prices(bids, bid_prices)
    return jnp.stack((bid_prices, bid_volumes)).flatten()


@partial(jax.jit, static_argnums=1)
def get_ask_state(asks: jnp.ndarray, n_levels: int) -> jnp.ndarray:
    ask_prices = jnp.unique(asks[:, Order.PRICE_IDX], size=n_levels + 1, fill_value=-1)
    ask_prices = ask_prices[jnp.where(ask_prices != -1, size=10)[0]]
    ask_volumes = get_quant_at_prices(asks, ask_prices)
    return jnp.stack((ask_prices, ask_volumes)).flatten()


vmap_get_L2_state = jax.vmap(get_L2_state, (0, 0, None), 0)


@partial(jax.jit, static_argnums=1)
def get_agent_trades(trades: jnp.ndarray, agent_id: int) -> jnp.ndarray:
    executed = jnp.where(
        (trades[:, Trade.ORDER_PRICE_IDX] >= 0)[:, jnp.newaxis], trades, 0
    )
    mask = (executed[:, Trade.ORDER_AGENT_ID_IDX] == agent_id) | (
        executed[:, Trade.MESSAGE_AGENT_ID_IDX] == agent_id
    )
    return jnp.where(mask[:, jnp.newaxis], executed, 0)


@partial(jax.jit, static_argnums=1)
def delta_portfolio(agent_trades: jnp.ndarray, agent_id: int) -> jnp.ndarray:
    """Calculate the delta of the portfolio from the given trades."""

    def _delta(agent_trade: jnp.ndarray) -> jnp.ndarray:
        is_buyer = (
            (agent_trade[Trade.ORDER_SIDE_IDX] == OrderSide.ASK)
            & (agent_trade[Trade.MESSAGE_AGENT_ID_IDX] == agent_id)
        ) | (
            (agent_trade[Trade.ORDER_SIDE_IDX] == OrderSide.BID)
            & (agent_trade[Trade.ORDER_AGENT_ID_IDX] == agent_id)
        )
        return jax.lax.cond(
            is_buyer,
            lambda trade: jnp.array(
                [
                    trade[Trade.ORDER_PRICE_IDX] * trade[Trade.ORDER_QUANTITY_IDX] * -1,
                    trade[Trade.ORDER_QUANTITY_IDX],
                ],
                dtype=jnp.int32,
            ),
            lambda trade: jnp.array(
                [
                    trade[Trade.ORDER_PRICE_IDX] * trade[Trade.ORDER_QUANTITY_IDX],
                    -1 * trade[Trade.ORDER_QUANTITY_IDX],
                ],
                dtype=jnp.int32,
            ),
            operand=agent_trade,
        )

    vmap_delta = jax.vmap(_delta)
    return jnp.sum(vmap_delta(agent_trades), axis=0)
