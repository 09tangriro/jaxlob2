"""Gym-like environments for optimal execution.

The optimal execution problem is a classic problem in finance where a trader aims to execute a large order of shares in a market while minimizing the cost of execution.
The cost of execution is typically measured as the difference between the price at which the shares are bought or sold and the price at which the shares would have been bought or sold in the absence of the large order.
The optimal execution problem can be posed as a reinforcement learning problem where the trader must learn a policy that determines the size and timing of trades in order to minimize the cost of execution.
"""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from jaxlob2 import dataloader, job


@struct.dataclass
class EnvState:
    asks: chex.Array
    bids: chex.Array
    trades: chex.Array
    market_messages: chex.Array
    agent_portfolio: chex.Array
    init_time: chex.Array
    time: chex.Array
    id_counter: int
    window_index: int
    step_counter: int
    max_steps_in_episode: int

    init_price: int


@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array


class OptimalExecutionDiscrete(environment.Environment):
    """
    The OptimalExecutionDiscrete environment is a discrete action environment for optimal execution.

    At each step, the agent can choose to sell 1 unit of stock at the best bid, the best ask, the mid price, or do nothing.

    The reward is a linear combination of the drift and advantage of the excecution price:
    drift = how much higher the execution price is compared to the initial best bid at the beginning of the episode
    advantage = how much higher the execution price is compared to the best bid at the current step

    :param data_path: Path to the folder containing the LOBSTER CSV files
    :param init_agent_portfolio: Initial portfolio of the agent [cash, stock]
    :param orders_per_side: Size of the array tracking the orders on each side of the book
    :param trades_size: Size of the array tracking the trades
    :param messages_per_step: Number of data messages to process at each step
    :param daily_start_time: Start time of the trading day in seconds
    :param daily_end_time: End time of the trading day in seconds
    :param episode_length: Length of the episode in seconds
    :param drift_coeff: Coefficient of the drift in the reward function
    """

    def __init__(
        self,
        data_path: str,
        init_agent_portfolio: jnp.ndarray = jnp.zeros(
            shape=(2,), dtype=jnp.int32
        ),  # [cash, stock]
        orders_per_side: int = 100,
        trades_size: int = 100,
        messages_per_step: int = 100,
        daily_start_time: int = 34200,
        daily_end_time: int = 57600,
        episode_length: int = 1800,
        drift_coeff: float = 1,
    ) -> None:
        super().__init__()
        self.drift_coeff = drift_coeff
        self._num_actions = 1
        self.tick_size = 100
        self.custom_id_counter = 0
        self.messages_per_step = messages_per_step
        self.agent_id = 1

        self.data_path = data_path
        self._agent_porfolio = init_agent_portfolio
        self.orders_per_side = orders_per_side
        self.trades_size = trades_size
        self.daily_start_time = daily_start_time
        self.daily_end_time = daily_end_time
        self.episode_length = episode_length

        self.messages, self.books, self.max_steps_in_episode_arr = (
            dataloader.load_cubes(
                data_path,
                daily_start_time,
                daily_end_time,
                episode_length,
                messages_per_step,
            )
        )
        self.num_windows = len(self.books)
        self.book_depth = self.books.shape[1] // 4

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(self.messages, self.books)

    def get_obs(
        self, state: EnvState, n_state: EnvState, params: EnvParams
    ) -> chex.Array:
        """Return observation from raw state trafo."""
        # Agent observation
        agent_inventory = n_state.agent_portfolio[1]
        time_left = self.episode_length - jnp.array(
            (n_state.time - n_state.init_time)[0], dtype=jnp.int32
        )
        init_price = n_state.init_price * self.tick_size

        # LOB observation
        market_messages = n_state.market_messages
        msg_type = market_messages[:, job.Message.TYPE_IDX]
        msg_quantity = (
            market_messages[:, job.Message.QUANTITY_IDX]
            * market_messages[:, job.Message.SIDE_IDX]
        )
        msg_price = market_messages[:, job.Message.PRICE_IDX]
        market_messages = jnp.stack([msg_type, msg_quantity, msg_price], axis=0).T
        l2_state = job.get_L2_state(n_state.asks, n_state.bids, self.book_depth // 2)

        return jnp.concatenate(
            [
                jnp.array([agent_inventory, time_left, init_price], dtype=jnp.float32),
                market_messages.flatten(),
                l2_state,
            ],
            axis=0,
            dtype=jnp.float32,
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return jax.lax.cond(
            state.agent_portfolio[1] <= 0,
            lambda x: True,
            lambda x: (x.time - x.init_time)[0] > self.episode_length,
            state,
        )

    def reward(self, state: EnvState, n_state: EnvState, params: EnvParams) -> float:
        """The reward functoin for the environment."""
        agent_cash = state.agent_portfolio[0]
        n_agent_cash = n_state.agent_portfolio[0]
        cash_gained = n_agent_cash - agent_cash

        agent_inventory = state.agent_portfolio[1]
        n_agent_inventory = n_state.agent_portfolio[1]
        inventory_sold = agent_inventory - n_agent_inventory

        exec_price = cash_gained / (inventory_sold + 1e-6)

        best_bid = job.get_best_bid(state.bids) // self.tick_size

        return inventory_sold * (
            (self.drift_coeff * jnp.log(exec_price / state.init_price))
            + jnp.log(exec_price / best_bid)
        )

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: jnp.ndarray, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Step the environment forward by one step."""
        # Obtain the messages for the step from the message data
        data_messages = job.get_data_messages(
            params.message_data, state.window_index, state.step_counter
        )

        cancel_msgs = job.get_cancel_msgs(
            state.asks, self.agent_id, self.num_actions, -1
        )

        # Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        order_type = 1
        side = -1
        agent_id = self.agent_id
        order_id = self.agent_id + state.id_counter
        time = state.time
        # Stack (Concatenate) the info into an array

        best_ask, best_bid = job.get_best_ask(state.asks), job.get_best_bid(state.bids)
        mid_price = (best_ask + best_bid) // 2
        prices = jnp.asarray(
            [
                best_bid,
                mid_price,
                best_ask,
                job.MAX_INT,  # Do nothing
            ],
            jnp.int32,
        )
        is_final_step = self.episode_length - (time - state.init_time)[0] <= 60
        price = jax.lax.cond(is_final_step, lambda _: 0, lambda _: prices[action], None)
        quantity = jax.lax.cond(
            is_final_step, lambda _: state.agent_portfolio[1], lambda _: 1, None
        )

        action_msgs = jnp.array(
            [order_type, side, price, quantity, agent_id, order_id, time[0], time[1]],
            dtype=jnp.int32,
        ).reshape(1, 8)

        # Add to the top of the data messages
        total_messages = jnp.concatenate(
            [cancel_msgs, action_msgs, data_messages], axis=0
        )
        # jax.debug.print("Step messages to process are: \n {}", total_messages)

        # Save time of final message to add to state
        time = total_messages[-1:][0][-2:]

        # Process messages of step (action+data) through the orderbook
        asks, bids, trades = job.process_messages(
            total_messages,
            state.asks,
            state.bids,
            (jnp.ones((self.trades_size, 7)) * -1).astype(jnp.int32),
        )

        agent_trades = job.get_agent_trades(trades, self.agent_id)
        delta_portfolio = job.delta_portfolio(agent_trades, agent_id)
        agent_portfolio = jnp.array(
            [
                state.agent_portfolio[0] + (delta_portfolio[0] // self.tick_size),
                state.agent_portfolio[1] + delta_portfolio[1],
            ],
            jnp.int32,
        )

        n_state = EnvState(
            asks=asks,
            bids=bids,
            trades=trades,
            market_messages=data_messages,
            agent_portfolio=agent_portfolio,
            init_time=state.init_time,
            time=time,
            id_counter=state.id_counter + self.num_actions,
            window_index=state.window_index,
            step_counter=state.step_counter + 1,
            max_steps_in_episode=state.max_steps_in_episode,
            init_price=state.init_price,
        )
        done = self.is_terminal(n_state, params)
        reward = self.reward(state, n_state, params)
        return self.get_obs(state, n_state, params), n_state, reward, done, {}

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position in OB."""
        idx_data_window = jax.random.randint(
            key, minval=0, maxval=self.num_windows, shape=()
        )

        # Get the init time based on the first message to be processed in the first step.
        time = job.get_initial_time(params.message_data, idx_data_window)
        # Get initial orders (2xNdepth)x6 based on the initial L2 orderbook for this window
        init_msg = job.get_initial_messages(
            params.book_data, idx_data_window, time, self.book_depth
        )
        # Initialise both sides of the book as being empty
        init_asks = job.init_orders(self.orders_per_side)
        init_bids = job.init_orders(self.orders_per_side)
        init_trades = (jnp.ones((self.trades_size, 7)) * -1).astype(jnp.int32)
        # Process the initial messages through the orderbook
        asks, bids, trades = job.process_messages(
            init_msg, init_asks, init_bids, init_trades
        )

        # Craft the first state
        state = EnvState(
            asks=asks,
            bids=bids,
            trades=trades,
            market_messages=jnp.zeros((self.messages_per_step, 8), dtype=jnp.int32),
            agent_portfolio=self._agent_porfolio,
            init_time=time,
            time=time,
            id_counter=0,
            window_index=idx_data_window,
            step_counter=0,
            max_steps_in_episode=self.max_steps_in_episode_arr[idx_data_window],
            init_price=job.get_best_bid(bids) // self.tick_size,
        )

        return self.get_obs(state, state, params), state

    @property
    def name(self) -> str:
        """Environment name."""
        return "OptimalExecution-v0"

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(53,),
            dtype=jnp.float32,
        )


class OptimalExecutionContinuous(OptimalExecutionDiscrete):
    def __init__(
        self,
        data_path: str,
        init_agent_portfolio: jnp.ndarray = jnp.zeros(
            shape=(2,), dtype=jnp.int32
        ),  # [cash, stock]
        orders_per_side: int = 100,
        trades_size: int = 100,
        messages_per_step: int = 100,
        daily_start_time: int = 34200,
        daily_end_time: int = 57600,
        episode_length: int = 1800,
        action_price_contraction: float = 1,
    ) -> None:
        super().__init__(
            data_path,
            init_agent_portfolio,
            orders_per_side,
            trades_size,
            messages_per_step,
            daily_start_time,
            daily_end_time,
            episode_length,
        )
        self.action_price_contraction = action_price_contraction

    def get_obs(
        self, state: EnvState, n_state: EnvState, params: EnvParams
    ) -> chex.Array:
        best_bid = job.get_best_bid(n_state.bids)
        bid_volume = job.get_volume_at_price(n_state.bids, best_bid)
        ask_state = job.get_ask_state(n_state.asks, self.book_depth)
        agent_inventory = n_state.agent_portfolio[1]
        time_left = self.episode_length - jnp.array(
            (n_state.time - n_state.init_time)[0], dtype=jnp.int32
        )

        old_best_bid = job.get_best_bid(state.bids)
        old_best_bid_volume = job.get_volume_at_price(state.bids, old_best_bid)
        old_ask_state = job.get_ask_state(state.asks, self.book_depth)

        partial_obs = jnp.array(
            [
                agent_inventory,
                time_left,
                best_bid,
                bid_volume,
                old_best_bid,
                old_best_bid_volume,
            ],
            jnp.int32,
        )
        return jnp.concatenate(
            [partial_obs, ask_state, old_ask_state], axis=0, dtype=jnp.int32
        )

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: jnp.ndarray, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        # Obtain the messages for the step from the message data
        data_messages = job.get_data_messages(
            params.message_data, state.window_index, state.step_counter
        )

        cancel_msgs = job.get_cancel_msgs(
            state.asks, self.agent_id, self.num_actions, -1
        )

        # Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        order_type = 1
        side = -1
        agent_id = self.agent_id
        order_id = self.agent_id + state.id_counter
        time = state.time
        # Stack (Concatenate) the info into an array

        best_bid = job.get_best_bid(state.bids)
        is_final_step = self.episode_length - (time - state.init_time)[0] <= 60
        action = jnp.clip(
            action, self.action_space(params).low, self.action_space(params).high
        )
        action_price = (
            best_bid * (1 + (action[0] / self.action_price_contraction))
        ).astype(jnp.int32)
        price = jax.lax.cond(is_final_step, lambda _: 0, lambda _: action_price, None)
        action_quantity = (state.agent_portfolio[1] * action[1]).astype(jnp.int32)
        quantity = jax.lax.cond(
            is_final_step,
            lambda _: state.agent_portfolio[1],
            lambda _: action_quantity,
            None,
        )

        action_msgs = jnp.array(
            [order_type, side, quantity, price, agent_id, order_id, time[0], time[1]],
            dtype=jnp.int32,
        ).reshape(1, 8)

        # Add to the top of the data messages
        total_messages = jnp.concatenate(
            [cancel_msgs, action_msgs, data_messages], axis=0
        )
        # jax.debug.print("Step messages to process are: \n {}", total_messages)

        # Save time of final message to add to state
        time = total_messages[-1:][0][-2:]

        # Process messages of step (action+data) through the orderbook
        asks, bids, trades = job.process_messages(
            total_messages,
            state.asks,
            state.bids,
            (jnp.ones((self.trades_size, 7)) * -1).astype(jnp.int32),
        )

        agent_trades = job.get_agent_trades(trades, self.agent_id)
        delta_portfolio = job.delta_portfolio(agent_trades, agent_id)
        agent_portfolio = jnp.array(
            [
                state.agent_portfolio[0] + (delta_portfolio[0] // self.tick_size),
                state.agent_portfolio[1] + delta_portfolio[1],
            ],
            jnp.int32,
        )

        n_state = EnvState(
            asks=asks,
            bids=bids,
            trades=trades,
            agent_portfolio=agent_portfolio,
            init_time=state.init_time,
            time=time,
            id_counter=state.id_counter + self.num_actions,
            window_index=state.window_index,
            step_counter=state.step_counter + 1,
            max_steps_in_episode=state.max_steps_in_episode,
            init_price=state.init_price,
        )
        done = self.is_terminal(n_state, params)
        reward = self.reward(state, n_state, params)
        return self.get_obs(state, n_state, params), n_state, reward, done, {}

    @property
    def name(self) -> str:
        """Environment name."""
        return "OptimalExecution-v1"

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(low=0, high=1, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return spaces.Box(
            low=0,
            high=jnp.inf,
            shape=(46,),
            dtype=jnp.int32,
        )
