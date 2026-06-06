# =============================================================================
# YOUR FILE — this is the only file you submit.
# Implement TradingEnv and Agent below. Do not modify anything in src/.
# =============================================================================

import numpy as np
from gymnasium import spaces

from src.env import BaseTradingEnv
from src.base import BaseAgent
from src.data import load_prices, split, build_features


# ── Environment ───────────────────────────────────────────────────────────────

class TradingEnv(BaseTradingEnv):

    def __init__(self, prices, transaction_cost_bps=10.0, initial_cash=10_000.0):
        super().__init__(prices, transaction_cost_bps, initial_cash)

        # TODO: how many past timesteps should the agent see?
        self._lookback = ...

        # TODO: how many allocation options? define what each action means before coding.
        self.action_space = spaces.Discrete(...)

        # TODO: shape must match exactly what _obs() returns — figure out _obs() first.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(...,), dtype=np.float32
        )

    def _obs(self) -> np.ndarray:
        # What does the agent see at each step?
        #
        # self.prices     — close prices, shape (T, 4)
        # self.data       — full OHLCV DataFrame
        # self._t         — current timestep
        # self._weights   — current portfolio weights, shape (4,)
        # self._lookback  — rows of history you declared visible
        #
        # Return: 1D float32 array matching observation_space.shape.
        raise NotImplementedError

    def _weights_from_action(self, action: int) -> np.ndarray:
        # Map action index to portfolio weights.
        # Shape (4,): [asset_0, asset_1, asset_2, cash]. Must sum to 1.
        # Risky asset weights may be negative (short). Cash must be >= 0.
        #
        # Long-only examples:
        #   [0.0,  0.0,  0.0,  1.0]   # all cash
        #   [1.0,  0.0,  0.0,  0.0]   # all asset_0
        #
        # Short examples (risky weights in [-1.0, 1.0], cash >= 0, sum = 1):
        #   [-0.5, 0.0,  0.0,  1.5]   # short 50% asset_0, hold 150% cash
        #   [1.0,  0.0, -0.5,  0.5]   # long asset_0, short asset_2, rest in cash
        #
        # Think carefully: shorting changes the risk profile significantly.
        # Does your reward function handle losses from a wrong-way short?
        raise NotImplementedError

    def _reward(self, prev_value: float, curr_value: float) -> float:
        # Scalar reward after each step.
        # Start simple (e.g. log return). Document what broke and why you changed it.
        raise NotImplementedError


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent(BaseAgent):

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__(obs_dim, n_actions)
        # TODO: Q-network, target network, optimizer, replay buffer.
        raise NotImplementedError

    def train(self, env, n_steps: int = 200_000) -> None:
        # TODO: training loop.
        #
        # obs, _ = env.reset()
        # for step in range(n_steps):
        #     action = <epsilon-greedy>
        #     next_obs, reward, terminated, truncated, _ = env.step(action)
        #     <store transition, sample batch, update Q-network>
        #     <periodically sync target network>
        #     obs = next_obs if not (terminated or truncated) else env.reset()[0]
        raise NotImplementedError

    def act(self, obs: np.ndarray) -> int:
        # Greedy action at evaluation time — no exploration.
        raise NotImplementedError
