"""
Fixed Inventory Management Environment.

All bugs from the original env.py are corrected:
1. Reward is negated so the agent minimises costs.
2. Ending inventory subtracts actual demand.
3. Episode is truncated after max_steps steps.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class InventoryManagementEnv(gym.Env):
    """Single-product periodic review inventory system (fixed)."""

    def __init__(self, max_steps=100):
        super().__init__()

        self.max_steps = max_steps

        # Cost parameters
        self.holding_cost_per_unit = 1.0
        self.stockout_penalty_per_unit = 5.0
        self.max_order = 20
        self.max_inventory = 50

        # Observation: [current_inventory, demand_forecast, on_order_quantity]
        self.observation_space = spaces.Box(
            low=np.array([-self.max_inventory, 0, 0], dtype=np.float32),
            high=np.array([self.max_inventory, 50, self.max_order], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: number of units to order (0..max_order)
        self.action_space = spaces.Discrete(self.max_order + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.starting_inventory = float(self.np_random.integers(0, 20))
        self.demand_forecast = float(self.np_random.integers(0, 30))
        self.on_order = 0.0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # Actual demand follows a normal distribution around the forecast
        demand = float(self.np_random.normal(self.demand_forecast, 5))
        demand = max(0.0, demand)

        # --- FIX 2: subtract demand from ending inventory ---
        ending_inv = self.starting_inventory + float(action) - demand

        # Calculate costs
        if ending_inv >= 0:
            holding_cost = self.holding_cost_per_unit * ending_inv
            stockout_penalty = 0.0
        else:
            holding_cost = 0.0
            stockout_penalty = self.stockout_penalty_per_unit * (-ending_inv)

        # --- FIX 1: negate the reward so the agent minimises total cost ---
        reward = -(holding_cost + stockout_penalty)

        # --- FIX 3: truncate episode after max_steps ---
        terminated = False
        truncated = self.step_count >= self.max_steps

        # Update state for next step
        self.starting_inventory = float(np.clip(ending_inv, -self.max_inventory, self.max_inventory))
        self.on_order = 0.0
        self.demand_forecast = float(self.np_random.integers(0, 30))

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array(
            [self.starting_inventory, self.demand_forecast, self.on_order],
            dtype=np.float32,
        )
