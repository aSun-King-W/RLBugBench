"""
Inventory Management Environment (Buggy Version)

A single-product periodic review inventory system environment containing
3 deliberately introduced defects for RL agent repair tasks.

Defects:
1. Reward sign reversed: returns positive cost instead of negative penalty
2. Inventory calculation bug: ending inventory doesn't subtract demand
3. No termination condition: episode never ends (no max_steps check)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class InventoryManagementEnv(gym.Env):
    """Single-product periodic review inventory system with bugs."""

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

        # --- BUG 2: ending_inv doesn't subtract demand ---
        ending_inv = self.starting_inventory + float(action)

        # Calculate costs
        if ending_inv >= 0:
            holding_cost = self.holding_cost_per_unit * ending_inv
            stockout_penalty = 0.0
        else:
            holding_cost = 0.0
            stockout_penalty = self.stockout_penalty_per_unit * (-ending_inv)

        # --- BUG 1: reward sign is reversed (positive cost) ---
        reward = holding_cost + stockout_penalty

        # --- BUG 3: no termination condition ---
        terminated = False
        truncated = False

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
