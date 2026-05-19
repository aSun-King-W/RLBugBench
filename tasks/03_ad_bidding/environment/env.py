"""
Ad Bidding / Campaign Optimization Environment (Buggy Version)

Multi-channel advertising budget allocation environment containing
4 deliberately introduced defects for RL agent repair tasks.

Defects:
1. Budget leakage: step() has no spend cap — total spend can exceed total budget
2. State missing remaining_budget: observation lacks remaining budget info
3. Non-stationary reward not normalized: reward uses raw conversion counts directly
4. Seed not fixed: random seed not fixed, making results non-reproducible
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


N_CHANNELS = 4
TOTAL_BUDGET = 95000.0

# Channel configurations: (base_conversion_rate, cost_per_click)
CHANNELS = [
    (0.08, 0.50),   # search
    (0.05, 0.30),   # social
    (0.02, 0.10),   # display
    (0.06, 0.40),   # video
]


class AdBiddingEnv(gym.Env):
    """Multi-channel ad budget allocation environment with bugs."""

    def __init__(self, max_steps=100):
        super().__init__()

        self.max_steps = max_steps
        self.daily_spend = 1000.0  # fixed daily budget (can exceed TOTAL_BUDGET over many steps)

        # --- BUG 2: observation missing remaining_budget ---
        # Fixed version would have dim N_CHANNELS + 2
        obs_dim = N_CHANNELS + 1  # cvrs + day
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )

        # Action: which channel to allocate daily budget to
        self.action_space = spaces.Discrete(N_CHANNELS)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        # --- BUG 4: no super().reset(seed=seed) — RNG is not seeded ---
        self.step_count = 0
        self.remaining_budget = TOTAL_BUDGET

        # Initialize non-stationary conversion rates (deterministic start)
        self.cvrs = np.array([c[0] for c in CHANNELS], dtype=np.float32)

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        # --- BUG 1: no budget cap check ---
        # Always spends daily_spend regardless of remaining budget.
        # When remaining_budget < daily_spend, this overspends.
        spend = self.daily_spend

        channel_cvr = self.cvrs[action]
        channel_cpc = CHANNELS[action][1]

        # Calculate results
        clicks = spend / channel_cpc
        conversions = clicks * channel_cvr

        # --- BUG 3: raw conversion count as reward (non-stationary) ---
        # Conversion rates drift over time, so the reward distribution shifts.
        reward = float(conversions)

        # Update remaining budget (can go negative — no cap)
        self.remaining_budget -= spend

        # Drift conversion rates (non-stationary random walk)
        drift = np.random.normal(0, 0.02, size=N_CHANNELS).astype(np.float32)
        self.cvrs = np.clip(self.cvrs + drift, 0.001, 0.50)

        # Step counter
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        # --- BUG 2: remaining_budget not included in observation ---
        obs = np.concatenate([
            self.cvrs,
            np.array([self.step_count / self.max_steps], dtype=np.float32),
        ])
        return obs
