"""
Fixed Ad Bidding / Campaign Optimization Environment.

All bugs from the original env.py are corrected:
1. Budget capped to remaining_budget to prevent overspend.
2. remaining_budget added to observation vector.
3. Reward normalized with running statistics to handle non-stationarity.
4. super().reset(seed=seed) properly seeds the RNG for reproducibility.
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
    """Multi-channel ad budget allocation environment (all bugs fixed)."""

    def __init__(self, max_steps=100):
        super().__init__()

        self.max_steps = max_steps
        self.daily_spend = 1000.0  # same daily budget as buggy, but capped in step()

        # FIX 2: observation includes remaining_budget
        obs_dim = N_CHANNELS + 2  # cvrs + remaining_budget + day
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )

        # Action: which channel to allocate daily budget to
        self.action_space = spaces.Discrete(N_CHANNELS)

        # FIX 3: running statistics for reward normalization
        self._reward_mean = 0.0
        self._reward_std = 1.0
        self._reward_alpha = 0.01  # EMA decay

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        # FIX 4: properly seed the RNG
        super().reset(seed=seed)

        self.step_count = 0
        self.remaining_budget = TOTAL_BUDGET

        # Initialize non-stationary conversion rates
        self.cvrs = np.array([c[0] for c in CHANNELS], dtype=np.float32)

        # Reset normalization statistics
        self._reward_mean = 0.0
        self._reward_std = 1.0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        # FIX 1: cap spend to remaining budget
        spend = min(self.daily_spend, self.remaining_budget)

        channel_cvr = self.cvrs[action]
        channel_cpc = CHANNELS[action][1]

        # Calculate results
        clicks = spend / channel_cpc
        conversions = clicks * channel_cvr

        raw_reward = float(conversions)

        # FIX 3: normalize reward with running statistics
        reward = self._normalize_reward(raw_reward)

        # Update remaining budget (spend is already capped)
        self.remaining_budget -= spend

        # Drift conversion rates (non-stationary random walk)
        drift = self.np_random.normal(0, 0.02, size=N_CHANNELS).astype(np.float32)
        self.cvrs = np.clip(self.cvrs + drift, 0.001, 0.50)

        # Step counter
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Reward normalization  (FIX 3)
    # ------------------------------------------------------------------

    def _normalize_reward(self, raw_reward):
        """Normalize reward using exponential moving average statistics.

        Keeps the reward distribution stable despite non-stationary
        conversion rates.
        """
        self._reward_mean = ((1 - self._reward_alpha) * self._reward_mean
                             + self._reward_alpha * raw_reward)
        self._reward_std = ((1 - self._reward_alpha) * self._reward_std
                            + self._reward_alpha * abs(raw_reward - self._reward_mean))
        return np.float32((raw_reward - self._reward_mean) / max(self._reward_std, 1e-8))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        # FIX 2: remaining_budget included in observation
        obs = np.concatenate([
            self.cvrs,
            np.array([
                self.remaining_budget / TOTAL_BUDGET,
                self.step_count / self.max_steps,
            ], dtype=np.float32),
        ])
        return obs
