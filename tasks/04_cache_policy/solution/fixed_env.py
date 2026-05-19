"""
Fixed Cache Replacement Policy Environment.

All bugs from the original env.py are corrected:
1. Observation now includes access_count, time_since_last_access, and
   frequency for each cache slot — enough to learn LRU/LFU policies.
2. Reward is given immediately per step: +1 for a cache hit, 0 for a miss.
3. History is cleared in reset(), preventing unbounded growth across episodes.
4. super().reset(seed=seed) properly seeds the RNG; termination uses
   self.current_step instead of the stale self.steps.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

CACHE_SIZE = 10
N_CONTENTS = 100
MAX_STEPS = 200

# Zipf-like popularity weights
_POP_PROBS = 1.0 / (np.arange(N_CONTENTS, dtype=np.float64) + 1.0)
_POP_PROBS /= _POP_PROBS.sum()


class CacheEnv(gym.Env):
    """CDN cache replacement environment (all bugs fixed)."""

    def __init__(self, max_steps=MAX_STEPS):
        super().__init__()

        self.max_steps = max_steps

        # FIX 1: observation includes temporal features
        # Per slot: content_id, access_count, time_since_last_access, frequency
        obs_dim = CACHE_SIZE * 4 + 1  # +1 for current request
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )

        self.action_space = spaces.Discrete(CACHE_SIZE)

        # FIX 3: use bounded deque for history
        self.history = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        # FIX 4: properly seed the RNG
        super().reset(seed=seed)

        self.current_step = 0

        # Pre-fill cache with random distinct content
        self.cache = np.array(
            self.np_random.choice(N_CONTENTS, size=CACHE_SIZE, replace=False),
            dtype=np.int32,
        )

        # Per-content metadata
        self.access_counts = np.zeros(N_CONTENTS, dtype=np.int32)
        self.last_access_step = np.full(N_CONTENTS, -1, dtype=np.int32)
        self.frequency = np.zeros(N_CONTENTS, dtype=np.int32)

        # Request sequence with temporal locality
        self.request_sequence = self._gen_requests()

        # FIX 3: reset history on new episode
        self.history = []

        # Set the first request (metadata updated in step(), not here)
        self._current_request = int(self.request_sequence[0])

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        request = self._current_request
        hit = request in self.cache

        if not hit:
            # Evict chosen slot and replace with new content
            self.cache[action] = request

        # FIX 2: immediate per-step reward
        reward = 1.0 if hit else 0.0

        # Update metadata
        self.access_counts[request] += 1
        self.last_access_step[request] = self.current_step
        self.frequency[request] += 1

        # Advance to next request
        next_step = self.current_step + 1
        if next_step < self.max_steps:
            self._current_request = int(self.request_sequence[next_step])

        self.current_step = next_step

        # FIX 4: correct termination variable
        terminated = False
        truncated = self.current_step >= self.max_steps

        # FIX 3: bounded history
        self.history.append({
            'step': self.current_step,
            'request': request,
            'action': action,
            'hit': hit,
        })
        if len(self.history) > 5000:
            self.history = self.history[-5000:]

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Observation  (FIX 1: rich temporal features per slot)
    # ------------------------------------------------------------------

    def _get_obs(self):
        """Build observation vector with full temporal features.

        For each cache slot: (content_id_norm, access_count_norm,
        time_since_last_access_norm, frequency_norm).
        Plus the current request id.
        """
        features = np.zeros(CACHE_SIZE * 4 + 1, dtype=np.float32)
        for i, cid in enumerate(self.cache):
            base = i * 4
            features[base] = cid / N_CONTENTS
            features[base + 1] = min(self.access_counts[cid] / 100.0, 1.0)
            if self.last_access_step[cid] >= 0:
                tsl = (self.current_step - self.last_access_step[cid]) / self.max_steps
                features[base + 2] = min(tsl, 1.0)
            else:
                features[base + 2] = 0.0
            features[base + 3] = min(self.frequency[cid] / 50.0, 1.0)

        features[-1] = self._current_request / N_CONTENTS
        return features

    # ------------------------------------------------------------------
    # Request generation
    # ------------------------------------------------------------------

    def _gen_requests(self):
        """Generate a request sequence with temporal locality.

        Uses a Zipf-like popularity distribution; 30 % of requests
        repeat a recently-seen item (temporal locality).
        """
        probs = _POP_PROBS
        requests = np.zeros(self.max_steps, dtype=np.int32)
        for i in range(self.max_steps):
            if i > 0 and self.np_random.random() < 0.3:
                start = max(0, i - 20)
                requests[i] = int(self.np_random.choice(requests[start:i]))
            else:
                r = self.np_random.random()
                cumulative = 0.0
                for cid in range(N_CONTENTS):
                    cumulative += probs[cid]
                    if r <= cumulative:
                        requests[i] = cid
                        break
        return requests
