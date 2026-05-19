"""
Cache Replacement Policy Environment (Buggy Version)

CDN cache replacement learning environment containing 4 deliberately
introduced defects for RL agent repair tasks.

Defects:
1. State missing temporal features: _get_obs() only returns content IDs,
   missing access_count, time_since_last_access, frequency.
2. Delayed reward accumulation: hit/miss reward is aggregated over a window
   and distributed at boundaries, breaking temporal credit assignment.
3. Memory leak: self.history grows unbounded across episodes (never cleared
   in reset()).
4. Termination condition variable error: uses self.steps (never updated)
   instead of self.current_step, so episodes never truncate naturally.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

CACHE_SIZE = 10
N_CONTENTS = 100     # total unique content items in the universe
MAX_STEPS = 200

# Zipf-like popularity weights
_POP_PROBS = 1.0 / (np.arange(N_CONTENTS, dtype=np.float64) + 1.0)
_POP_PROBS /= _POP_PROBS.sum()


class CacheEnv(gym.Env):
    """CDN cache replacement environment with bugs."""

    def __init__(self, max_steps=MAX_STEPS):
        super().__init__()

        self.max_steps = max_steps

        # --- BUG 1: observation missing temporal features ---
        # Fixed version: CACHE_SIZE * 4 + 1 features
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(CACHE_SIZE + 1,), dtype=np.float32,
        )

        self.action_space = spaces.Discrete(CACHE_SIZE)

        # History tracks every (request, action, hit) — never cleared
        # --- BUG 3: memory leak (unbounded history) ---
        self.history = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        # --- BUG 4: super().reset(seed=seed) not called ---
        # --- BUG 4: self.steps is set but never incremented ---
        self.steps = 0
        self.current_step = 0

        # --- BUG 4: no seeding → uses global np.random (non-reproducible) ---
        # Pre-fill cache with random distinct content
        self.cache = np.array(
            np.random.choice(N_CONTENTS, size=CACHE_SIZE, replace=False),
            dtype=np.int32,
        )

        # Per-content metadata
        self.access_counts = np.zeros(N_CONTENTS, dtype=np.int32)
        self.last_access_step = np.full(N_CONTENTS, -1, dtype=np.int32)
        self.frequency = np.zeros(N_CONTENTS, dtype=np.int32)

        # Request sequence with temporal locality
        self.request_sequence = self._gen_requests()

        # --- BUG 3: self.history is NOT reset (accumulates across episodes) ---
        # --- BUG 2: reward accumulation state ---
        self._window_hits = 0
        self._reward_window = 5

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

        # --- BUG 2: delayed reward accumulation ---
        # Instead of giving immediate hit/miss reward, accumulate over a window
        if hit:
            self._window_hits += 1

        if self.current_step > 0 and self.current_step % self._reward_window == 0:
            reward = self._window_hits / self._reward_window
            self._window_hits = 0
        else:
            reward = 0.0

        # Update metadata
        self.access_counts[request] += 1
        self.last_access_step[request] = self.current_step
        self.frequency[request] += 1

        # Advance to next request
        next_step = self.current_step + 1
        if next_step < self.max_steps:
            self._current_request = int(self.request_sequence[next_step])

        self.current_step = next_step

        # --- BUG 4: uses self.steps (always 0) instead of self.current_step ---
        terminated = False
        truncated = self.steps >= self.max_steps  # never true → never truncates

        # --- BUG 3: history grows unbounded ---
        self.history.append({
            'step': self.current_step,
            'request': request,
            'action': action,
            'hit': hit,
        })

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Observation  (BUG 1: only content IDs, no temporal features)
    # ------------------------------------------------------------------

    def _get_obs(self):
        """Build observation vector.

        Buggy version returns only the content_id for each cache slot
        plus the current request id — no access metadata.
        """
        slot_features = self.cache.astype(np.float32) / N_CONTENTS
        current_req = np.array([self._current_request / N_CONTENTS], dtype=np.float32)
        return np.concatenate([slot_features, current_req])

    # ------------------------------------------------------------------
    # Request generation
    # ------------------------------------------------------------------

    def _gen_requests(self):
        """Generate a request sequence with temporal locality.

        Uses a Zipf-like popularity distribution; 30 % of requests
        repeat a recently-seen item (temporal locality).
        """
        # --- BUG 4: uses global np.random (not reproducible) ---
        probs = _POP_PROBS
        requests = np.zeros(self.max_steps, dtype=np.int32)
        for i in range(self.max_steps):
            if i > 0 and np.random.random() < 0.3:
                start = max(0, i - 20)
                requests[i] = int(np.random.choice(requests[start:i]))
            else:
                r = np.random.random()
                cumulative = 0.0
                for cid in range(N_CONTENTS):
                    cumulative += probs[cid]
                    if r <= cumulative:
                        requests[i] = cid
                        break
        return requests
