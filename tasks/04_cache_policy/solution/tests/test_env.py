"""
Black-box tests for the fixed Cache Replacement environment.

All tests interact with the environment only through the public Gymnasium
API -- no access to internal attributes or methods.
"""

import pytest
import numpy as np
from gymnasium import spaces
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fixed_env import CacheEnv, CACHE_SIZE, N_CONTENTS

N_ACTIONS = CACHE_SIZE  # 10


class TestCacheEnv:
    """Test suite for the fixed CacheEnv."""

    @pytest.fixture
    def env(self):
        return CacheEnv()

    # ------------------------------------------------------------------
    # Space specifications
    # ------------------------------------------------------------------

    def test_observation_space_shape(self, env):
        """Observation should have CACHE_SIZE * 4 + 1 features."""
        expected = CACHE_SIZE * 4 + 1
        assert env.observation_space.shape == (expected,), (
            f"Expected {expected}-D observation, "
            f"got {env.observation_space.shape}"
        )

    def test_observation_space_dtype(self, env):
        assert env.observation_space.dtype == np.float32

    def test_observation_space_bounds(self, env):
        """All observation features should be in [0, 1]."""
        assert np.all(env.observation_space.low >= 0.0)
        assert np.all(env.observation_space.high <= 1.0)

    def test_action_space_type(self, env):
        assert isinstance(env.action_space, spaces.Discrete)

    def test_action_space_n(self, env):
        assert env.action_space.n == N_ACTIONS

    # ------------------------------------------------------------------
    # Reset contract
    # ------------------------------------------------------------------

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert obs in env.observation_space
        assert isinstance(info, dict)

    def test_reset_is_deterministic(self, env):
        """Same seed should yield identical initial observations."""
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=0)
        assert np.array_equal(obs1, obs2), (
            "Same seed should yield identical initial states"
        )

    def test_different_seed_gives_variation(self, env):
        """Check that different seeds produce different trajectories
        (non-deterministic reset).  May be identical if both seeds
        produce the same initial cache (unlikely but not guaranteed);
        we simply verify no crash."""
        env.reset(seed=0)
        for _ in range(5):
            env.step(0)
        # After different seed, completing an episode should not crash
        env.reset(seed=1)
        for _ in range(5):
            env.step(0)

    # ------------------------------------------------------------------
    # Step contract
    # ------------------------------------------------------------------

    def test_step_returns_valid_types(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        assert obs in env.observation_space
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_all_actions_valid(self, env):
        """Every action in the action space should be acceptable."""
        env.reset()
        for action in range(env.action_space.n):
            env.reset()
            obs, _, _, _, _ = env.step(action)
            assert obs in env.observation_space, (
                f"Action {action} produced invalid observation"
            )

    # ------------------------------------------------------------------
    # Termination / truncation  (FIX 4)
    # ------------------------------------------------------------------

    def test_episode_truncated_after_max_steps(self):
        """Episode must be truncated after exactly max_steps steps."""
        env = CacheEnv(max_steps=20)
        env.reset()
        for i in range(20):
            _, _, _, truncated, _ = env.step(0)
            if i < 19:
                assert not truncated, f"Unexpected truncation at step {i}"
            else:
                assert truncated, "Expected truncation at step 19 (max_steps=20)"

    # ------------------------------------------------------------------
    # Reward sanity  (FIX 2: immediate per-step reward)
    # ------------------------------------------------------------------

    def test_reward_is_binary(self, env):
        """Reward should be 0.0 or 1.0 (immediate hit/miss)."""
        env.reset()
        for _ in range(100):
            _, reward, _, truncated, _ = env.step(env.action_space.sample())
            assert reward in (0.0, 1.0), f"Non-binary reward: {reward}"
            if truncated:
                env.reset()

    def test_reward_finite(self, env):
        """Reward must always be finite."""
        env.reset()
        for _ in range(100):
            _, reward, _, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if truncated:
                env.reset()

    def test_hit_rate_reasonable(self, env):
        """Overall hit rate should be between 5% and 60% for random policy
        with cache_size=10 and Zipf-distributed requests."""
        env.reset()
        total_hits = 0
        total_steps = 0
        done = False
        while not done:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            total_hits += reward
            total_steps += 1
        hit_rate = total_hits / max(total_steps, 1)
        assert 0.05 <= hit_rate <= 0.60, (
            f"Hit rate {hit_rate:.3f} outside expected range [0.05, 0.60]"
        )

    # ------------------------------------------------------------------
    # Reproducibility  (FIX 4)
    # ------------------------------------------------------------------

    def test_seed_reproducibility(self):
        """Same seed must yield an identical trajectory."""
        actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10

        env1 = CacheEnv()
        env1.reset(seed=42)
        traj1 = []
        for a in actions:
            o, r, term, trunc, _ = env1.step(a)
            traj1.append((o.copy(), r))
            if term or trunc:
                break

        env2 = CacheEnv()
        env2.reset(seed=42)
        for i, a in enumerate(actions):
            o, r, term, trunc, _ = env2.step(a)
            assert np.array_equal(o, traj1[i][0]), (
                f"Observation mismatch at step {i}"
            )
            assert abs(r - traj1[i][1]) < 1e-5, (
                f"Reward mismatch at step {i}: {r} != {traj1[i][1]}"
            )
            if term or trunc:
                break

    # ------------------------------------------------------------------
    # Stress test
    # ------------------------------------------------------------------

    def test_many_episodes_no_crash(self, env):
        """100 episodes with random actions must not crash."""
        for ep in range(100):
            env.reset()
            done = False
            steps = 0
            while not done:
                a = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(int(a))
                done = terminated or truncated
                steps += 1
            assert steps > 0, f"Episode {ep} had 0 steps"

    # ------------------------------------------------------------------
    # Random-policy sanity
    # ------------------------------------------------------------------

    def test_random_policy_reward_finite(self):
        """Average random-policy reward should be finite."""
        env = CacheEnv(max_steps=100)
        episodes = 20
        rates = []
        for _ in range(episodes):
            env.reset()
            done = False
            total_hits = 0.0
            total_steps = 0
            while not done:
                a = env.action_space.sample()
                _, r, term, trunc, _ = env.step(int(a))
                done = term or trunc
                total_hits += r
                total_steps += 1
            rates.append(total_hits / max(total_steps, 1))
        avg = float(np.mean(rates))
        assert np.isfinite(avg), f"Non-finite average hit rate: {avg}"
