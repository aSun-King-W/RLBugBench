"""
Black-box tests for the fixed Inventory Management environment.

All tests interact with the environment only through the public Gymnasium API —
no access to internal attributes or methods.
"""

import pytest
import numpy as np
from gymnasium import spaces
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fixed_env import InventoryManagementEnv


class TestInventoryManagementEnv:
    """Test suite for the fixed InventoryManagementEnv."""

    @pytest.fixture
    def env(self):
        return InventoryManagementEnv()

    # ------------------------------------------------------------------
    # Space specifications
    # ------------------------------------------------------------------

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (3,), "Expected 3-D observation"

    def test_observation_space_dtype(self, env):
        assert env.observation_space.dtype == np.float32

    def test_action_space_type(self, env):
        assert isinstance(env.action_space, spaces.Discrete)

    def test_action_space_n(self, env):
        assert env.action_space.n == 21  # 0..20 inclusive

    # ------------------------------------------------------------------
    # Reset contract
    # ------------------------------------------------------------------

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert obs in env.observation_space
        assert isinstance(info, dict)

    def test_reset_seed_gives_different_trajectories(self, env):
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=1)
        assert not np.array_equal(obs1, obs2), "Different seeds should yield different initial states"

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
            assert obs in env.observation_space, f"Action {action} produced invalid observation"

    # ------------------------------------------------------------------
    # Reward correctness
    # ------------------------------------------------------------------

    def test_reward_always_non_positive(self, env):
        """Fixed environment penalises costs → reward must be ≤ 0."""
        env.reset()
        for _ in range(50):
            _, reward, _, truncated, _ = env.step(env.action_space.sample())
            assert reward <= 0, f"Expected non-positive reward, got {reward}"
            if truncated:
                env.reset()

    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------

    def test_episode_truncated_after_max_steps(self):
        """Episode must be truncated after max_steps steps."""
        env = InventoryManagementEnv(max_steps=10)
        env.reset()
        for i in range(10):
            _, _, _, truncated, _ = env.step(0)
            expected_truncated = (i >= 9)  # step_count starts at 1, 10th step → truncated
            assert truncated == expected_truncated, (
                f"Expected truncated={expected_truncated} at step {i}, got {truncated}"
            )

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def test_seed_reproducibility(self):
        """Same seed must yield an identical trajectory."""
        actions = [3, 7, 12, 5, 0, 15, 10, 8, 2, 6]

        env1 = InventoryManagementEnv()
        env1.reset(seed=42)
        traj1 = [env1.step(a) for a in actions]

        env2 = InventoryManagementEnv()
        env2.reset(seed=42)
        traj2 = [env2.step(a) for a in actions]

        for i, ((o1, r1, _, _, _), (o2, r2, _, _, _)) in enumerate(zip(traj1, traj2)):
            assert np.array_equal(o1, o2), f"Observation mismatch at step {i}"
            assert r1 == r2, f"Reward mismatch at step {i}: {r1} != {r2}"

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
            assert steps > 0, f"Episode {ep} terminated with 0 steps"

    # ------------------------------------------------------------------
    # Random-policy sanity
    # ------------------------------------------------------------------

    def test_random_policy_reward_negative(self):
        """Average random-policy reward should be negative."""
        env = InventoryManagementEnv(max_steps=50)
        episodes = 20
        total_rewards = []
        for _ in range(episodes):
            env.reset()
            done = False
            ep_rew = 0.0
            while not done:
                a = env.action_space.sample()
                _, r, term, trunc, _ = env.step(int(a))
                done = term or trunc
                ep_rew += r
            total_rewards.append(ep_rew)
        avg = float(np.mean(total_rewards))
        assert avg < 0, f"Expected negative avg reward, got {avg:.2f}"
