"""
Black-box tests for the fixed Job Shop Scheduling environment.

All tests interact with the environment only through the public Gymnasium API —
no access to internal attributes or methods.
"""

import pytest
import numpy as np
from gymnasium import spaces
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fixed_env import JobSchedulingEnv, NUM_JOBS, NUM_MACHINES


class TestJobSchedulingEnv:
    """Test suite for the fixed JobSchedulingEnv."""

    @pytest.fixture
    def env(self):
        return JobSchedulingEnv()

    # ------------------------------------------------------------------
    # Space specifications
    # ------------------------------------------------------------------

    def test_observation_space_shape(self, env):
        expected = 2 * NUM_JOBS + NUM_MACHINES + 1  # 14
        assert env.observation_space.shape == (expected,), (
            f"Expected {expected}-D observation"
        )

    def test_observation_space_dtype(self, env):
        assert env.observation_space.dtype == np.float32

    def test_action_space_type(self, env):
        assert isinstance(env.action_space, spaces.Discrete)

    def test_action_space_n(self, env):
        assert env.action_space.n == NUM_JOBS  # 5

    # ------------------------------------------------------------------
    # Reset contract
    # ------------------------------------------------------------------

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert obs in env.observation_space
        assert isinstance(info, dict)

    def test_reset_is_deterministic(self, env):
        """JSS initial state is deterministic — same seed → same obs."""
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=0)
        assert np.array_equal(obs1, obs2), (
            "Same seed should yield identical initial states"
        )

    def test_reset_returns_action_mask(self, env):
        """Fixed env must return action_mask in info from reset()."""
        _, info = env.reset()
        assert "action_mask" in info, "info must contain action_mask"
        mask = info["action_mask"]
        assert len(mask) == NUM_JOBS
        assert all(isinstance(m, bool) for m in mask)

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

    def test_step_returns_action_mask(self, env):
        """Fixed env must return action_mask in info from step()."""
        env.reset()
        for _ in range(5):
            _, _, _, _, info = env.step(0)
            assert "action_mask" in info, "step info must contain action_mask"
            mask = info["action_mask"]
            assert len(mask) == NUM_JOBS

    # ------------------------------------------------------------------
    # Action masking correctness
    # ------------------------------------------------------------------

    def test_action_mask_eventually_all_false(self, env):
        """When all jobs complete, the mask should be all False."""
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            done = terminated or truncated
        mask = info["action_mask"]
        assert all(not m for m in mask), (
            "All jobs completed but mask still has True entries"
        )

    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------

    def test_episode_truncated_after_max_steps(self):
        """Episode must be truncated after exactly max_steps steps."""
        env = JobSchedulingEnv(max_steps=20)
        env.reset()
        for i in range(20):
            _, _, _, truncated, _ = env.step(0)
            if i < 19:
                assert not truncated, f"Unexpected truncation at step {i}"
            else:
                assert truncated, "Expected truncation at step 19 (max_steps=20)"

    def test_episode_terminated_when_all_jobs_done(self):
        """Episode must terminate when all jobs are completed."""
        env = JobSchedulingEnv(max_steps=500)  # generous limit
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
            if steps > 500:
                pytest.fail("Episode did not terminate within 500 steps")
        # If truncated, we rely on max_steps being large enough
        assert steps < 500, (
            "Episode should terminate naturally (all jobs done) before max_steps"
        )

    # ------------------------------------------------------------------
    # Reward sanity
    # ------------------------------------------------------------------

    def test_reward_finite(self, env):
        """Reward must always be finite (no inf / NaN)."""
        env.reset()
        for _ in range(100):
            _, reward, _, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if truncated:
                env.reset()

    def test_reward_always_negative(self, env):
        """Reward should always be negative (step penalty)."""
        env.reset()
        for _ in range(50):
            _, reward, _, truncated, _ = env.step(env.action_space.sample())
            assert reward < 0, f"Expected negative reward, got {reward}"
            if truncated:
                env.reset()

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def test_seed_reproducibility(self):
        """Same seed must yield identical trajectories."""
        actions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

        env1 = JobSchedulingEnv()
        env1.reset(seed=42)
        traj1 = []
        for a in actions:
            o, r, term, trunc, info = env1.step(a)
            traj1.append((o.copy(), r))
            if term or trunc:
                break

        env2 = JobSchedulingEnv()
        env2.reset(seed=42)
        for i, a in enumerate(actions):
            o, r, term, trunc, info = env2.step(a)
            assert np.array_equal(o, traj1[i][0]), (
                f"Observation mismatch at step {i}"
            )
            assert r == traj1[i][1], f"Reward mismatch at step {i}: {r} != {traj1[i][1]}"
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
        env = JobSchedulingEnv(max_steps=50)
        episodes = 20
        rewards = []
        for _ in range(episodes):
            env.reset()
            done = False
            ep_rew = 0.0
            while not done:
                a = env.action_space.sample()
                _, r, term, trunc, _ = env.step(int(a))
                done = term or trunc
                ep_rew += r
            rewards.append(ep_rew)
        avg = float(np.mean(rewards))
        assert np.isfinite(avg), f"Non-finite average reward: {avg}"
