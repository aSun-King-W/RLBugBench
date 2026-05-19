"""
Verify that every buggy environment fails key black-box checks.

This script imports each task's buggy environment (environment/env.py)
and runs targeted tests that the buggy version should fail.  Its exit
code is 0 only when ALL expected bugs are detected, confirming that the
buggy environments are indeed distinguishable from the fixed versions.

Usage:
    python verify/test_buggy_envs.py
"""

import sys
import os
import importlib.util
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def load_env_module(task_rel_path):
    """Import a Python file as a module from its absolute path."""
    abs_path = os.path.join(ROOT, task_rel_path)
    spec = importlib.util.spec_from_file_location(f"buggy_env_{task_rel_path}", abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BUG_COUNT = 0


def expect_bug(description: str, bug_condition: bool):
    """Check that the buggy behaviour is present.

    If bug_condition is True, the bug is confirmed present -> BUG-DETECTED.
    If bug_condition is False, the bug is missing -> UNEXPECTED (may mean
    the bug was accidentally fixed or the test is wrong).
    """
    global BUG_COUNT
    if bug_condition:
        BUG_COUNT += 1
        print(f"  [BUG-DETECTED] {description}")
    else:
        print(f"  [UNEXPECTED] {description} -- bug NOT detected")


# ===========================================================================
# Task 1 -- Inventory Management
# ===========================================================================

def test_inventory_buggy():
    print("\n=== Task 1: Inventory Management (Buggy) ===")

    mod = load_env_module(
        os.path.join("tasks", "01_inventory_management", "environment", "env.py")
    )
    InventoryManagementEnv = mod.InventoryManagementEnv

    # BUG 1: reward sign reversed -> reward is positive (cost not negated)
    env = InventoryManagementEnv()
    env.reset()
    r_total = 0.0
    for _ in range(50):
        _, r, _, _, _ = env.step(5)
        r_total += r
    expect_bug("BUG 1: Reward should be positive (sign not negated)", r_total > 0)

    # BUG 2: ending_inv doesn't subtract demand -> inventory stays high
    env.reset()
    env.step(20)  # max order
    obs_after, _, _, _, _ = env.step(0)  # no new order
    # If demand was subtracted, inventory would drop noticeably
    # With bug: ending_inv = 20 + 0 = 20, so starting_inv ~= 20
    expect_bug("BUG 2: Inventory stays high after ordering 0 (demand not subtracted)",
               obs_after[0] > 15.0)

    # BUG 3: no truncation -> episode never ends (or continues past max_steps)
    env = InventoryManagementEnv(max_steps=100)
    env.reset()
    steps = 0
    truncated = False
    for _ in range(200):
        steps += 1
        _, _, _, t, _ = env.step(0)
        truncated = t
        if truncated:
            break
    expect_bug("BUG 3: Episode still runs after max_steps=100 (no truncation)",
               not truncated or steps > 100)


# ===========================================================================
# Task 2 -- Job Shop Scheduling
# ===========================================================================

def test_job_scheduling_buggy():
    print("\n=== Task 2: Job Shop Scheduling (Buggy) ===")

    mod = load_env_module(
        os.path.join("tasks", "02_job_scheduling", "environment", "env.py")
    )
    JobSchedulingEnv = mod.JobSchedulingEnv

    # BUG 1: no action masking -> info dict has no action_mask
    env = JobSchedulingEnv()
    obs, info = env.reset()
    expect_bug("BUG 1: info from reset() should NOT contain action_mask",
               "action_mask" not in info)

    _, _, _, _, info = env.step(0)
    expect_bug("BUG 1: info from step() should NOT contain action_mask",
               "action_mask" not in info)

    # BUG 2: step_count starts at 1 -> truncated fires at step 20 (not 19)
    env = JobSchedulingEnv(max_steps=20)
    env.reset()
    steps_run = 0
    for _ in range(25):
        _, _, _, truncated, _ = env.step(0)
        steps_run += 1
        if truncated:
            break
    expect_bug("BUG 2: Episode truncates at step 19 instead of 20 (off-by-one)",
               steps_run == 19)

    # BUG 3: reward normalisation explosion -> -inf when all jobs complete
    env = JobSchedulingEnv(max_steps=100)
    env.reset()
    found_nonfinite = False
    for _ in range(500):
        _, r, term, _, _ = env.step(env.action_space.sample())
        if not np.isfinite(r):
            found_nonfinite = True
            break
        if term:
            break
    expect_bug("BUG 3: Non-finite reward (-inf) occurs (division by zero)",
               found_nonfinite)


# ===========================================================================
# Task 3 -- Ad Bidding
# ===========================================================================

def test_ad_bidding_buggy():
    print("\n=== Task 3: Ad Bidding (Buggy) ===")

    mod = load_env_module(
        os.path.join("tasks", "03_ad_bidding", "environment", "env.py")
    )
    AdBiddingEnv = mod.AdBiddingEnv

    # BUG 1 + 2: observation dimension is 5, should be 6 (missing remaining_budget)
    env = AdBiddingEnv()
    expect_bug("BUG 1+2: Observation has 5 dims, missing remaining_budget",
               env.observation_space.shape[0] == 5)

    # BUG 3: non-stationary reward not normalized -> raw conversion values
    env.reset()
    rewards = []
    for _ in range(100):
        _, r, _, truncated, _ = env.step(env.action_space.sample())
        rewards.append(r)
        if truncated:
            break
    max_r = max(rewards) if rewards else 0
    expect_bug("BUG 3: Reward is large (raw conversions, not normalized)",
               max_r >= 100)

    # BUG 4: seed not fixed -> same seed produces different trajectories
    env1 = AdBiddingEnv()
    env1.reset(seed=42)
    r1 = [env1.step(0)[1] for _ in range(5)]

    env2 = AdBiddingEnv()
    env2.reset(seed=42)
    r2 = [env2.step(0)[1] for _ in range(5)]

    expect_bug("BUG 4: Same seed should NOT yield identical rewards (no seeding)",
               r1 != r2)


# ===========================================================================
# Task 4 -- Cache Replacement Policy
# ===========================================================================

def test_cache_buggy():
    print("\n=== Task 4: Cache Policy (Buggy) ===")

    mod = load_env_module(
        os.path.join("tasks", "04_cache_policy", "environment", "env.py")
    )
    CacheEnv = mod.CacheEnv
    CACHE_SIZE = mod.CACHE_SIZE

    env = CacheEnv()

    # BUG 1: observation missing temporal features (CACHE_SIZE+1 vs CACHE_SIZE*4+1)
    expect_bug(f"BUG 1: Observation has {CACHE_SIZE+1} dims (missing temporal features)",
               env.observation_space.shape[0] == CACHE_SIZE + 1)

    # BUG 2: delayed reward accumulation -> reward is not 0 or 1
    env.reset()
    rewards = set()
    for _ in range(200):
        _, r, _, truncated, _ = env.step(env.action_space.sample())
        rewards.add(r)
        if truncated:
            break
    expect_bug("BUG 2: Reward includes fractional values (delayed window accumulation)",
               not rewards.issubset({0.0, 1.0}))

    # BUG 3: history not cleared in reset -> keeps growing across episodes
    env.reset()
    for _ in range(50):
        env.step(0)
    len_after_ep1 = len(env.history)
    env.reset()
    for _ in range(50):
        env.step(0)
    len_after_ep2 = len(env.history)
    expect_bug("BUG 3: History accumulates across episodes (not cleared in reset)",
               len_after_ep2 > 50)

    # BUG 4: wrong variable in termination -> episode never truncates
    env = CacheEnv(max_steps=30)
    env.reset()
    truncated = False
    for _ in range(50):
        _, _, _, t, _ = env.step(0)
        truncated = t
        if truncated:
            break
    expect_bug("BUG 4: Episode never truncates (uses self.steps not self.current_step)",
               not truncated)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    EXPECTED_BUGS = 14  # total number of expect_bug calls above

    print("=" * 60)
    print("  Buggy Environment Verification")
    print("  Expecting all checks to show [BUG-DETECTED]")
    print("=" * 60)

    test_inventory_buggy()
    test_job_scheduling_buggy()
    test_ad_bidding_buggy()
    test_cache_buggy()

    print(f"\n{'=' * 60}")
    print(f"  Detected: {BUG_COUNT} / {EXPECTED_BUGS} bugs confirmed")
    print(f"{'=' * 60}")

    if BUG_COUNT == EXPECTED_BUGS:
        print("\n[SUCCESS] All buggy environments correctly fail their expected checks.")
        sys.exit(0)
    else:
        print(f"\n[WARNING] {EXPECTED_BUGS - BUG_COUNT} bugs not detected.")
        sys.exit(1)
