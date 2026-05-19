# RL Benchmark: Bug-Fixing & Automated Evaluation Suite

A benchmark for evaluating large language models on real-world RL engineering tasks.
Four progressively harder tasks, each containing deliberately crafted environment bugs
that agents must find and fix.

## Overview

| Task | Domain | Difficulty | Bugs | Topic |
|------|--------|-----------|------|-------|
| 1 | Inventory Management | ★☆☆☆☆ | 3 | Reward sign, inventory update, termination |
| 2 | Job Shop Scheduling | ★★☆☆☆ | 3 | Action masking, off-by-one, numerical stability |
| 3 | Ad Bidding / Campaign Optimization | ★★★☆☆ | 4 | Budget enforcement, state design, reward normalisation, seeding |
| 4 | Cache Replacement Policy | ★★★★☆ | 4 | Temporal features, credit assignment, memory leak, termination |

## Task Structure

Each task follows the same layout:

```
tasks/<id>_<name>/
  environment/env.py          -- Buggy environment (to fix)
  solution/
    fixed_env.py              -- Corrected environment
    train.py                  -- Minimal PPO training script
    tests/test_env.py         -- Black-box pytest suite
    solve.sh                  -- One-shot verification script
  README.md                   -- Task-specific instructions
```

## Environment Interface

All environments follow the Gymnasium `Env` interface (`gymnasium >= 0.29`):

```python
import gymnasium as gym

env = gym.make(...)  # or instantiate directly
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

- **reset(seed, options)** -- returns initial observation, info dict
- **step(action)** -- returns (obs, reward, terminated, truncated, info)
- **observation_space** / **action_space** -- Gymnasium Space objects

## Buggy Environment Properties

- All buggy environments execute without raising unhandled exceptions
- Training on a buggy environment produces measurably worse results than on the fixed version
- Each bug is detectable through black-box Gymnasium API calls
- No bug requires source-code access to identify (though inspecting the code makes them obvious)

## Task Descriptions

### Task 1: Inventory Management [tasks/01_inventory_management/](tasks/01_inventory_management/)

Single-product periodic review inventory system. The agent observes current inventory,
demand forecast, and on-order quantity, then decides how many units to order.

**Bugs:**
1. **Reward sign reversed** -- `reward = cost` instead of `reward = -cost`. Agent learns to maximise cost.
2. **Inventory calculation missing demand** -- `ending_inv = starting_inv + order` ignores demand.
3. **No truncation** -- Episode never ends; missing `max_steps` check.

### Task 2: Job Shop Scheduling [tasks/02_job_scheduling/](tasks/02_job_scheduling/)

Multi-job, multi-machine scheduling with 5 jobs across 3 machines. Each job has 2-3
operations with machine-duration pairs. Objective: minimise makespan.

**Bugs:**
1. **No action masking** -- Completed jobs remain selectable, wasting decision steps.
2. **Early termination** -- Step count starts at 1 instead of 0, episode ends one step early.
3. **Reward normalisation explosion** -- Division by zero when active jobs = 0, producing `-inf`.

### Task 3: Ad Bidding / Campaign Optimization [tasks/03_ad_bidding/](tasks/03_ad_bidding/)

Multi-channel advertising budget allocation. The agent distributes a fixed daily spend
across 4 channels with different conversion rates that drift over time.

**Bugs:**
1. **Budget leakage** -- No spend cap; total spend can exceed remaining budget.
2. **Missing state** -- Observation lacks `remaining_budget`, making informed decisions impossible.
3. **Non-stationary reward** -- Raw conversion counts without normalisation; reward distribution shifts.
4. **Seed not fixed** -- RNG not seeded, results are non-reproducible.

### Task 4: Cache Replacement Policy [tasks/04_cache_policy/](tasks/04_cache_policy/)

CDN content delivery network cache replacement. The agent manages a fixed-size cache
and decides which content to evict when the cache is full. Requests follow a Zipf-like
distribution with temporal locality.

**Bugs:**
1. **Missing temporal features** -- Observation only contains content IDs, no access frequency or recency.
2. **Delayed reward accumulation** -- Hit/miss rewards are aggregated over windows, corrupting credit assignment.
3. **Memory leak** -- History list grows unbounded across episodes, degrading performance.
4. **Wrong termination variable** -- Uses un-updated `self.steps` instead of `self.current_step`.

## Verification

```bash
# Run all task verifications sequentially
bash verify/verify_all.sh

# Run buggy-environment detection checks
python verify/test_buggy_envs.py

# Run a single task's verification
cd tasks/01_inventory_management && bash solution/solve.sh
```

## Requirements

- Python >= 3.10
- gymnasium >= 0.29, < 1.1
- numpy >= 1.24, < 2.0
- torch >= 2.0, < 3.0
- stable-baselines3 >= 2.0
- pytest >= 7.0

## Docker

```bash
docker compose build
docker compose run --rm app bash verify/verify_all.sh
```
