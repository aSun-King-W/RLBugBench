# RL Environment Bug-Fixing Benchmark

A curated benchmark for evaluating LLM agents on real-world reinforcement learning
engineering tasks. Four Gymnasium environments with deliberately crafted bugs test
an agent's ability to debug, fix, and verify RL code.

## Tasks

| # | Task | Domain | Bugs | Difficulty |
|---|------|--------|------|-----------|
| 1 | [Inventory Management](tasks/01_inventory_management/) | Supply chain | 3 | Beginner |
| 2 | [Job Shop Scheduling](tasks/02_job_scheduling/) | Manufacturing | 3 | Intermediate |
| 3 | [Ad Bidding / Campaign Optimization](tasks/03_ad_bidding/) | Advertising | 4 | Advanced |
| 4 | [Cache Replacement Policy](tasks/04_cache_policy/) | CDN / Systems | 4 | Expert |

**Bug categories**: reward function errors, state/action space design flaws, termination
logic bugs, numerical stability issues, performance traps, and reproducibility gaps.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all task verifications
bash verify/verify_all.sh

# Confirm buggy environments are detectably broken
python verify/test_buggy_envs.py
```

## Project Structure

```
.
+-- tasks/
|   +-- 01_inventory_management/
|   |   +-- environment/env.py          # Buggy environment
|   |   +-- solution/
|   |       +-- fixed_env.py            # Corrected environment
|   |       +-- train.py                # Minimal PPO training
|   |       +-- tests/test_env.py       # Black-box pytest suite
|   |       +-- solve.sh                # One-shot verification
|   |   +-- README.md                   # Task description (English)
|   +-- 02_job_scheduling/              # Same structure
|   +-- 03_ad_bidding/                  # Same structure
|   +-- 04_cache_policy/                # Same structure
+-- verify/
|   +-- verify_all.sh                   # Sequential verification runner
|   +-- test_buggy_envs.py              # Bug detection validation
+-- Dockerfile                          # python:3.10-slim based
+-- docker-compose.yml                  # One-command dev environment
+-- requirements.txt                    # Pinned dependencies
+-- TASK.md                             # English master description
+-- execution_plan.md                   # Chinese execution plan
```

## How It Works

Each task provides a **buggy environment** (`environment/env.py`) and a **fixed solution**
(`solution/fixed_env.py`). The fixed solution ships with black-box pytest tests and a
PPO training script. The buggy environment is designed to:

- Run without crashing (no unhandled exceptions)
- Produce measurably worse RL training results than the fixed version
- Have each bug detectable through the public Gymnasium API

A capable LLM should be able to:
1. Understand the environment semantics from code inspection
2. Identify and fix each bug
3. Verify fixes pass the provided tests
4. Train a working RL agent on the corrected environment

## Technical Stack

- **RL interface**: Gymnasium >= 0.29
- **Deep learning**: PyTorch >= 2.0
- **RL library**: Stable-Baselines3 >= 2.0 (PPO)
- **Testing**: pytest >= 7.0
- **Containerisation**: Docker (python:3.10-slim)

## Verification

```bash
# Full pipeline via Docker
docker compose build
docker compose run --rm app bash verify/verify_all.sh

# Expected output:
# >>> [01_inventory_management] PASSED
# >>> [02_job_scheduling] PASSED
# >>> [03_ad_bidding] PASSED
# >>> [04_cache_policy] PASSED
# All tasks passed. Exit code 0.
```
