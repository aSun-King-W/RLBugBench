# Task 3: Ad Bidding / Campaign Optimization

## Scenario

An online advertising campaign with a fixed total budget to allocate across
**4 channels** (search, social, display, video). Each channel has its own
**conversion rate** (probability of conversion per click) and **cost per click**.

At each daily step, the agent selects one channel to spend the day's budget on.
Conversion rates drift over time via a random walk, making the environment
**non-stationary** -- the optimal channel allocation changes throughout the
campaign.

The objective is to **maximise total conversions** while staying within budget.

## Environment API

| Property | Type | Shape / Range |
|---|---|---|
| Observation | `Box(float32)` | `(6,)` -- see below |
| Action | `Discrete(4)` | `0` ... `3` (channel index) |
| Reward | `float` | Normalised (mean `~0`, std `~1`) |
| Episode end | `truncated` | After `max_steps` steps (default 100) |

### Observation details (6-D vector)

| Index | Content | Description |
|---|---|---|
| `[0, 1, 2, 3]` | Channel conversion rates | Normalised `[0, 1]`, drift over time |
| `[4]` | `remaining_budget / TOTAL_BUDGET` | Fraction of budget left |
| `[5]` | `step_count / max_steps` | Normalised time |

### Channels

| Index | Name | Base CVR | Cost per Click |
|---|---|---|---|
| 0 | Search | 8% | $0.50 |
| 1 | Social | 5% | $0.30 |
| 2 | Display | 2% | $0.10 |
| 3 | Video | 6% | $0.40 |

### Dynamics

- Daily spend is `1000`. Total campaign budget is `95000`.
- Each step, the agent chooses a channel. The spend is capped to the remaining budget.
- Clicks = `spend / cost_per_click`, conversions = `clicks * conversion_rate`.
- After each step, conversion rates drift via a random walk `N(0, 0.02)`.
- When the budget is exhausted, no further spending occurs.

## Defects in the Buggy Environment

The supplied `environment/env.py` contains four deliberate bugs:

| # | Bug | Location | Effect |
|---|---|---|---|
| 1 | **Budget leakage** | `step()` -- spend cap | Always spends `daily_spend` even when `remaining_budget < daily_spend`. Total campaign spend can exceed the total budget. |
| 2 | **State missing `remaining_budget`** | `_get_obs()` -- observation vector | Observation is 5-D (missing budget info). The agent cannot tell how much budget remains and cannot plan spending. |
| 3 | **Non-stationary reward not normalised** | `step()` -- reward | Raw conversion counts are used as reward. Since CVRs drift over time, the reward distribution shifts, destabilising learning. |
| 4 | **Seed not fixed** | `reset()` -- RNG seeding | `super().reset(seed=seed)` is not called. The random number generator is not seeded, making trajectories non-reproducible. |

## How to Verify a Fix

```bash
bash solution/solve.sh
```

This runs:
1. Black-box pytest suite (observation spaces, budget constraint, episode length,
   reward sanity, seed reproducibility, 100-episode stress test).
2. A short PPO training loop that checks the trained policy outperforms a
   random baseline.

All tests interact with the environment **only through the public Gymnasium
API** -- no internal state is inspected.
