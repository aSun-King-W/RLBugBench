# Task 1: Inventory Management

## Scenario

A single-product periodic review inventory system. At the beginning of each period,
the agent observes the current inventory level, a demand forecast for the period,
and any outstanding orders. The agent must decide how many units to order.

The goal is to **minimise the total cost** composed of:
- **Holding cost**: incurred when inventory is positive (cost per unit held).
- **Stockout penalty**: incurred when demand exceeds available inventory (cost per
  unit of unmet demand, typically higher than the holding cost).

## Environment API

| Property | Type | Shape / Range |
|---|---|---|
| Observation | `Box(float32)` | `(3,)` — `[inventory, demand_forecast, on_order]` |
| Action | `Discrete(21)` | `0` … `20` (units to order) |
| Reward | `float` | Always `≤ 0` (negated total cost) |
| Episode end | `truncated` | After `max_steps` steps (default 100) |

### Observation details
1. **inventory** (`[-50, 50]`): Current on-hand inventory. Negative values
   represent backlogged demand.
2. **demand_forecast** (`[0, 50]`): Forecast for the current period. Actual
   demand is sampled from `Normal(forecast, 5)`.
3. **on_order** (`[0, 20]`): Quantity arriving this period (always 0 in the
   base environment — orders arrive instantly).

## Defects in the Buggy Environment

The supplied `environment/env.py` contains three deliberate bugs:

| # | Bug | Location | Effect |
|---|---|---|---|
| 1 | **Reward sign reversed** | `step()` — reward calculation | Reward is positive when costs are high; the agent learns to *maximise* costs. |
| 2 | **Demand not subtracted** | `step()` — inventory update | `ending_inv = starting_inv + order` instead of `... + order - demand`. Inventory never decreases, making stockouts impossible. |
| 3 | **No episode termination** | `step()` — termination logic | Neither `terminated` nor `truncated` is ever set to `True`. The episode runs forever. |

## How to Verify a Fix

```bash
bash solution/solve.sh
```

This runs:
1. Black-box pytest suite (observation spaces, reward sign, episode length,
   seed reproducibility, 100-episode stress test).
2. A short PPO training loop that checks the trained policy outperforms a
   random baseline.

All tests interact with the environment **only through the public Gymnasium
API** — no internal state is inspected.
