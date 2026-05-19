# Task 4: Cache Replacement Policy

## Scenario

A **CDN (Content Delivery Network) cache** with limited storage capacity.
The agent manages which content items are kept in the cache. At each step,
a content request arrives. If the requested item is already cached, a **hit**
occurs (reward +1). If not, it is a **miss** (reward 0), and the agent must
**evict** one of the currently cached items to make room for the new content.

Content requests follow a **Zipf-like popularity distribution** (a small number
of popular items account for most requests) with **temporal locality** (recently
seen items are more likely to appear again soon).

The objective is to **maximise the cache hit rate** -- the fraction of requests
served from the cache rather than the origin server.

## Environment API

| Property | Type | Shape / Range |
|---|---|---|
| Observation | `Box(float32)` | `(41,)` -- see below |
| Action | `Discrete(10)` | `0` ... `9` (cache slot index to evict) |
| Reward | `float` | `+1` on cache hit, `0` on miss |
| Episode end | `truncated` | After `max_steps` steps (default 200) |

### Observation details (41-D vector)

The cache has **10 slots**. Each slot contributes **4 features**:

| Feature | Description |
|---|---|
| `content_id / N_CONTENTS` | Normalised content identifier |
| `min(access_count / 100, 1.0)` | How many times this content has been requested ever |
| `min(time_since_last_access / max_steps, 1.0)` | Recency of last access (0 = just accessed) |
| `min(frequency / 50, 1.0)` | Access frequency in a recent sliding window |

The final element `[40]` is the **current request id** (normalised).

The observation contains enough information to implement standard policies:
- **LRU** (evict the slot with highest `time_since_last_access`)
- **LFU** (evict the slot with lowest `access_count` or `frequency`)
- **Adaptive** (learn which feature combination predicts future misses)

### Action

The agent selects a **slot index** (`0` -- `9`). On a cache miss, the chosen
slot is evicted and replaced with the newly requested content. On a cache hit,
the action is ignored (no eviction occurs).

### Dynamics

- The cache is pre-filled with random content at the start of each episode.
- Requests are generated from a Zipf-like distribution: item 0 is ~100× more
  popular than item 99.
- 30 % of requests exhibit temporal locality (repeat a request from the last
  20 steps).
- Per-content access metadata (`access_count`, `last_access_step`, `frequency`)
  is updated on every request (hit or miss).

## Defects in the Buggy Environment

The supplied `environment/env.py` contains four deliberate bugs:

| # | Bug | Location | Effect |
|---|---|---|---|
| 1 | **State missing temporal features** | `_get_obs()` -- observation vector | Observation is 11-D (only content IDs). The agent cannot distinguish frequently-accessed content from stale content. |
| 2 | **Delayed reward accumulation** | `step()` -- reward | Hit/miss rewards are accumulated over a 5-step window and distributed at window boundaries. The reward signal is temporally misaligned with the eviction action. |
| 3 | **Memory leak** | `history` list | Every `(request, action, hit)` tuple is appended to `self.history` without limit. Across many episodes the list grows unbounded, degrading performance. |
| 4 | **Termination variable error** | `step()` -- truncation | Uses `self.steps` (initialised once but never updated) instead of `self.current_step`. The episode never truncates (runs until `terminated`, which is always `False`). |

## How to Verify a Fix

```bash
bash solution/solve.sh
```

This runs:
1. Black-box pytest suite (observation spaces, action validity, episode length,
   binary reward, reproducibility, 100-episode stress test).
2. A short PPO training loop that checks the trained policy outperforms a
   random baseline.

All tests interact with the environment **only through the public Gymnasium
API** -- no internal state is inspected.
