# Task 2: Job Shop Scheduling

## Scenario

A multi-job, multi-machine job shop scheduling problem. There are **5 jobs**
that must be processed on **3 machines**. Each job consists of a fixed sequence
of operations — each operation specifies the required machine and its processing
time.

At each discrete time step the agent selects a job to dispatch. If the job's
next operation's machine is available, the operation starts and the machine
becomes busy for the specified duration. If the machine is busy or the job is
already complete, the action is a no-op (the step is wasted).

The objective is to **minimise the makespan** (total time to complete all jobs).

## Environment API

| Property | Type | Shape / Range |
|---|---|---|
| Observation | `Box(float32)` | `(14,)` — see below |
| Action | `Discrete(5)` | `0` … `4` (job index to dispatch) |
| Reward | `float` | Always `< 0` (negative step penalty) |
| Episode end | `terminated` / `truncated` | All jobs done / `max_steps` reached |

### Observation details (14-D vector)

| Index range | Content | Description |
|---|---|---|
| `[0, 1]` | Job 0: `[next_op_idx/3, completed_flag]` | — |
| `[2, 3]` | Job 1: `[next_op_idx/3, completed_flag]` | — |
| `[4, 5]` | Job 2: `[next_op_idx/3, completed_flag]` | — |
| `[6, 7]` | Job 3: `[next_op_idx/3, completed_flag]` | — |
| `[8, 9]` | Job 4: `[next_op_idx/3, completed_flag]` | — |
| `[10, 11, 12]` | Machine 0-2: `[remaining_time/4]` | Normalised remaining duration |
| `[13]` | `step_count / max_steps` | Normalised time |

### Job definitions

| Job | Operations `(machine, duration)` |
|---|---|
| 0 | `(M0, 3)`, `(M1, 2)` |
| 1 | `(M1, 2)`, `(M2, 1)`, `(M0, 3)` |
| 2 | `(M2, 4)`, `(M0, 1)`, `(M1, 2)` |
| 3 | `(M0, 2)`, `(M2, 3)` |
| 4 | `(M1, 3)`, `(M2, 2)`, `(M0, 1)` |

## Defects in the Buggy Environment

The supplied `environment/env.py` contains three deliberate bugs:

| # | Bug | Location | Effect |
|---|---|---|---|
| 1 | **No action masking** | `step()` — action validation | Completed jobs are still selectable. Selecting one wastes a decision step, increasing the makespan. |
| 2 | **Early termination** | `reset()` — step counter init | `step_count` starts at 1 instead of 0, causing the episode to end one step early (`max_steps - 1` steps total). |
| 3 | **Reward normalisation explosion** | `_compute_reward()` | Reward divides by active job count using raw numpy floats. When all jobs complete, `active = 0` produces `-inf` silently, destabilising training. |

## How to Verify a Fix

```bash
bash solution/solve.sh
```

This runs:
1. Black-box pytest suite (observation spaces, action masks, episode length,
   reward sanity, seed reproducibility, 100-episode stress test).
2. A short PPO training loop that checks the trained policy outperforms a
   random baseline.

All tests interact with the environment **only through the public Gymnasium
API** — no internal state is inspected.
