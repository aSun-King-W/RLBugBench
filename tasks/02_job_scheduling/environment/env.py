"""
Job Shop Scheduling Environment (Buggy Version)

Multi-job, multi-machine scheduling environment containing 3 deliberately
introduced defects for RL agent repair tasks.

Defects:
1. No action masking: completed jobs can still be selected (wasted steps)
2. Early termination: step_count starts at 1 instead of 0 (off-by-one)
3. Reward normalization explosion: divisor can be zero via numpy (produces -inf)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# 5 jobs x 3 machines — each job has 2-3 operations (machine_id, duration)
JOBS = [
    [(0, 3), (1, 2)],           # Job 0
    [(1, 2), (2, 1), (0, 3)],   # Job 1
    [(2, 4), (0, 1), (1, 2)],   # Job 2
    [(0, 2), (2, 3)],           # Job 3
    [(1, 3), (2, 2), (0, 1)],   # Job 4
]

NUM_JOBS = len(JOBS)
NUM_MACHINES = 3
MAX_OPS = max(len(j) for j in JOBS)   # 3
MAX_DURATION = max(op[1] for j in JOBS for op in j)  # 4


class JobSchedulingEnv(gym.Env):
    """Job Shop Scheduling environment with deliberate bugs."""

    def __init__(self, max_steps=100):
        super().__init__()

        self.max_steps = max_steps
        self.jobs = JOBS  # fixed job definitions

        # Observation components (concatenated):
        #   per job:     [next_op_idx / MAX_OPS,  completed_flag]        → 2 * NUM_JOBS
        #   per machine: [remaining_time / MAX_DURATION]                  → NUM_MACHINES
        #   scalar:      [step_count / max_steps]                         → 1
        obs_dim = 2 * NUM_JOBS + NUM_MACHINES + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )

        # Action: which job to dispatch at this decision point
        self.action_space = spaces.Discrete(NUM_JOBS)  # 0 … 4

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- BUG 2: step_count starts at 1 instead of 0 ---
        # Fix would be: self.step_count = 0
        self.step_count = 1

        self.makespan = 0

        # Per-job: index of the next operation to start (0 = first op).
        # Set to -1 when the job is fully completed.
        self.job_op_idx = [0] * NUM_JOBS

        # Per-machine: remaining processing time (0 = idle)
        self.machine_timer = [0] * NUM_MACHINES
        # Which job is currently occupying each machine (-1 = idle)
        self.machine_job = [-1] * NUM_MACHINES

        # Whether each job is fully completed
        self.job_completed = [False] * NUM_JOBS

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        # ----- Process action -----
        # BUG 1: no action-masking check — completed jobs are accepted
        if not self.job_completed[action]:
            job = self.jobs[action]
            op_idx = self.job_op_idx[action]
            if 0 <= op_idx < len(job):
                op_machine, op_duration = job[op_idx]
                if self.machine_timer[op_machine] == 0 and self.machine_job[op_machine] == -1:
                    # Dispatch: machine starts processing this operation
                    self.machine_timer[op_machine] = op_duration
                    self.machine_job[op_machine] = action
                    self.job_op_idx[action] += 1

        # ----- Advance time by 1 unit -----
        self.makespan += 1
        for m in range(NUM_MACHINES):
            if self.machine_timer[m] > 0:
                self.machine_timer[m] -= 1
                if self.machine_timer[m] == 0:
                    # Operation finished — check if the job is now complete
                    finished_job = self.machine_job[m]
                    if finished_job >= 0:
                        job = self.jobs[finished_job]
                        if self.job_op_idx[finished_job] >= len(job):
                            # All operations of this job have been dispatched AND the
                            # last one just finished.
                            self.job_completed[finished_job] = True
                            self.job_op_idx[finished_job] = -1
                    self.machine_job[m] = -1

        # ----- Step counter -----
        self.step_count += 1
        terminated = self._all_completed()
        # BUG 2: off-by-one — step_count starts at 1 then is incremented here,
        # so after max_steps-1 steps it already triggers.
        truncated = self.step_count >= self.max_steps

        # ----- Reward: BUG 3 (normalisation explosion) -----
        reward = self._compute_reward()

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Reward  (BUG 3)
    # ------------------------------------------------------------------

    def _compute_reward(self):
        """Step penalty 'normalised' by the number of active (non-completed) jobs.

        BUG 3: the division uses raw numpy floats, so when active_jobs == 0
        the result is -inf (numpy returns inf without raising).  This poisons
        the reward signal and destabilises training.
        Fix:  reward = -1.0 / max(float(active_jobs), 1e-8)
        """
        active = self._count_active_jobs()
        # When all jobs complete, active = 0 → reward = -inf
        # Using numpy ensures this produces -inf silently instead of crashing.
        reward = np.float32(-1.0) / np.float32(active)
        return reward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_active_jobs(self):
        return sum(1 for c in self.job_completed if not c)

    def _all_completed(self):
        return all(self.job_completed)

    def _get_obs(self):
        parts = []

        # Per-job info (NUM_JOBS × 2)
        for j in range(NUM_JOBS):
            if self.job_op_idx[j] < 0:
                # Completed job
                parts.append(0.0)
                parts.append(1.0)
            else:
                parts.append(self.job_op_idx[j] / MAX_OPS)
                parts.append(0.0)

        # Machine remaining times
        for m in range(NUM_MACHINES):
            parts.append(self.machine_timer[m] / MAX_DURATION)

        # Normalised step count
        parts.append(self.step_count / self.max_steps)

        return np.array(parts, dtype=np.float32)
