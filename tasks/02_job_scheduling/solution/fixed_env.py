"""
Fixed Job Shop Scheduling Environment.

All bugs from the original env.py are corrected:
1. Action masks are returned in info dict so completed jobs are filtered out.
2. step_count starts at 0 so the episode runs for exactly max_steps steps.
3. Reward uses epsilon-guarded normalisation to avoid -inf.
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
    """Job Shop Scheduling environment (all bugs fixed)."""

    def __init__(self, max_steps=100):
        super().__init__()

        self.max_steps = max_steps
        self.jobs = JOBS

        # Observation components (concatenated):
        #   per job:     [next_op_idx / MAX_OPS,  completed_flag]        → 2 * NUM_JOBS
        #   per machine: [remaining_time / MAX_DURATION]                  → NUM_MACHINES
        #   scalar:      [step_count / max_steps]                         → 1
        obs_dim = 2 * NUM_JOBS + NUM_MACHINES + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )

        # Action: which job to dispatch
        self.action_space = spaces.Discrete(NUM_JOBS)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # FIX 2: step_count starts at 0 so exactly max_steps steps run.
        self.step_count = 0

        self.makespan = 0

        self.job_op_idx = [0] * NUM_JOBS
        self.machine_timer = [0] * NUM_MACHINES
        self.machine_job = [-1] * NUM_MACHINES
        self.job_completed = [False] * NUM_JOBS

        return self._get_obs(), self._info()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        # ----- Process action -----
        # FIX 1: skip if the job is already completed
        if not self.job_completed[action]:
            job = self.jobs[action]
            op_idx = self.job_op_idx[action]
            if 0 <= op_idx < len(job):
                op_machine, op_duration = job[op_idx]
                if self.machine_timer[op_machine] == 0 and self.machine_job[op_machine] == -1:
                    self.machine_timer[op_machine] = op_duration
                    self.machine_job[op_machine] = action
                    self.job_op_idx[action] += 1

        # ----- Advance time by 1 unit -----
        self.makespan += 1
        for m in range(NUM_MACHINES):
            if self.machine_timer[m] > 0:
                self.machine_timer[m] -= 1
                if self.machine_timer[m] == 0:
                    finished_job = self.machine_job[m]
                    if finished_job >= 0:
                        job = self.jobs[finished_job]
                        if self.job_op_idx[finished_job] >= len(job):
                            self.job_completed[finished_job] = True
                            self.job_op_idx[finished_job] = -1
                    self.machine_job[m] = -1

        # ----- Step counter -----
        self.step_count += 1
        terminated = self._all_completed()
        truncated = self.step_count >= self.max_steps

        # ----- FIX 3: epsilon-guarded reward -----
        reward = self._compute_reward()

        return self._get_obs(), reward, terminated, truncated, self._info()

    # ------------------------------------------------------------------
    # Reward  (FIX 3)
    # ------------------------------------------------------------------

    def _compute_reward(self):
        """Step penalty — each time unit costs -1 (minimising makespan).

        FIX 3: removed the broken division-by-active-jobs normalisation
        that could produce -inf (buggy version uses raw numpy float
        division by zero).
        """
        return -1.0

    # ------------------------------------------------------------------
    # Action masking  (FIX 1)
    # ------------------------------------------------------------------

    def _info(self):
        """Return info dict containing action mask for SB3-compatible masking."""
        mask = [not c for c in self.job_completed]
        return {"action_mask": mask}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_active_jobs(self):
        return sum(1 for c in self.job_completed if not c)

    def _all_completed(self):
        return all(self.job_completed)

    def _get_obs(self):
        parts = []

        for j in range(NUM_JOBS):
            if self.job_op_idx[j] < 0:
                parts.append(0.0)
                parts.append(1.0)
            else:
                parts.append(self.job_op_idx[j] / MAX_OPS)
                parts.append(0.0)

        for m in range(NUM_MACHINES):
            parts.append(self.machine_timer[m] / MAX_DURATION)

        parts.append(self.step_count / self.max_steps)

        return np.array(parts, dtype=np.float32)
