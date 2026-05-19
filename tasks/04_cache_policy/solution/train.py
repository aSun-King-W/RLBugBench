"""
Minimal PPO training script for the fixed Cache Replacement environment.

Trains briefly and reports average hit rate to verify the environment
works correctly with Stable-Baselines3.
"""

import numpy as np
from stable_baselines3 import PPO
from fixed_env import CacheEnv, CACHE_SIZE

EVAL_SEED = 1234


def evaluate(env, model, episodes=20):
    """Evaluate a policy using fixed seeds and return the mean hit rate."""
    rates = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + ep * 7)
        total_hits = 0
        total_steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total_hits += reward
            total_steps += 1
        rates.append(float(total_hits / max(total_steps, 1)))
    return float(np.mean(rates))


def random_baseline(env, episodes=20):
    """Evaluate a random policy using fixed seeds."""
    rates = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + 999 + ep * 7)
        total_hits = 0
        total_steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total_hits += reward
            total_steps += 1
        rates.append(float(total_hits / max(total_steps, 1)))
    return float(np.mean(rates))


def main():
    env = CacheEnv(max_steps=100)

    # Train a PPO agent with fixed seed for reproducibility
    model = PPO("MlpPolicy", env, verbose=0, n_steps=2048, seed=0)
    model.learn(total_timesteps=65536)

    trained_rate = evaluate(env, model)
    print(f"Trained average hit rate: {trained_rate:.3f}")

    # Baseline: random policy
    random_rate = random_baseline(env)
    print(f"Random average hit rate:   {random_rate:.3f}")

    improvement = trained_rate - random_rate
    print(f"Improvement:               {improvement:+.3f}")
    if improvement > 0:
        print(">>> Training check PASSED -- trained policy outperforms random")
    else:
        print(">>> Training check WARNING: no improvement over random")


if __name__ == "__main__":
    main()
