"""
Minimal PPO training script for the fixed Inventory Management environment.

Trains briefly and reports average reward to verify the environment
works correctly with Stable-Baselines3.
"""

import numpy as np
from stable_baselines3 import PPO
from fixed_env import InventoryManagementEnv


def evaluate(env, model, episodes=10):
    """Evaluate a policy and return the mean episode reward."""
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total += reward
        rewards.append(total)
    return float(np.mean(rewards))


def main():
    env = InventoryManagementEnv(max_steps=50)

    # Train a PPO agent briefly
    model = PPO("MlpPolicy", env, verbose=0, n_steps=1024)
    model.learn(total_timesteps=4096)

    trained_reward = evaluate(env, model)
    print(f"Trained average reward: {trained_reward:.2f}")

    # Baseline: random policy
    random_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total += reward
        random_rewards.append(total)
    random_reward = float(np.mean(random_rewards))
    print(f"Random average reward:   {random_reward:.2f}")

    # The trained policy should outperform the random baseline
    improvement = trained_reward - random_reward
    print(f"Improvement:             {improvement:+.2f}")
    if improvement > 0:
        print(">>> Training check PASSED")
    else:
        print(">>> Training check WARNING: no improvement over random (short training expected)")


if __name__ == "__main__":
    main()
