"""
Minimal PPO training script for the fixed Ad Bidding environment.

Trains briefly and reports average reward to verify the environment
works correctly with Stable-Baselines3.
"""

import numpy as np
from stable_baselines3 import PPO
from fixed_env import AdBiddingEnv, N_CHANNELS


def evaluate(env, model, episodes=20):
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


def random_baseline(env, episodes=20):
    """Evaluate a random policy."""
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total += reward
        rewards.append(total)
    return float(np.mean(rewards))


def main():
    env = AdBiddingEnv(max_steps=50)

    # Train a PPO agent
    model = PPO("MlpPolicy", env, verbose=0, n_steps=2048)
    model.learn(total_timesteps=16384)

    trained_reward = evaluate(env, model)
    print(f"Trained average reward: {trained_reward:.2f}")

    # Baseline: random policy
    random_reward = random_baseline(env)
    print(f"Random average reward:   {random_reward:.2f}")

    improvement = trained_reward - random_reward
    print(f"Improvement:             {improvement:+.2f}")
    if improvement > 0:
        print(">>> Training check PASSED -- trained policy outperforms random")
    else:
        print(">>> Training check WARNING: no improvement over random")


if __name__ == "__main__":
    main()
