"""Load and evaluate a trained Go2 policy."""

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Go2 policy")
    parser.add_argument(
        "--model", type=str, default="models/go2_ppo_final.zip",
        help="Path to trained model"
    )
    parser.add_argument(
        "--vec-normalize", type=str, default="models/vec_normalize.pkl",
        help="Path to VecNormalize stats"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    render_mode = "human" if args.render else None

    env = DummyVecEnv([lambda: Go2Env(render_mode=render_mode)])
    if os.path.exists(args.vec_normalize):
        env = VecNormalize.load(args.vec_normalize, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args.model)

    rewards = []
    for ep in range(args.episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
        rewards.append(total_reward)
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")

    print(f"\nMean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    env.close()


if __name__ == "__main__":
    main()
