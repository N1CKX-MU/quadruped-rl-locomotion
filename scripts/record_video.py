"""Record demo videos of the trained policy."""

import argparse
import os
import sys

import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser(description="Record Go2 policy video")
    parser.add_argument(
        "--model", type=str, default="models/go2_ppo_final.zip",
    )
    parser.add_argument(
        "--vec-normalize", type=str, default="models/vec_normalize.pkl",
    )
    parser.add_argument("--output", type=str, default="assets/demo.mp4")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Create env in rgb_array mode for recording
    raw_env = Go2Env(render_mode="rgb_array")
    env = DummyVecEnv([lambda: raw_env])
    if os.path.exists(args.vec_normalize):
        env = VecNormalize.load(args.vec_normalize, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args.model)

    frames = []
    obs = env.reset()
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frame = raw_env.render()
        if frame is not None:
            frames.append(frame)
        if done[0]:
            obs = env.reset()

    imageio.mimsave(args.output, frames, fps=25)
    print(f"Video saved to {args.output} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
