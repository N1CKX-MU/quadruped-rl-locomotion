"""Verification script: load Go2 env, step random actions, print diagnostics."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.go2_env import Go2Env


def main():
    print("Loading Go2 environment...")
    env = Go2Env()

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape:      {env.action_space.shape}")
    print(f"Action range:      [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    print(f"Num actuators:     {env.n_actuators}")
    print(f"Control dt:        {env.dt:.4f}s ({1/env.dt:.0f} Hz)")
    print(f"Default joint pos: {env.default_joint_pos}")
    print(f"Cmd velocity:      {env.cmd_vel}")
    print()

    total_reward = 0.0
    episodes = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            episodes += 1
            print(f"  Episode ended at step {step + 1} "
                  f"(terminated={terminated}, height={info.get('body_height', 'N/A'):.3f})")
            obs, info = env.reset()

    print(f"\n100 random steps completed ({episodes} episode resets).")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Reward components from last step: {info}")

    env.close()
    print("\nVerification passed!")


if __name__ == "__main__":
    main()
