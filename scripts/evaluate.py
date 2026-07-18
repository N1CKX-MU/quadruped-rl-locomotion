"""Evaluate a trained Go2 policy and report metrics."""

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
    parser.add_argument("--model", type=str, default="models/go2_ppo_final.zip")
    parser.add_argument("--vec-normalize", type=str, default="models/vecnormalize_final.pkl")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--cmd-vel", type=float, nargs=3, default=[0.8, 0.0, 0.0])
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    cmd_vel = tuple(args.cmd_vel)

    env = DummyVecEnv([lambda: Go2Env(render_mode=render_mode, cmd_vel=cmd_vel)])
    if os.path.exists(args.vec_normalize):
        env = VecNormalize.load(args.vec_normalize, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args.model)

    # Metrics accumulators
    ep_rewards = []
    ep_lengths = []
    ep_velocities = []
    ep_survived = 0

    for ep in range(args.episodes):
        obs = env.reset()
        total_reward = 0.0
        velocities = []
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            if "x_velocity" in info[0]:
                velocities.append(info[0]["x_velocity"])
            steps += 1

        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        ep_velocities.append(np.mean(velocities) if velocities else 0.0)
        if steps >= 1000:
            ep_survived += 1

        print(f"Episode {ep + 1:3d}: reward={total_reward:8.2f}  "
              f"steps={steps:4d}  avg_vel={ep_velocities[-1]:.3f} m/s")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 60)
    print(f"{'Episodes':<30} {args.episodes:>15d}")
    print(f"{'Mean reward':<30} {np.mean(ep_rewards):>15.2f}")
    print(f"{'Std reward':<30} {np.std(ep_rewards):>15.2f}")
    print(f"{'Mean episode length':<30} {np.mean(ep_lengths):>15.1f}")
    print(f"{'Survival rate':<30} {ep_survived/args.episodes*100:>14.1f}%")
    print(f"{'Mean forward speed':<30} {np.mean(ep_velocities):>13.3f} m/s")
    print(f"{'Std forward speed':<30} {np.std(ep_velocities):>13.3f} m/s")
    print(f"{'Action smoothness (approx)':<30} {'N/A':>15}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
