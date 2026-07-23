"""Record a video showing the robot recovering from external pushes.

Applies strong velocity impulses at regular intervals and captures
the robot's recovery behavior.
"""

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
    parser = argparse.ArgumentParser(description="Record push recovery demo")
    parser.add_argument("--model", type=str, default="models/go2_ppo_final.zip")
    parser.add_argument("--vec-normalize", type=str, default="models/vecnormalize_final.pkl")
    parser.add_argument("--output", type=str, default="assets/go2_push_recovery.gif")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--push-interval", type=int, default=80,
                        help="Steps between pushes")
    parser.add_argument("--push-force", type=float, default=2.0,
                        help="Push velocity impulse magnitude (m/s)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    raw_env = Go2Env(render_mode="rgb_array", cmd_vel=(0.8, 0.0, 0.0))
    env = DummyVecEnv([lambda: raw_env])
    if os.path.exists(args.vec_normalize):
        env = VecNormalize.load(args.vec_normalize, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args.model)

    frames = []
    obs = env.reset()
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        # Apply lateral push at regular intervals
        if step > 0 and step % args.push_interval == 0:
            direction = 1.0 if (step // args.push_interval) % 2 == 0 else -1.0
            raw_env.data.qvel[1] += direction * args.push_force  # Lateral push
            raw_env.data.qvel[0] += np.random.uniform(-0.5, 0.5)  # Small forward perturbation
            print(f"  Push at step {step}: lateral={'right' if direction > 0 else 'left'}")

        frame = raw_env.render()
        if frame is not None:
            small = frame[::2, ::2]
            frames.append(small)

        if done[0]:
            print(f"  Fell at step {step}")
            obs = env.reset()

    if frames:
        imageio.mimsave(args.output, frames[::2], fps=12, loop=0)
        print(f"\nGIF saved: {args.output} ({len(frames[::2])} frames)")

    env.close()


if __name__ == "__main__":
    main()
