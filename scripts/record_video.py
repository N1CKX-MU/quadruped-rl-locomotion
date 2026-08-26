"""Record a video of the trained policy.

    python scripts/record_video.py
    python scripts/record_video.py --script demo         # a scripted command tour
    python scripts/record_video.py --cmd 1.0 0 0 --steps 500

The `--script demo` mode is the one worth using: it drives a sequence of
commands — forward, strafe, turn, backward, stop — so a single clip shows the
capability v1 did not have. A video of the robot walking in a straight line does
not distinguish v2 from v1.
"""

import argparse
import os
import pickle
import sys

import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.go2_env import Go2Env  # noqa: E402

# (seconds, vx, vy, yaw, label)
DEMO_SCRIPT = [
    (2.0, 0.0, 0.0, 0.0, "stand"),
    (4.0, 1.0, 0.0, 0.0, "forward"),
    (3.0, 0.0, 0.6, 0.0, "strafe left"),
    (3.0, 0.0, -0.6, 0.0, "strafe right"),
    (3.0, 0.0, 0.0, 1.2, "turn in place"),
    (3.0, -0.7, 0.0, 0.0, "backward"),
    (3.0, 0.8, 0.0, 0.8, "arc"),
    (2.0, 0.0, 0.0, 0.0, "stop"),
]


def load_policy(model_path, vecnormalize_path, device="cpu"):
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device=device)
    obs_rms, clip_obs, eps = None, 10.0, 1e-8
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        with open(vecnormalize_path, "rb") as f:
            vec = pickle.load(f)
        obs_rms, clip_obs, eps = vec.obs_rms, vec.clip_obs, vec.epsilon
        print("loaded observation normalisation from " + vecnormalize_path)
    else:
        print("WARNING: no VecNormalize statistics found; the policy will not walk.")

    def predict(obs):
        x = obs
        if obs_rms is not None:
            x = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + eps),
                        -clip_obs, clip_obs).astype(np.float32)
        a, _ = model.predict(x, deterministic=True)
        return a

    return predict


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="models/go2_ppo_final.zip")
    p.add_argument("--vec-normalize", default=None)
    p.add_argument("--xml", default="mujoco_menagerie/unitree_go2/scene.xml")
    p.add_argument("--output", default="assets/go2_walking.mp4")
    p.add_argument("--gif", default="assets/go2_walking.gif")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--cmd", type=float, nargs=3, default=[1.0, 0.0, 0.0])
    p.add_argument("--gait", default="trot")
    p.add_argument("--script", choices=("none", "demo"), default="demo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif-stride", type=int, default=5,
                   help="Keep every Nth frame in the GIF.")
    p.add_argument("--gif-scale", type=int, default=2,
                   help="Downsample the GIF by this factor in each axis. A "
                        "full-resolution 20-second GIF is ~40 MB, which has no "
                        "business being committed to a git repository.")
    args = p.parse_args()

    vecnorm = args.vec_normalize
    if vecnorm is None:
        for guess in ("models/go2_ppo_vecnormalize.pkl",
                      "models/vecnormalize_final.pkl"):
            if os.path.exists(guess):
                vecnorm = guess
                break

    predict = load_policy(args.model, vecnorm)
    env = Go2Env(xml_path=args.xml, render_mode="rgb_array",
                 randomize_dynamics=False, push_enabled=False,
                 command_resample_interval_s=0.0, max_episode_steps=10 ** 9)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fps = int(round(1.0 / env.dt))

    obs, _ = env.reset(seed=args.seed)
    frames = []

    if args.script == "demo":
        segments = DEMO_SCRIPT
    else:
        segments = [(args.steps * env.dt, args.cmd[0], args.cmd[1], args.cmd[2],
                     "command")]

    for seconds, vx, vy, yaw, label in segments:
        moving = abs(vx) > 0.05 or abs(vy) > 0.05 or abs(yaw) > 0.05
        env.set_command(lin_vel_x=vx, lin_vel_y=vy, ang_vel_yaw=yaw,
                        gait=args.gait if moving else "stand")
        print("%-14s vx=%+.2f vy=%+.2f yaw=%+.2f  (%.0fs)"
              % (label, vx, vy, yaw, seconds))
        for _ in range(int(seconds / env.dt)):
            obs, _, terminated, _, _ = env.step(predict(obs))
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if terminated:
                print("  fell - resetting")
                obs, _ = env.reset()
                env.set_command(lin_vel_x=vx, lin_vel_y=vy, ang_vel_yaw=yaw,
                                gait=args.gait if moving else "stand")

    if not frames:
        print("No frames captured.")
        env.close()
        return

    imageio.mimsave(args.output, frames, fps=fps)
    print("video saved: %s (%d frames, %d fps)" % (args.output, len(frames), fps))
    if args.gif:
        k, sc = max(args.gif_stride, 1), max(args.gif_scale, 1)
        small = [f[::sc, ::sc] for f in frames[::k]]
        imageio.mimsave(args.gif, small, fps=max(fps // k, 1), loop=0)
        size_mb = os.path.getsize(args.gif) / 1e6
        print("gif saved:   %s (%d frames, %.1f MB)"
              % (args.gif, len(small), size_mb))
    env.close()


if __name__ == "__main__":
    main()
