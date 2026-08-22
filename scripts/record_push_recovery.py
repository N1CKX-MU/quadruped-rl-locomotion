"""Record the robot recovering from external pushes.

    python scripts/record_push_recovery.py

The push model here is the same one the environment uses during training
(docs/12-curriculum-and-domain-randomization.md, §12.8): a real force applied to
the trunk through `xfrc_applied` for a fixed duration, not an instantaneous
edit of `qvel`.

That distinction matters for a demo as much as for training. v1's version wrote
directly into `data.qvel[0:3]`, teleporting up to 3 m/s of momentum into the
robot with no force and no impulse through the contacts. What you filmed was not
a push - it was the robot's response to a discontinuity that cannot physically
occur. Here the legs feel the shove through the ground, as they would.
"""

import argparse
import os
import pickle
import sys

import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.go2_env import Go2Env  # noqa: E402


def load_policy(model_path, vecnormalize_path, device="cpu"):
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device=device)
    obs_rms, clip_obs, eps = None, 10.0, 1e-8
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        with open(vecnormalize_path, "rb") as f:
            vec = pickle.load(f)
        obs_rms, clip_obs, eps = vec.obs_rms, vec.clip_obs, vec.epsilon
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
    p.add_argument("--output", default="assets/go2_push_recovery.gif")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--push-interval", type=int, default=100,
                   help="Control steps between pushes (100 = 2 s at 50 Hz).")
    p.add_argument("--push-force", type=float, default=80.0,
                   help="Force magnitude in newtons. The robot weighs 15.2 kg, "
                        "so 80 N held for 0.15 s is about 0.8 m/s of impulse.")
    p.add_argument("--push-duration", type=float, default=0.15)
    p.add_argument("--cmd", type=float, nargs=3, default=[0.8, 0.0, 0.0])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    vecnorm = args.vec_normalize
    if vecnorm is None:
        for guess in ("models/go2_ppo_vecnormalize.pkl",
                      "models/vecnormalize_final.pkl"):
            if os.path.exists(guess):
                vecnorm = guess
                break

    predict = load_policy(args.model, vecnorm)
    # The environment's own push machinery is disabled so this script controls
    # exactly when the pushes happen, which is what makes a legible demo.
    env = Go2Env(xml_path=args.xml, render_mode="rgb_array",
                 randomize_dynamics=False, push_enabled=False,
                 command_resample_interval_s=0.0, max_episode_steps=10 ** 9)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fps = int(round(1.0 / env.dt))

    obs, _ = env.reset(seed=args.seed)
    env.set_command(lin_vel_x=args.cmd[0], lin_vel_y=args.cmd[1],
                    ang_vel_yaw=args.cmd[2], gait="trot")

    frames = []
    push_steps_left = 0
    survived = 0
    pushes = 0
    rng = np.random.default_rng(args.seed)

    for step in range(args.steps):
        if step > 0 and step % args.push_interval == 0:
            # Alternate the side so the clip shows recovery in both directions.
            direction = 1.0 if pushes % 2 == 0 else -1.0
            force = np.array([
                float(rng.uniform(-0.3, 0.3)) * args.push_force,
                direction * args.push_force,
                0.0,
            ])
            env.data.xfrc_applied[env.base_body_id, :3] = force
            push_steps_left = int(args.push_duration / env.dt)
            pushes += 1
            print("push %d at step %4d: %s N" % (pushes, step, np.round(force, 1)))

        if push_steps_left > 0:
            push_steps_left -= 1
            if push_steps_left == 0:
                env.data.xfrc_applied[env.base_body_id, :3] = 0.0

        obs, _, terminated, _, _ = env.step(predict(obs))
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated:
            print("  fell at step %d - resetting" % step)
            env.data.xfrc_applied[:] = 0.0
            push_steps_left = 0
            obs, _ = env.reset()
            env.set_command(lin_vel_x=args.cmd[0], lin_vel_y=args.cmd[1],
                            ang_vel_yaw=args.cmd[2], gait="trot")
        else:
            survived += 1

    print("\nsurvived %d of %d steps across %d pushes" % (survived, args.steps, pushes))
    if frames:
        imageio.mimsave(args.output, frames[::3], fps=fps // 3, loop=0)
        print("saved " + args.output)
    env.close()


if __name__ == "__main__":
    main()
