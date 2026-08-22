"""Fast sanity check on the environment. Run this before every training run.

    python scripts/check_env.py

A training run that fails is expensive to diagnose because everything happens
slowly and out of sight. Most of what actually goes wrong is visible in ten
seconds of random actions: NaNs, a nominal pose the robot cannot hold, a reward
term with the wrong sign, foot contacts wired to the wrong geoms. This script
looks for exactly those.

It is deliberately not a pytest file. It prints numbers you read, rather than
assertions you only notice when they fail.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from envs.go2_env import Go2Env  # noqa: E402
from envs.gait import desired_contact, gait_params  # noqa: E402


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_config.yaml")
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_cfg = dict(config["environment"])
    env_cfg["reward_weights"] = (config.get("reward") or {}).get("weights")
    env_cfg["randomize_dynamics"] = False
    env_cfg["push_enabled"] = False
    env = Go2Env(**env_cfg)

    section("Model")
    print("physics dt      : %.4f s" % env.physics_dt)
    print("decimation      : %d" % env.decimation)
    print("control rate    : %.1f Hz" % (1.0 / env.dt))
    print("observation dim : %d" % env.obs_dim)
    print("action dim      : %d" % env.n_joints)
    print("foot geoms      : %s -> ids %s"
          % (list(env.FOOT_GEOM_NAMES), env.foot_geom_ids.tolist()))
    print("torque limits   : %s" % np.round(env.torque_limits, 1).tolist())

    section("Nominal pose (bug B1)")
    print("default_joint_pos : %s" % np.round(env.default_joint_pos, 3).tolist())
    print("nominal height    : %.3f m" % env.nominal_base_height)
    expected = np.tile([0.0, 0.9, -1.8], 4)
    ok = np.allclose(env.default_joint_pos, expected, atol=1e-6)
    print("matches Go2 home  : %s" % ("YES" if ok else "NO  <-- B1 has regressed"))

    section("Standing test")
    env.reset(seed=0)
    env.data.qpos[7:] = env.default_joint_pos
    env.data.qpos[2] = env.nominal_base_height
    env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qvel[:] = 0.0
    env.set_command(lin_vel_x=0, lin_vel_y=0, ang_vel_yaw=0, gait="stand")
    heights, contacts = [], []
    fell = False
    for _ in range(150):
        _, _, terminated, _, info = env.step(np.zeros(12))
        heights.append(info["base_height"])
        contacts.append(info["contacts"].sum())
        if terminated:
            fell = True
            break
    if fell:
        print("FAILED: the robot fell over while asked to stand still.")
    else:
        h = np.array(heights[50:])
        print("height mean/std : %.3f / %.4f m" % (h.mean(), h.std()))
        print("mean feet down  : %.2f / 4" % np.mean(contacts[50:]))
        print("verdict         : %s"
              % ("OK" if h.std() < 0.02 and h.mean() > 0.2 else "UNSTABLE"))

    section("Gait schedule")
    offsets, duty = gait_params("trot")
    rows = []
    for phase in np.linspace(0, 1, 8, endpoint=False):
        d = desired_contact(phase, offsets, duty)
        rows.append("  phi=%.2f  " % phase + " ".join(
            ("[]" if x else "..") for x in d))
    print("  " + "     ".join(env.FOOT_GEOM_NAMES))
    print("\n".join(rows))

    section("Random rollout (%d steps)" % args.steps)
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(1)
    rewards = []
    term_sums = {}
    episodes = 0
    lengths = []
    length = 0
    bad = []
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 12))
        length += 1
        if not np.all(np.isfinite(obs)):
            bad.append("non-finite observation")
        if not np.isfinite(reward):
            bad.append("non-finite reward")
        rewards.append(reward)
        for k, v in info.items():
            if k.startswith("rew/"):
                term_sums[k] = term_sums.get(k, 0.0) + v
                if not np.isfinite(v):
                    bad.append("non-finite " + k)
        if terminated or truncated:
            episodes += 1
            lengths.append(length)
            length = 0
            obs, _ = env.reset()

    print("mean reward/step : %+.4f" % np.mean(rewards))
    print("episodes         : %d" % episodes)
    if lengths:
        print("mean length      : %.1f steps (%.2f s)"
              % (np.mean(lengths), np.mean(lengths) * env.dt))
    print("obs range        : [%.2f, %.2f]" % (obs.min(), obs.max()))

    print("\nreward decomposition (mean per step):")
    # Share of the total *magnitude*, not of the net sum. Dividing by the net
    # sum is meaningless when carrots and sticks nearly cancel - you get shares
    # over 100%, which tells you nothing about which term is loud.
    magnitude = sum(abs(v) for v in term_sums.values()) or 1.0
    for name, value in sorted(term_sums.items(), key=lambda kv: -abs(kv[1])):
        print("  %-28s %+9.5f   (%5.1f%% of gross)"
              % (name[len("rew/"):], value / args.steps,
                 abs(value) / magnitude * 100))
    print("  %-28s %+9.5f" % ("NET", sum(term_sums.values()) / args.steps))

    section("Reward-term coverage")
    logged = {k[len("rew/"):] for k in term_sums}
    configured = {k for k, w in env.reward_weights.items() if w != 0.0}
    missing = configured - logged
    extra = logged - configured
    print("configured non-zero terms : %d" % len(configured))
    print("terms seen in info        : %d" % len(logged))
    if missing:
        print("MISSING FROM INFO         : %s" % sorted(missing))
    if extra:
        print("UNEXPECTED IN INFO        : %s" % sorted(extra))

    section("Result")
    if bad:
        print("FAIL: " + "; ".join(sorted(set(bad))))
        env.close()
        return 1
    if missing or extra or not ok:
        print("FAIL: see the flagged sections above.")
        env.close()
        return 1
    print("All checks passed.")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
