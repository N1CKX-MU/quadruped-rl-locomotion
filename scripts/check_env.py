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

from envs.commands import ranges_from_config  # noqa: E402
from envs.gait import desired_contact, gait_params  # noqa: E402
from envs.go2_env import Go2Env  # noqa: E402


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_config.yaml")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--ranges", choices=("initial", "final"), default="initial",
                   help="Which command ranges to check. 'initial' is what the "
                        "curriculum starts from and is the one that decides "
                        "whether the task is learnable at all.")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_cfg = dict(config["environment"])
    cmd_cfg = config.get("commands", {}) or {}
    # Build the env exactly as training would, so the checks below report the
    # command distribution the policy will actually see.
    env_cfg.update(
        reward_weights=(config.get("reward") or {}).get("weights"),
        command_ranges=ranges_from_config(cmd_cfg.get(args.ranges)),
        stand_probability=cmd_cfg.get("stand_probability", 0.10),
        command_resample_interval_s=cmd_cfg.get("resample_interval_s", 5.0),
        gaits=tuple(cmd_cfg.get("gaits", ["trot"])),
        feasibility_margin=cmd_cfg.get("feasibility_margin", 0.85),
        randomize_dynamics=False,
        push_enabled=False,
    )
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

    section("Speed envelope (bug B20)")
    stride = env.max_stride_length
    sampler = env.command_sampler
    cr = sampler.ranges
    margin = sampler.feasibility_margin
    print("action scale        : %.2f rad" % env.action_scale)
    print("PD offset torque    : %.1f Nm at |action|=1 (limit %.1f)"
          % (env.kp * env.action_scale, env.torque_limits[0]))
    print("stance foot travel  : %.1f cm  (static kinematic sweep)" % (stride * 100))
    print()
    print("The static sweep is a LOWER bound - body pitch, foot roll and dynamic")
    print("effects add stride it cannot see. The sampler uses measured limits:")
    print("  max_speed_per_hz  : %s" % sampler.max_speed_per_hz)
    print("  max_speed         : %s" % sampler.max_speed)
    print("  margin            : %.2f" % margin)
    print()
    print("%-9s %12s %14s" % ("frequency", "static bound", "sampler limit"))
    for f in (1.5, 2.0, 2.5, 3.0):
        v = float("inf")
        if sampler.max_speed_per_hz:
            v = sampler.max_speed_per_hz * f
        if sampler.max_speed:
            v = min(v, sampler.max_speed)
        print("%7.1f Hz %11.2f %14.2f" % (f, stride * f, v * margin))
    asked = max(abs(cr.lin_vel_x[0]), abs(cr.lin_vel_x[1]))
    v = float("inf")
    if sampler.max_speed_per_hz:
        v = sampler.max_speed_per_hz * cr.gait_frequency[1]
    if sampler.max_speed:
        v = min(v, sampler.max_speed)
    reachable = v * margin
    print()
    print("configured |vx| range: %.2f m/s" % asked)
    print("sampler will ask up to %.2f m/s at %.1f Hz"
          % (reachable, cr.gait_frequency[1]))
    if asked > reachable + 1e-6:
        print("NOTE: commands are clamped to the reachable value, so nothing")
        print("      infeasible is ever issued - the configured range is simply")
        print("      wider than the robot can use.")
    else:
        print("verdict             : every configured command is reachable")

    section("Do-nothing baseline")
    rng2 = np.random.default_rng(0)
    cmds = [env.command_sampler.sample(rng2) for _ in range(20000)]
    vecs = np.array([c.vec for c in cmds])
    lin_err = np.linalg.norm(vecs[:, :2], axis=1)
    ang_err = np.abs(vecs[:, 2])
    score = (0.5 * np.exp(-lin_err ** 2 / env.lin_vel_sigma)
             + 0.5 * np.exp(-ang_err ** 2 / env.ang_vel_sigma))
    print("What a policy that never moves scores on the tracking terms, over the")
    print("configured command distribution. If this is close to the curriculum's")
    print("promotion threshold, the standing local optimum is unavoidable.")
    print()
    print("do-nothing tracking score : %.3f" % score.mean())
    print("curriculum promotes above : 0.85")
    print("verdict                   : %s"
          % ("OK" if score.mean() < 0.70 else "TOO HIGH - see docs/10, 10.4"))

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
