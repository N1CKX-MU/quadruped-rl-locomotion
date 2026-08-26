"""Evaluate a trained Go2 policy over the whole command space.

v1's evaluation answered one question - "how fast does it go forwards?" - which
was the only question the policy could answer. A command-conditioned policy
needs a different measurement: for each commanded velocity, how close does the
achieved body-frame velocity get?

    python scripts/evaluate.py                       # summary over the envelope
    python scripts/evaluate.py --grid                # tracking-error grid + plot
    python scripts/evaluate.py --cmd 1.0 0 0         # one specific command
    python scripts/evaluate.py --render --episodes 5

The headline number is the mean tracking error per axis. A policy that only
walks forwards scores well on vx and badly on vy and yaw, which is exactly what
we want the measurement to expose.
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.go2_env import Go2Env  # noqa: E402


def load_policy(model_path, vecnormalize_path, device="cpu"):
    """Load a policy plus the observation normalisation it was trained with."""
    from stable_baselines3 import PPO, SAC, TD3

    algo = PPO
    for name, cls in (("sac", SAC), ("td3", TD3), ("ppo", PPO)):
        if name in os.path.basename(model_path).lower():
            algo = cls
            break
    model = algo.load(model_path, device=device)

    obs_rms, clip_obs, epsilon = None, 10.0, 1e-8
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        with open(vecnormalize_path, "rb") as f:
            vec = pickle.load(f)
        obs_rms, clip_obs, epsilon = vec.obs_rms, vec.clip_obs, vec.epsilon
    else:
        print("WARNING: no VecNormalize statistics; results will be meaningless "
              "if the policy was trained with normalisation (it was).")

    def predict(obs):
        x = obs
        if obs_rms is not None:
            x = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                        -clip_obs, clip_obs).astype(np.float32)
        action, _ = model.predict(x, deterministic=True)
        return action

    return predict


def rollout(env, predict, command, steps, settle_steps=50, seed=0):
    """Hold one command for ``steps`` control steps and measure what happens.

    The first ``settle_steps`` are discarded: the policy needs a stride or two
    to converge onto a newly issued command, and including that transient makes
    every command look badly tracked.
    """
    obs, _ = env.reset(seed=seed)
    env.set_command(
        lin_vel_x=command[0], lin_vel_y=command[1], ang_vel_yaw=command[2],
        gait="stand" if np.linalg.norm(command) < 0.1 else "trot",
    )

    lin, ang, heights, contacts, actions = [], [], [], [], []
    fell = False
    prev_action = np.zeros(env.n_joints)
    jerk = []

    for i in range(steps):
        action = predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        if i >= settle_steps:
            lin.append(info["lin_vel_b"][:2])
            ang.append(info["ang_vel_b"][2])
            heights.append(info["base_height"])
            contacts.append(info["contacts"])
            actions.append(action)
            jerk.append(np.mean(np.abs(action - prev_action)))
        prev_action = np.asarray(action, dtype=float)
        if terminated:
            fell = True
            break
        if truncated:
            break

    if not lin:
        return dict(fell=True, steps=0)

    lin = np.array(lin)
    ang = np.array(ang)
    return dict(
        fell=fell,
        steps=len(lin),
        achieved_vx=float(lin[:, 0].mean()),
        achieved_vy=float(lin[:, 1].mean()),
        achieved_yaw=float(ang.mean()),
        err_vx=float(abs(lin[:, 0].mean() - command[0])),
        err_vy=float(abs(lin[:, 1].mean() - command[1])),
        err_yaw=float(abs(ang.mean() - command[2])),
        height=float(np.mean(heights)),
        duty=float(np.mean(contacts)),
        action_jerk=float(np.mean(jerk)),
    )


def make_env(args):
    return Go2Env(
        xml_path=args.xml,
        render_mode="human" if args.render else None,
        randomize_dynamics=False,
        push_enabled=False,
        command_resample_interval_s=0.0,   # the harness owns the command
        max_episode_steps=args.steps + 10,
    )


def print_table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def run_grid(env, predict, args):
    """Sweep each command axis independently and report tracking error."""
    sweeps = {
        "forward/back (vx)": [(v, 0.0, 0.0) for v in
                              np.round(np.arange(-1.0, 1.51, 0.25), 2)],
        "strafe (vy)": [(0.0, v, 0.0) for v in
                        np.round(np.arange(-0.75, 0.76, 0.25), 2)],
        "turn (yaw)": [(0.0, 0.0, v) for v in
                       np.round(np.arange(-1.5, 1.51, 0.5), 2)],
        "diagonal": [(0.6, 0.4, 0.0), (0.6, -0.4, 0.0),
                     (-0.5, 0.3, 0.0), (0.8, 0.0, 0.8)],
    }

    all_rows = {}
    for title, commands in sweeps.items():
        print("\n" + title)
        rows = []
        for cmd in commands:
            r = rollout(env, predict, cmd, args.steps, seed=args.seed)
            if r["steps"] == 0 or r.get("fell") and r["steps"] < args.steps // 4:
                rows.append([f"{cmd}", "FELL", "-", "-", "-", "-"])
                continue
            rows.append([
                "(%.2f, %.2f, %.2f)" % cmd,
                "%.3f" % r["achieved_vx"],
                "%.3f" % r["achieved_vy"],
                "%.3f" % r["achieved_yaw"],
                "%.3f" % (r["err_vx"] + r["err_vy"]),
                "%.3f" % r["err_yaw"],
            ])
        print_table(rows, ["command (vx,vy,yaw)", "vx", "vy", "yaw",
                           "lin err", "ang err"])
        all_rows[title] = rows
    return all_rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="models/go2_ppo_final.zip")
    p.add_argument("--vec-normalize", default=None)
    p.add_argument("--xml", default="mujoco_menagerie/unitree_go2/scene.xml")
    p.add_argument("--episodes", type=int, default=20,
                   help="Random commands drawn for the summary statistics.")
    p.add_argument("--steps", type=int, default=400,
                   help="Control steps held per command (400 = 8 s at 50 Hz).")
    p.add_argument("--cmd", type=float, nargs=3, default=None,
                   help="Evaluate a single command instead of a sweep.")
    p.add_argument("--grid", action="store_true",
                   help="Sweep each command axis and tabulate tracking error.")
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--config", default="configs/training_config.yaml",
                   help="Read the trained command envelope from here, so the "
                        "random-command summary samples what the policy was "
                        "actually trained on.")
    p.add_argument("--full-envelope", action="store_true",
                   help="Sample beyond the trained ranges, to measure how the "
                        "policy degrades outside them.")
    args = p.parse_args()

    vecnorm = args.vec_normalize
    if vecnorm is None:
        for guess in ("models/go2_ppo_vecnormalize.pkl",
                      "models/vecnormalize_final.pkl"):
            if os.path.exists(guess):
                vecnorm = guess
                break

    predict = load_policy(args.model, vecnorm, args.device)
    env = make_env(args)

    try:
        if args.cmd is not None:
            r = rollout(env, predict, tuple(args.cmd), args.steps, seed=args.seed)
            for k, v in r.items():
                print("%-14s %s" % (k, v))
            return

        if args.grid:
            run_grid(env, predict, args)
            return

        # Random commands. By default these come from the config's trained
        # ranges: sampling wider than the policy was trained on measures
        # extrapolation, not tracking, and quietly inflates every error.
        rng = np.random.default_rng(args.seed)
        vx_r, vy_r, yaw_r = (-1.0, 1.5), (-0.7, 0.7), (-1.5, 1.5)
        if not args.full_envelope and os.path.exists(args.config):
            import yaml

            with open(args.config) as f:
                final = ((yaml.safe_load(f).get("commands") or {}).get("final")
                         or {})
            vx_r = tuple(final.get("lin_vel_x", vx_r))
            vy_r = tuple(final.get("lin_vel_y", vy_r))
            yaw_r = tuple(final.get("ang_vel_yaw", yaw_r))
        scope = ("  (FULL envelope - extrapolation)" if args.full_envelope
                 else "  (trained ranges)")
        print("sampling commands from vx%s vy%s yaw%s%s"
              % (vx_r, vy_r, yaw_r, scope))
        print()
        results = []
        for ep in range(args.episodes):
            cmd = (
                float(rng.uniform(*vx_r)),
                float(rng.uniform(*vy_r)),
                float(rng.uniform(*yaw_r)),
            )
            r = rollout(env, predict, cmd, args.steps, seed=args.seed + ep)
            r["cmd"] = cmd
            results.append(r)
            print("ep %2d  cmd=(%+.2f,%+.2f,%+.2f)  err_lin=%.3f  err_ang=%.3f  %s"
                  % (ep + 1, cmd[0], cmd[1], cmd[2],
                     r.get("err_vx", float("nan")) + r.get("err_vy", float("nan")),
                     r.get("err_yaw", float("nan")),
                     "FELL" if r["fell"] else ""))

        ok = [r for r in results if r["steps"] > 0]
        print("\n" + "=" * 64)
        print("EVALUATION SUMMARY  (%d commands, %d control steps each)"
              % (len(results), args.steps))
        print("=" * 64)
        print("%-32s %10.3f m/s" % ("mean |vx| error",
                                    np.mean([r["err_vx"] for r in ok])))
        print("%-32s %10.3f m/s" % ("mean |vy| error",
                                    np.mean([r["err_vy"] for r in ok])))
        print("%-32s %10.3f rad/s" % ("mean |yaw rate| error",
                                      np.mean([r["err_yaw"] for r in ok])))
        print("%-32s %10.1f %%" % ("survival rate",
                                   100.0 * sum(not r["fell"] for r in results)
                                   / max(len(results), 1)))
        print("%-32s %10.3f m" % ("mean body height",
                                  np.mean([r["height"] for r in ok])))
        print("%-32s %10.2f / 4" % ("mean feet in contact",
                                    4 * np.mean([r["duty"] for r in ok])))
        print("%-32s %10.4f" % ("action jerk (mean |da|)",
                                np.mean([r["action_jerk"] for r in ok])))
        print("=" * 64)
    finally:
        env.close()


if __name__ == "__main__":
    main()
