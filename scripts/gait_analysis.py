"""Measure the gait a trained policy actually produces, and compare it to the
gait it was commanded.

    python scripts/gait_analysis.py                       # trot, 1.0 m/s
    python scripts/gait_analysis.py --gait pace --cmd 0.8 0 0
    python scripts/gait_analysis.py --all-gaits           # one figure per gait

In v1 the gait was whatever fell out of the optimiser, so "gait analysis" meant
describing an emergent pattern after the fact. In v2 the gait is *commanded*
through a phase schedule, which turns this script into a proper measurement:
the reference schedule and the achieved contacts can be plotted on the same
axes, and the error between them is a number.

What is measured
    duty factor      fraction of the stride each foot spends loaded
    stride frequency touchdowns per second, per foot
    phase offset     each foot's touchdown time relative to FL, in cycles
    schedule match   fraction of control steps where actual contact equals the
                     commanded schedule; this is exactly the gait_phase reward

Reference offsets: trot (0, .5, .5, 0), pace (0, .5, 0, .5),
bound (0, 0, .5, .5), walk (0, .5, .25, .75).
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:  # pragma: no cover - plotting is optional
    HAS_MPL = False

from envs.gait import FOOT_ORDER, GAIT_NAMES, desired_contact, gait_params  # noqa: E402
from envs.go2_env import Go2Env  # noqa: E402

FOOT_LABELS = ["FL (front left)", "FR (front right)",
               "RL (rear left)", "RR (rear right)"]
FOOT_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]


# --------------------------------------------------------------------------- #
#  Measurement                                                                #
# --------------------------------------------------------------------------- #


def duty_factors(contacts):
    """Fraction of the recorded window each foot spent loaded."""
    return contacts.mean(axis=0)


def touchdown_indices(foot_contacts):
    """Indices where a foot transitions from swing to stance."""
    return np.where(np.diff(foot_contacts.astype(int)) > 0)[0] + 1


def stride_frequencies(contacts, dt):
    """Touchdowns per second, per foot. Zero if fewer than two touchdowns."""
    out = np.zeros(contacts.shape[1])
    for i in range(contacts.shape[1]):
        td = touchdown_indices(contacts[:, i])
        if len(td) >= 2:
            out[i] = 1.0 / (np.mean(np.diff(td)) * dt)
    return out


def phase_offsets(contacts, dt):
    """Each foot's touchdown phase relative to FL, in cycles [0, 1).

    Computed from the *median* offset over all FL cycles rather than the first
    one, so a single missed step does not move the answer.
    """
    fl = touchdown_indices(contacts[:, 0])
    if len(fl) < 2:
        return np.full(contacts.shape[1], np.nan)
    period = float(np.mean(np.diff(fl)))
    offsets = np.zeros(contacts.shape[1])
    for i in range(contacts.shape[1]):
        td = touchdown_indices(contacts[:, i])
        if len(td) == 0:
            offsets[i] = np.nan
            continue
        rel = []
        for t in fl[:-1]:
            following = td[td >= t]
            if len(following):
                rel.append(((following[0] - t) % period) / period)
        offsets[i] = float(np.median(rel)) if rel else np.nan
    return offsets


def schedule_match(contacts, phases, gait):
    """Fraction of steps where the actual contact matched the commanded one.

    This is numerically the same quantity as the ``gait_phase`` reward term, so
    a policy scoring 0.95 here is collecting 95% of that term's value.
    """
    offsets, duty = gait_params(gait)
    desired = np.array([desired_contact(p, offsets, duty) for p in phases])
    return float(np.mean(contacts == desired)), desired


# --------------------------------------------------------------------------- #
#  Rollout                                                                    #
# --------------------------------------------------------------------------- #


def load_policy(model_path, vecnormalize_path, device="cpu"):
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device=device)
    obs_rms, clip_obs, eps = None, 10.0, 1e-8
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        with open(vecnormalize_path, "rb") as f:
            vec = pickle.load(f)
        obs_rms, clip_obs, eps = vec.obs_rms, vec.clip_obs, vec.epsilon

    def predict(obs):
        x = obs
        if obs_rms is not None:
            x = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + eps),
                        -clip_obs, clip_obs).astype(np.float32)
        a, _ = model.predict(x, deterministic=True)
        return a

    return predict


def record(env, predict, command, gait, frequency, steps, settle=100, seed=0):
    obs, _ = env.reset(seed=seed)
    env.set_command(lin_vel_x=command[0], lin_vel_y=command[1],
                    ang_vel_yaw=command[2], gait=gait, gait_frequency=frequency)

    contacts, phases, vels = [], [], []
    for i in range(steps + settle):
        action = predict(obs)
        obs, _, terminated, _, info = env.step(action)
        if i >= settle:
            contacts.append(info["contacts"])
            phases.append(info["gait_phase"])
            vels.append(info["lin_vel_b"][:2])
        if terminated:
            break
    return np.array(contacts), np.array(phases), np.array(vels)


# --------------------------------------------------------------------------- #
#  Reporting                                                                  #
# --------------------------------------------------------------------------- #


def report(contacts, phases, vels, gait, dt, command):
    if len(contacts) < 10:
        print("The robot fell before enough data was collected.")
        return None

    duty = duty_factors(contacts)
    freq = stride_frequencies(contacts, dt)
    offsets = phase_offsets(contacts, dt)
    match, desired = schedule_match(contacts, phases, gait)
    ref_offsets, ref_duty = gait_params(gait)

    print("\ncommanded gait      : %s (reference offsets %s, duty %.2f)"
          % (gait, tuple(ref_offsets), ref_duty))
    print("commanded velocity  : vx=%.2f vy=%.2f yaw=%.2f"
          % (command[0], command[1], command[2]))
    print("achieved velocity   : vx=%.3f vy=%.3f"
          % (vels[:, 0].mean(), vels[:, 1].mean()))
    print("schedule match      : %.1f%%  (the gait_phase reward, as a percentage)"
          % (100 * match))
    print()
    print("%-20s %10s %10s %12s %12s" % ("foot", "duty", "stride Hz",
                                         "phase (meas)", "phase (ref)"))
    print("-" * 68)
    for i, label in enumerate(FOOT_LABELS):
        print("%-20s %10.3f %10.2f %12s %12.2f"
              % (label, duty[i], freq[i],
                 "nan" if np.isnan(offsets[i]) else "%.2f" % offsets[i],
                 ref_offsets[i]))
    print("-" * 68)
    print("mean duty factor    : %.3f   (reference %.2f)" % (duty.mean(), ref_duty))
    print("mean stride freq    : %.2f Hz" % freq.mean())

    # The interpretation the numbers are actually for.
    notes = []
    if duty.mean() > ref_duty + 0.15:
        notes.append("Feet stay down longer than commanded: the policy is "
                     "shuffling rather than swinging. Raise feet_air_time.")
    if match < 0.7:
        notes.append("Poor schedule match: either gait_phase's weight is too "
                     "low, or the commanded frequency is outside what the "
                     "policy was trained on.")
    if np.nanmax(np.abs(offsets - ref_offsets)) > 0.2:
        notes.append("Measured phase offsets do not match the reference: the "
                     "policy is running a different gait from the one asked for.")
    if notes:
        print("\nnotes:")
        for n in notes:
            print("  - " + n)

    return dict(duty=duty, freq=freq, offsets=offsets, match=match,
                desired=desired)


def plot(contacts, desired, dt, gait, path):
    if not HAS_MPL:
        print("matplotlib is not installed; skipping the figure.")
        return
    n = len(contacts)
    t = np.arange(n) * dt
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for ax, data, title in (
        (axes[0], desired, "commanded schedule"),
        (axes[1], contacts, "measured contacts"),
    ):
        for i in range(4):
            on = data[:, i] > 0.5
            ax.fill_between(t, i - 0.4, i + 0.4, where=on,
                            color=FOOT_COLORS[i], step="mid", linewidth=0)
        ax.set_yticks(range(4))
        ax.set_yticklabels(FOOT_ORDER)
        ax.set_ylim(-0.6, 3.6)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", alpha=0.25)

    axes[1].set_xlabel("time (s)")
    fig.suptitle("Gait diagram - %s" % gait, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote " + path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="models/go2_ppo_final.zip")
    p.add_argument("--vec-normalize", default=None)
    p.add_argument("--xml", default="mujoco_menagerie/unitree_go2/scene.xml")
    p.add_argument("--gait", default="trot", choices=list(GAIT_NAMES))
    p.add_argument("--all-gaits", action="store_true")
    p.add_argument("--cmd", type=float, nargs=3, default=[1.0, 0.0, 0.0])
    p.add_argument("--frequency", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="assets")
    args = p.parse_args()

    vecnorm = args.vec_normalize
    if vecnorm is None:
        for guess in ("models/go2_ppo_vecnormalize.pkl",
                      "models/vecnormalize_final.pkl"):
            if os.path.exists(guess):
                vecnorm = guess
                break

    predict = load_policy(args.model, vecnorm)
    env = Go2Env(xml_path=args.xml, randomize_dynamics=False, push_enabled=False,
                 command_resample_interval_s=0.0,
                 max_episode_steps=args.steps + 200)
    os.makedirs(args.out_dir, exist_ok=True)

    gaits = [g for g in GAIT_NAMES if g != "stand"] if args.all_gaits else [args.gait]
    try:
        for gait in gaits:
            contacts, phases, vels = record(
                env, predict, args.cmd, gait, args.frequency, args.steps,
                seed=args.seed
            )
            result = report(contacts, phases, vels, gait, env.dt, args.cmd)
            if result is not None:
                plot(contacts, result["desired"], env.dt, gait,
                     os.path.join(args.out_dir, "gait_%s.png" % gait))
    finally:
        env.close()


if __name__ == "__main__":
    main()
