"""Analyze the emergent gait pattern of a trained policy.

Records foot contact patterns over one episode and generates a gait
diagram showing which feet are on the ground at each timestep. Also
computes gait metrics: stride frequency, duty factor, and phase offsets.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.go2_env import Go2Env


FOOT_NAMES = ["FL (Front Left)", "FR (Front Right)", "RL (Rear Left)", "RR (Rear Right)"]
FOOT_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]


def compute_gait_metrics(contacts, dt):
    """Compute gait metrics from contact data."""
    n_steps, n_feet = contacts.shape
    total_time = n_steps * dt

    metrics = {}
    for i, name in enumerate(FOOT_NAMES):
        foot = contacts[:, i]
        duty_factor = np.mean(foot)

        # Find stride periods (time between consecutive touch-downs)
        touch_downs = np.where(np.diff(foot) > 0)[0]
        if len(touch_downs) >= 2:
            stride_periods = np.diff(touch_downs) * dt
            stride_freq = 1.0 / np.mean(stride_periods)
        else:
            stride_freq = 0.0

        metrics[name] = {
            "duty_factor": duty_factor,
            "stride_freq": stride_freq,
        }

    # Phase offsets relative to FL
    fl_touchdowns = np.where(np.diff(contacts[:, 0]) > 0)[0]
    if len(fl_touchdowns) >= 2:
        fl_period = np.mean(np.diff(fl_touchdowns))
        for i in range(1, n_feet):
            other_touchdowns = np.where(np.diff(contacts[:, i]) > 0)[0]
            if len(other_touchdowns) > 0 and fl_period > 0:
                # Find closest touchdown to first FL touchdown
                diffs = other_touchdowns - fl_touchdowns[0]
                closest = diffs[np.argmin(np.abs(diffs))]
                phase = (closest / fl_period) % 1.0
                metrics[FOOT_NAMES[i]]["phase_offset"] = phase
            else:
                metrics[FOOT_NAMES[i]]["phase_offset"] = 0.0
        metrics[FOOT_NAMES[0]]["phase_offset"] = 0.0
    else:
        for name in FOOT_NAMES:
            metrics[name]["phase_offset"] = 0.0

    return metrics


def plot_gait_diagram(contacts, dt, metrics, output_path):
    """Plot a gait diagram showing foot contact patterns."""
    n_steps = contacts.shape[0]
    time = np.arange(n_steps) * dt

    fig, axes = plt.subplots(4, 1, figsize=(14, 5), sharex=True)
    fig.suptitle("Gait Diagram — Foot Contact Patterns", fontsize=14, fontweight="bold")

    for i, (ax, name, color) in enumerate(zip(axes, FOOT_NAMES, FOOT_COLORS)):
        foot = contacts[:, i]

        # Draw contact bars
        in_contact = False
        start = 0
        for t in range(n_steps):
            if foot[t] > 0.5 and not in_contact:
                start = t
                in_contact = True
            elif foot[t] < 0.5 and in_contact:
                ax.axvspan(time[start], time[t], alpha=0.7, color=color)
                in_contact = False
        if in_contact:
            ax.axvspan(time[start], time[-1], alpha=0.7, color=color)

        m = metrics[name]
        label = f"{name}  |  duty={m['duty_factor']:.0%}  freq={m['stride_freq']:.1f}Hz  phase={m['phase_offset']:.2f}"
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.text(0.01, 0.5, label, transform=ax.transAxes, fontsize=9,
                verticalalignment="center", fontfamily="monospace")
        ax.set_xlim(time[0], time[-1])

    axes[-1].set_xlabel("Time (s)")

    # Identify gait type
    phases = [metrics[name]["phase_offset"] for name in FOOT_NAMES]
    fr_phase = phases[1]
    rl_phase = phases[2]
    rr_phase = phases[3]

    gait_type = "Unknown"
    if abs(fr_phase - 0.5) < 0.15 and abs(rr_phase - 0.5) < 0.15:
        gait_type = "Trot (diagonal pairs alternate)"
    elif abs(fr_phase - 0.5) < 0.15 and abs(rl_phase - 0.5) < 0.15:
        gait_type = "Pace (lateral pairs alternate)"
    elif abs(rr_phase - 0.25) < 0.15:
        gait_type = "Walk (sequential)"
    elif all(p < 0.1 or p > 0.9 for p in phases):
        gait_type = "Bound (front/rear pairs)"

    fig.text(0.5, 0.01, f"Detected gait: {gait_type}", ha="center", fontsize=11,
             fontstyle="italic", color="#333")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Gait diagram saved to {output_path}")
    return gait_type


def main():
    parser = argparse.ArgumentParser(description="Analyze gait pattern")
    parser.add_argument("--model", type=str, default="models/go2_ppo_final.zip")
    parser.add_argument("--vec-normalize", type=str, default="models/vecnormalize_final.pkl")
    parser.add_argument("--output", type=str, default="assets/gait_analysis.png")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--cmd-vel", type=float, nargs=3, default=[0.8, 0.0, 0.0])
    args = parser.parse_args()

    if not HAS_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    raw_env = Go2Env(cmd_vel=tuple(args.cmd_vel))
    env = DummyVecEnv([lambda: raw_env])
    if os.path.exists(args.vec_normalize):
        env = VecNormalize.load(args.vec_normalize, env)
        env.training = False
        env.norm_reward = False

    algo_map = {"ppo": PPO, "sac": SAC, "td3": TD3}
    algo_cls = PPO
    for name, cls in algo_map.items():
        if name in args.model.lower():
            algo_cls = cls
            break
    model = algo_cls.load(args.model)

    # Collect contact data
    all_contacts = []
    obs = env.reset()
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        contacts = raw_env._get_foot_contacts()
        all_contacts.append(contacts)
        if done[0]:
            obs = env.reset()

    contacts = np.array(all_contacts)
    dt = raw_env.dt

    # Compute and print metrics
    metrics = compute_gait_metrics(contacts, dt)
    print("\nGait Metrics:")
    print(f"{'Foot':<25} {'Duty Factor':>12} {'Stride Freq':>12} {'Phase':>8}")
    print("-" * 60)
    for name in FOOT_NAMES:
        m = metrics[name]
        print(f"{name:<25} {m['duty_factor']:>11.0%} {m['stride_freq']:>10.1f} Hz {m['phase_offset']:>7.2f}")

    # Plot
    gait_type = plot_gait_diagram(contacts, dt, metrics, args.output)
    print(f"\nDetected gait: {gait_type}")

    env.close()


if __name__ == "__main__":
    main()
