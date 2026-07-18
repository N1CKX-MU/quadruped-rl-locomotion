"""Plot training curves from TensorBoard logs for algorithm comparison."""

import argparse
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False


def load_tb_scalar(log_dir, tag="rollout/ep_rew_mean"):
    """Load a scalar from TensorBoard event files."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        print(f"  Warning: tag '{tag}' not found in {log_dir}")
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--logdir", type=str, default="logs/tensorboard/")
    parser.add_argument("--output", type=str, default="assets/training_curves.png")
    parser.add_argument("--tag", type=str, default="rollout/ep_rew_mean")
    args = parser.parse_args()

    if not HAS_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    if not HAS_TB:
        print("tensorboard not installed. Install with: pip install tensorboard")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Find all run directories
    run_dirs = {}
    for name in sorted(os.listdir(args.logdir)):
        full_path = os.path.join(args.logdir, name)
        if os.path.isdir(full_path):
            run_dirs[name] = full_path

    if not run_dirs:
        print(f"No runs found in {args.logdir}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for run_name, run_path in run_dirs.items():
        steps, values = load_tb_scalar(run_path, args.tag)
        if steps:
            ax.plot(steps, values, label=run_name, alpha=0.8)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(args.tag.split("/")[-1].replace("_", " ").title())
    ax.set_title("Training Curves: Algorithm Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
