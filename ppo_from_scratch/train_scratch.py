"""Train Go2Env with the from-scratch PPO in ppo.py.

    python ppo_from_scratch/train_scratch.py --timesteps 2000000

The point of this script is comparability. It trains the *same* environment,
with the *same* hyperparameters, as scripts/train.py, and writes TensorBoard
scalars under the same names. Put both runs on one TensorBoard and the curves
should sit on top of each other within seed noise. If they do, you have
verified that you understand PPO well enough to rebuild it; if they diverge,
the difference is a specific, findable implementation detail - and hunting it
down teaches more than reading any paper.

Note what is borrowed and what is not. The vectorised environment wrapper and
the Monitor come from Stable-Baselines3, because process management is not the
algorithm. Everything from the rollout buffer inwards - GAE, the surrogate
objective, the clipping, the KL early stop, the optimiser loop - is in ppo.py.
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
)
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from envs.commands import ranges_from_config  # noqa: E402
from envs.go2_env import Go2Env  # noqa: E402
from ppo_from_scratch.ppo import PPO, PPOConfig  # noqa: E402


def build_env_kwargs(config):
    env_cfg = dict(config["environment"])
    cmd_cfg = config.get("commands", {}) or {}
    env_cfg.update(
        command_ranges=ranges_from_config(cmd_cfg.get("initial")),
        stand_probability=cmd_cfg.get("stand_probability", 0.10),
        command_resample_interval_s=cmd_cfg.get("resample_interval_s", 5.0),
        gaits=tuple(cmd_cfg.get("gaits", ["trot"])),
        reward_weights=(config.get("reward") or {}).get("weights"),
    )
    return env_cfg


def make_env(env_kwargs, rank, seed):
    def _init():
        env = Go2Env(**env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/training_config.yaml")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-name", default="go2_scratch")
    p.add_argument("--vec-env", choices=("subproc", "dummy"), default="subproc")
    p.add_argument("--curriculum", action="store_true", default=True)
    p.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    cmd_cfg = config.get("commands", {}) or {}
    curr_cfg = config.get("curriculum", {}) or {}
    n_envs = args.n_envs or train_cfg.get("n_envs", 16)

    env_kwargs = build_env_kwargs(config)
    vec_cls = SubprocVecEnv if args.vec_env == "subproc" and n_envs > 1 else DummyVecEnv
    env = vec_cls([make_env(env_kwargs, i, args.seed) for i in range(n_envs)])

    # Read the SAME hyperparameters SB3 uses, so the comparison is fair.
    cfg = PPOConfig(
        n_steps=train_cfg["n_steps"],
        batch_size=train_cfg["batch_size"],
        n_epochs=train_cfg["n_epochs"],
        gamma=train_cfg["gamma"],
        gae_lambda=train_cfg["gae_lambda"],
        clip_range=train_cfg["clip_range"],
        ent_coef=train_cfg["ent_coef"],
        vf_coef=train_cfg["vf_coef"],
        max_grad_norm=train_cfg["max_grad_norm"],
        learning_rate=float(train_cfg["learning_rate"]),
        anneal_lr=train_cfg.get("lr_schedule", "constant") == "linear",
        target_kl=train_cfg.get("target_kl"),
        hidden=tuple(train_cfg["policy_kwargs"]["net_arch"]),
        log_std_init=train_cfg["policy_kwargs"].get("log_std_init", -1.0),
        device=args.device,
    )

    agent = PPO(env, cfg, seed=args.seed)
    print("device: %s   envs: %d   rollout: %d transitions"
          % (agent.device, n_envs, cfg.n_steps * n_envs))

    log_dir = os.path.join(config["logging"]["tensorboard_log"], args.run_name)
    writer = SummaryWriter(log_dir)
    model_dir = config["logging"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    # The adaptive command curriculum, reimplemented against this loop. It is
    # deliberately not the SB3 callback: this file is meant to be readable
    # without knowing SB3's callback protocol.
    curriculum = None
    if args.curriculum and curr_cfg.get("enabled", True):
        from envs.commands import CommandCurriculum

        curriculum = CommandCurriculum(
            initial=ranges_from_config(cmd_cfg.get("initial")),
            final=ranges_from_config(cmd_cfg.get("final")),
            threshold=curr_cfg.get("threshold", 0.85),
            decay_threshold=curr_cfg.get("decay_threshold", 0.65),
            step=curr_cfg.get("step", 0.05),
        )
        env.env_method("set_command_ranges", curriculum.ranges)

    start = time.perf_counter()
    last_save = [0]

    def callback(agent_, stats):
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl",
                    "clip_fraction", "explained_variance", "std", "lr"):
            if key in stats:
                writer.add_scalar("train/" + key, stats[key], stats["timesteps"])
        for key in ("ep_rew_mean", "ep_len_mean"):
            if key in stats:
                writer.add_scalar("rollout/" + key, stats[key], stats["timesteps"])
        elapsed = time.perf_counter() - start
        writer.add_scalar("time/fps", stats["timesteps"] / max(elapsed, 1e-9),
                          stats["timesteps"])

        if curriculum is not None:
            # The tracking score is averaged over the environments' most recent
            # info dicts. Cheaper and simpler than threading it through the
            # buffer, and the curriculum only needs a rollout-scale average.
            scores = [
                float(s) for s in env.env_method("_last_tracking_score")
                if s is not None
            ]
            if scores:
                score = float(np.mean(scores))
                level = curriculum.update(score)
                env.env_method("set_command_ranges", curriculum.ranges)
                writer.add_scalar("curriculum/level", level, stats["timesteps"])
                writer.add_scalar("curriculum/tracking_score", score,
                                  stats["timesteps"])

        if stats["timesteps"] - last_save[0] >= 500_000:
            last_save[0] = stats["timesteps"]
            agent_.save(os.path.join(model_dir, args.run_name + ".pt"))

    try:
        agent.learn(args.timesteps, log_every=1, callback=callback)
    finally:
        agent.save(os.path.join(model_dir, args.run_name + ".pt"))
        writer.close()
        env.close()
        print("saved " + os.path.join(model_dir, args.run_name + ".pt"))


if __name__ == "__main__":
    main()
