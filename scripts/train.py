"""Training entry point for Go2 quadruped locomotion with PPO (v2).

Fixes over v1, in the order they bite you:

B12 Resuming a run now restores the VecNormalize statistics. v1 called
    ``PPO.load(path, env=env)`` with a *freshly constructed* VecNormalize, so
    the running observation mean and variance reset to 0/1. The resumed policy
    then received inputs on a completely different scale from the ones it was
    trained on, and the run silently collapsed. This is the nastiest bug in the
    original repo because nothing errors - the loss just gets worse.
B13 The evaluation env no longer updates its own normalisation statistics.
B14 The PPO hyperparameters match the rollout size. v1 paired a 16384-transition
    rollout with ``batch_size=64`` and ``n_epochs=10``: 2560 gradient steps per
    rollout, which is far outside PPO's trust region regardless of the clip
    range, and about an order of magnitude more compute per sample than needed.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/training_config.yaml --seed 1
    python scripts/train.py --resume models/checkpoints/go2_ppo_5000000_steps.zip
"""

import argparse
import os
import sys

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callbacks.curriculum import CommandCurriculumCallback  # noqa: E402
from callbacks.logging import (  # noqa: E402
    EpisodeStatsCallback,
    RewardTermLoggerCallback,
)
from envs.commands import ranges_from_config  # noqa: E402
from envs.go2_env import Go2Env  # noqa: E402


def build_env_kwargs(config):
    """Translate the YAML into Go2Env constructor arguments."""
    env_cfg = dict(config["environment"])
    cmd_cfg = config.get("commands", {}) or {}
    reward_cfg = (config.get("reward", {}) or {}).get("weights")

    # Start from the *initial* (narrow) ranges. The curriculum callback widens
    # them from there; without a curriculum the env would sample the full
    # envelope from step one, which is a reliable way to never learn anything.
    initial = ranges_from_config(cmd_cfg.get("initial"))

    env_cfg.update(
        command_ranges=initial,
        stand_probability=cmd_cfg.get("stand_probability", 0.10),
        command_resample_interval_s=cmd_cfg.get("resample_interval_s", 5.0),
        gaits=tuple(cmd_cfg.get("gaits", ["trot"])),
        reward_weights=reward_cfg,
    )
    return env_cfg


def make_env(env_kwargs, rank, seed=0):
    """Factory for a monitored Go2 environment.

    ``Monitor`` is given the reward-term keys so that episode summaries carry
    them too, which is what the evaluation scripts read back.
    """

    def _init():
        env = Go2Env(**env_kwargs)
        env = Monitor(env, info_keywords=("tracking_lin_err", "tracking_ang_err"))
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Go2 quadruped with PPO")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved .zip. Its sibling VecNormalize "
                             "pkl is loaded automatically (fixes B12).")
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="Explicit VecNormalize .pkl to restore with --resume.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--vec-env", choices=("auto", "subproc", "dummy"),
                        default="auto",
                        help="Vectorisation backend. On Windows the SubprocVecEnv "
                             "pipe round trip can cost more than the physics "
                             "step itself; 'dummy' runs every env in-process, "
                             "which is often faster here. Benchmark both.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="go2_ppo")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    log_cfg = config["logging"]
    cmd_cfg = config.get("commands", {}) or {}
    curriculum_cfg = config.get("curriculum", {}) or {}

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 0)
    n_envs = args.n_envs or train_cfg.get("n_envs", 16)
    total_timesteps = args.timesteps or train_cfg["total_timesteps"]

    model_dir = log_cfg["model_dir"]
    for sub in ("", "checkpoints", "best"):
        os.makedirs(os.path.join(model_dir, sub), exist_ok=True)
    os.makedirs(log_cfg["tensorboard_log"], exist_ok=True)

    env_kwargs = build_env_kwargs(config)
    # The eval env is deterministic on purpose: no pushes, no randomised
    # dynamics. Otherwise the eval curve measures luck as much as skill.
    eval_kwargs = dict(env_kwargs)
    eval_kwargs.update(randomize_dynamics=False, push_enabled=False)

    if args.vec_env == "subproc":
        vec_cls = SubprocVecEnv
    elif args.vec_env == "dummy":
        vec_cls = DummyVecEnv
    else:
        vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = vec_cls([make_env(env_kwargs, i, seed) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(eval_kwargs, 0, seed + 10_000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # B13: freeze the eval env's statistics. EvalCallback copies the training
    # env's running mean/var across before each evaluation; leaving training=True
    # lets the eval env drift its own statistics in between, so the eval curve
    # measures a slightly different normalisation every time.
    eval_env.training = False

    checkpoint_cb = CheckpointCallback(
        save_freq=max(log_cfg["save_freq"] // n_envs, 1),
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix=args.run_name,
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=os.path.join(log_cfg["log_dir"], "eval"),
        eval_freq=max(log_cfg["eval_freq"] // n_envs, 1),
        n_eval_episodes=log_cfg["n_eval_episodes"],
        deterministic=True,
    )
    callbacks = [checkpoint_cb, eval_cb, RewardTermLoggerCallback(),
                 EpisodeStatsCallback()]

    if curriculum_cfg.get("enabled", True):
        callbacks.append(
            CommandCurriculumCallback(
                initial_ranges=ranges_from_config(cmd_cfg.get("initial")),
                final_ranges=ranges_from_config(cmd_cfg.get("final")),
                threshold=curriculum_cfg.get("threshold", 0.75),
                decay_threshold=curriculum_cfg.get("decay_threshold", 0.50),
                step=curriculum_cfg.get("step", 0.05),
            )
        )

    policy_kwargs = dict(train_cfg.get("policy_kwargs", {}) or {})
    arch = policy_kwargs.get("net_arch")
    if isinstance(arch, list):
        # Separate trunks for actor and critic. Sharing them is a false economy
        # here: the critic wants to model the return of a stochastic, pushed
        # robot, the actor wants a smooth control law, and the two objectives
        # fight over shared features.
        policy_kwargs["net_arch"] = dict(pi=list(arch), vf=list(arch))

    lr = train_cfg["learning_rate"]
    if train_cfg.get("lr_schedule", "constant") == "linear":
        # Anneal to zero. PPO's clip range assumes the policy is not moving far
        # per update; a constant LR late in training keeps pushing after the
        # advantages have stopped being informative.
        lr = get_linear_fn(start=float(lr), end=0.0, end_fraction=1.0)

    if args.resume:
        print("Resuming from " + args.resume)
        model = PPO.load(args.resume, env=env, device=args.device)
        # B12: restore the observation/reward normalisation that goes with this
        # checkpoint. Without it the resumed policy sees inputs on a different
        # scale than it was trained on and quietly falls apart.
        stats_path = args.vecnormalize or _guess_vecnormalize_path(args.resume)
        if stats_path and os.path.exists(stats_path):
            print("Restoring VecNormalize statistics from " + stats_path)
            env = VecNormalize.load(stats_path, env.venv)
            env.training = True
            env.norm_reward = True
            model.set_env(env)
        else:
            raise SystemExit(
                "Could not find the VecNormalize statistics for this checkpoint.\n"
                "Pass --vecnormalize explicitly. Resuming without them silently\n"
                "destroys the policy (see docs/14-debugging-log.md, B12)."
            )
    else:
        model = PPO(
            policy=train_cfg["policy"],
            env=env,
            learning_rate=lr,
            n_steps=train_cfg["n_steps"],
            batch_size=train_cfg["batch_size"],
            n_epochs=train_cfg["n_epochs"],
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"],
            clip_range=train_cfg["clip_range"],
            ent_coef=train_cfg["ent_coef"],
            vf_coef=train_cfg["vf_coef"],
            max_grad_norm=train_cfg["max_grad_norm"],
            target_kl=train_cfg.get("target_kl"),
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_cfg["tensorboard_log"],
            seed=seed,
            verbose=1,
            device=args.device,
        )

    print(
        "Training PPO for %s timesteps with %d envs "
        "(rollout %d transitions, %d minibatches x %d epochs)"
        % (
            f"{total_timesteps:,}",
            n_envs,
            n_envs * train_cfg["n_steps"],
            max(n_envs * train_cfg["n_steps"] // train_cfg["batch_size"], 1),
            train_cfg["n_epochs"],
        )
    )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name=args.run_name,
            progress_bar=True,
            reset_num_timesteps=not args.resume,
        )
    finally:
        # Save on Ctrl-C too. Losing a twelve-hour run to an interrupt is a
        # self-inflicted wound.
        model.save(os.path.join(model_dir, args.run_name + "_final"))
        env.save(os.path.join(model_dir, args.run_name + "_vecnormalize.pkl"))
        print("Saved model and normalisation statistics to " + model_dir)


def _guess_vecnormalize_path(model_path):
    """CheckpointCallback writes ``<prefix>_<steps>_steps.zip`` alongside
    ``<prefix>_vecnormalize_<steps>_steps.pkl``. Reconstruct the latter."""
    directory, filename = os.path.split(model_path)
    stem = os.path.splitext(filename)[0]
    if "_" in stem:
        head, _, tail = stem.partition("_")
        candidate = os.path.join(directory, head + "_vecnormalize_" + tail + ".pkl")
        if os.path.exists(candidate):
            return candidate
    for name in os.listdir(directory or "."):
        if name.startswith("vecnormalize") or "vecnormalize" in name:
            if name.endswith(".pkl"):
                return os.path.join(directory, name)
    return None


if __name__ == "__main__":
    main()
