"""Training entry point for Go2 quadruped locomotion with PPO."""

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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.go2_env import Go2Env
from callbacks.curriculum import CurriculumCallback


def make_env(env_cfg, rank, seed=0):
    """Factory function for creating a monitored Go2 environment."""
    def _init():
        env = Go2Env(
            xml_path=env_cfg["xml_path"],
            frame_skip=env_cfg["frame_skip"],
            cmd_vel=tuple(env_cfg["cmd_vel"]),
            action_scale=env_cfg["action_scale"],
            healthy_z_range=tuple(env_cfg["healthy_z_range"]),
            max_pitch_roll=env_cfg["max_pitch_roll"],
            reset_noise_scale=env_cfg["reset_noise_scale"],
            max_episode_steps=env_cfg["max_episode_steps"],
            randomize_dynamics=env_cfg.get("randomize_dynamics", False),
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Go2 quadruped with PPO")
    parser.add_argument(
        "--config", type=str, default="configs/training_config.yaml",
    )
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_cfg = config["environment"]
    train_cfg = config["training"]
    log_cfg = config["logging"]
    curriculum_cfg = config.get("curriculum", {})

    os.makedirs(log_cfg["model_dir"], exist_ok=True)
    os.makedirs(log_cfg["tensorboard_log"], exist_ok=True)
    os.makedirs(os.path.join(log_cfg["model_dir"], "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_cfg["model_dir"], "best"), exist_ok=True)

    n_envs = train_cfg.get("n_envs", 8)

    # Parallel training environments
    env = SubprocVecEnv([make_env(env_cfg, i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Single eval environment
    eval_env = DummyVecEnv([make_env(env_cfg, 100)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(log_cfg["save_freq"] // n_envs, 1),
        save_path=os.path.join(log_cfg["model_dir"], "checkpoints"),
        name_prefix="go2_ppo",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_cfg["model_dir"], "best"),
        log_path=os.path.join(log_cfg["log_dir"], "eval"),
        eval_freq=max(log_cfg["eval_freq"] // n_envs, 1),
        n_eval_episodes=log_cfg["n_eval_episodes"],
        deterministic=True,
    )

    callbacks = [checkpoint_cb, eval_cb]

    if curriculum_cfg.get("enabled", False):
        curriculum_cb = CurriculumCallback(
            max_vel=curriculum_cfg.get("max_vel", 1.2),
            start_vel=curriculum_cfg.get("start_vel", 0.3),
            warmup_steps=curriculum_cfg.get("warmup_steps", 500_000),
        )
        callbacks.append(curriculum_cb)

    # Build PPO model
    policy_kwargs = train_cfg.get("policy_kwargs", {})
    if "net_arch" in policy_kwargs and isinstance(policy_kwargs["net_arch"], list):
        arch = policy_kwargs["net_arch"]
        policy_kwargs["net_arch"] = dict(pi=arch, vf=arch)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            policy=train_cfg["policy"],
            env=env,
            learning_rate=train_cfg["learning_rate"],
            n_steps=train_cfg["n_steps"],
            batch_size=train_cfg["batch_size"],
            n_epochs=train_cfg["n_epochs"],
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"],
            clip_range=train_cfg["clip_range"],
            ent_coef=train_cfg["ent_coef"],
            vf_coef=train_cfg["vf_coef"],
            max_grad_norm=train_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_cfg["tensorboard_log"],
            verbose=1,
            device="auto",
        )

    print(f"Training PPO for {train_cfg['total_timesteps']} timesteps with {n_envs} envs...")
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=CallbackList(callbacks),
        tb_log_name="go2_ppo",
        progress_bar=True,
    )

    model.save(os.path.join(log_cfg["model_dir"], "go2_ppo_final"))
    env.save(os.path.join(log_cfg["model_dir"], "vecnormalize_final.pkl"))
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
