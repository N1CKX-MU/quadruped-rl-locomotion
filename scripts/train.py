"""Training entry point for Go2 quadruped locomotion."""

import argparse
import os
import sys
import yaml

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.go2_env import Go2Env


def make_env(config):
    """Create a Go2 environment from config."""
    env_cfg = config["environment"]
    def _init():
        return Go2Env(
            xml_path=env_cfg["xml_path"],
            frame_skip=env_cfg["frame_skip"],
            forward_reward_weight=env_cfg["forward_reward_weight"],
            ctrl_cost_weight=env_cfg["ctrl_cost_weight"],
            healthy_reward=env_cfg["healthy_reward"],
            healthy_z_range=tuple(env_cfg["healthy_z_range"]),
            reset_noise_scale=env_cfg["reset_noise_scale"],
            max_episode_steps=env_cfg["max_episode_steps"],
        )
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Go2 quadruped with PPO")
    parser.add_argument(
        "--config", type=str, default="configs/training_config.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    log_cfg = config["logging"]

    # Create directories
    os.makedirs(log_cfg["model_dir"], exist_ok=True)
    os.makedirs(log_cfg["tensorboard_log"], exist_ok=True)

    # Create vectorized environment with observation normalization
    env = DummyVecEnv([make_env(config)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval environment
    eval_env = DummyVecEnv([make_env(config)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=log_cfg["save_freq"],
        save_path=log_cfg["model_dir"],
        name_prefix="go2_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_cfg["model_dir"], "best"),
        log_path=log_cfg["log_dir"],
        eval_freq=log_cfg["eval_freq"],
        n_eval_episodes=log_cfg["n_eval_episodes"],
        deterministic=True,
    )

    # Create or load model
    if args.resume:
        print(f"Resuming training from {args.resume}")
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
            policy_kwargs=train_cfg["policy_kwargs"],
            tensorboard_log=log_cfg["tensorboard_log"],
            verbose=1,
        )

    print(f"Training for {train_cfg['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # Save final model and normalization stats
    model.save(os.path.join(log_cfg["model_dir"], "go2_ppo_final"))
    env.save(os.path.join(log_cfg["model_dir"], "vec_normalize.pkl"))
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
