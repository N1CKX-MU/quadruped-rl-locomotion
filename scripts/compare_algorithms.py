"""Train and compare PPO, SAC, and TD3 on the Go2 environment."""

import argparse
import os
import sys

import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.go2_env import Go2Env


def make_env(env_cfg, seed=0):
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
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


ALGORITHMS = {
    "PPO": lambda env, tb_log: PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=tb_log,
        verbose=1,
        device="auto",
    ),
    "SAC": lambda env, tb_log: SAC(
        "MlpPolicy", env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=tb_log,
        verbose=1,
        device="auto",
    ),
    "TD3": lambda env, tb_log: TD3(
        "MlpPolicy", env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_delay=2,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=tb_log,
        verbose=1,
        device="auto",
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Compare RL algorithms on Go2")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--algorithms", nargs="+", default=["PPO", "SAC", "TD3"],
        choices=list(ALGORITHMS.keys()),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    env_cfg = config["environment"]
    log_cfg = config["logging"]

    tb_log = log_cfg["tensorboard_log"]
    os.makedirs(tb_log, exist_ok=True)

    for algo_name in args.algorithms:
        print(f"\n{'='*60}")
        print(f"Training {algo_name} for {args.timesteps} timesteps")
        print(f"{'='*60}\n")

        env = DummyVecEnv([make_env(env_cfg, seed=42)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        eval_env = DummyVecEnv([make_env(env_cfg, seed=100)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        model = ALGORITHMS[algo_name](env, tb_log)

        save_dir = os.path.join(log_cfg["model_dir"], algo_name.lower())
        os.makedirs(save_dir, exist_ok=True)

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(save_dir, "best"),
            log_path=os.path.join(log_cfg["log_dir"], "eval", algo_name.lower()),
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
        )

        model.learn(
            total_timesteps=args.timesteps,
            callback=eval_cb,
            tb_log_name=f"go2_{algo_name.lower()}",
            progress_bar=True,
        )

        model.save(os.path.join(save_dir, f"go2_{algo_name.lower()}_final"))
        env.save(os.path.join(save_dir, "vecnormalize.pkl"))
        print(f"{algo_name} training complete. Model saved to {save_dir}/")

        env.close()
        eval_env.close()

    print("\nAll algorithms trained. Compare with:")
    print(f"  tensorboard --logdir {tb_log}")


if __name__ == "__main__":
    main()
