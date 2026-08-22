"""Compare PPO, SAC and TD3 on the Go2 environment - fairly.

    python scripts/compare_algorithms.py --timesteps 3000000 --seeds 0 1 2

Read docs/07-why-not-sac-td3.md before trusting any output from this script.
That chapter re-reads the comparison this repository originally shipped
(`logs/algorithm_comparison.md`) and finds four problems with it. This version
fixes all four; the notes below say how, because the corrections are the
interesting part.

**1. TD3 now gets exploration noise.** The original constructed TD3 with no
`action_noise`. Stable-Baselines3 defaults that to `None`, and TD3's policy is
deterministic - unlike SAC, it has no other source of exploration. So TD3 was
run with essentially none, on a 12-dimensional continuous control problem. That
it converged to standing still is the expected outcome and says nothing about
the algorithm.

**2. Each algorithm runs at an environment count that suits it.** PPO is
on-policy and scales with parallel environments; SAC and TD3 are off-policy and
are throttled by gradient steps, not by samples. Running all three at one shared
environment count guarantees you handicap one of them. The original used a
single environment - PPO's worst case - and then concluded PPO was best.

**3. Wall-clock is reported alongside timesteps.** Equal timesteps is not equal
compute. The original gave PPO ~156k gradient steps and the off-policy methods
1,000,000 each, over two critics and a target network.

**4. Multiple seeds.** Single-seed RL results are close to meaningless;
between-seed variance on this task easily exceeds the between-algorithm
differences the original reported.

And the one that voids everything else: the original ran on the v1 environment,
where the nominal pose was outside the robot's joint limits and walking was
kinematically unreachable (bug B1). All three algorithms were being compared on
a task none of them could do.
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC, TD3  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from envs.commands import ranges_from_config  # noqa: E402
from envs.go2_env import Go2Env  # noqa: E402

# Environment counts chosen per algorithm rather than shared. See note 2.
DEFAULT_N_ENVS = {"PPO": 16, "SAC": 4, "TD3": 4}


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
        env = Monitor(env, info_keywords=("tracking_lin_err", "tracking_ang_err"))
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def build_model(name, env, tb_log, seed, action_dim):
    common = dict(policy="MlpPolicy", env=env, tensorboard_log=tb_log,
                  seed=seed, verbose=0, device="auto")
    if name == "PPO":
        return PPO(
            learning_rate=3e-4, n_steps=512, batch_size=2048, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.002,
            max_grad_norm=1.0, target_kl=0.02,
            policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128],
                                             vf=[512, 256, 128]),
                               log_std_init=-1.0),
            **common,
        )
    if name == "SAC":
        return SAC(
            learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
            tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
            learning_starts=10_000,
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            **common,
        )
    if name == "TD3":
        return TD3(
            learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
            tau=0.005, gamma=0.99, policy_delay=2, learning_starts=10_000,
            # Note 1: without this, TD3 has NO exploration at all.
            action_noise=NormalActionNoise(
                mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim)),
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            **common,
        )
    raise KeyError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/training_config.yaml")
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--algorithms", nargs="+", default=["PPO", "SAC", "TD3"],
                   choices=["PPO", "SAC", "TD3"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-envs", type=int, default=None,
                   help="Override the per-algorithm default. Doing so makes the "
                        "comparison unfair again; see note 2 in the docstring.")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    log_cfg = config["logging"]
    env_kwargs = build_env_kwargs(config)
    tb_log = log_cfg["tensorboard_log"]
    os.makedirs(tb_log, exist_ok=True)

    results = []
    for algo in args.algorithms:
        n_envs = args.n_envs or DEFAULT_N_ENVS[algo]
        for seed in args.seeds:
            tag = "%s_seed%d" % (algo.lower(), seed)
            print("\n" + "=" * 66)
            print("%s  seed %d  %d envs  %s timesteps"
                  % (algo, seed, n_envs, f"{args.timesteps:,}"))
            print("=" * 66)

            vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
            env = vec_cls([make_env(env_kwargs, i, seed) for i in range(n_envs)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

            eval_kwargs = dict(env_kwargs)
            eval_kwargs.update(randomize_dynamics=False, push_enabled=False)
            eval_env = DummyVecEnv([make_env(eval_kwargs, 0, seed + 10_000)])
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                                    clip_obs=10.0)
            eval_env.training = False   # bug B13

            save_dir = os.path.join(log_cfg["model_dir"], tag)
            os.makedirs(save_dir, exist_ok=True)
            eval_cb = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(save_dir, "best"),
                log_path=os.path.join(log_cfg["log_dir"], "eval", tag),
                eval_freq=max(50_000 // n_envs, 1),
                n_eval_episodes=10,
                deterministic=True,
            )

            model = build_model(algo, env, tb_log, seed,
                                int(np.prod(env.action_space.shape)))

            t0 = time.perf_counter()
            model.learn(total_timesteps=args.timesteps, callback=eval_cb,
                        tb_log_name="go2_" + tag, progress_bar=True)
            elapsed = time.perf_counter() - t0

            model.save(os.path.join(save_dir, "final"))
            env.save(os.path.join(save_dir, "vecnormalize.pkl"))

            best = float(eval_cb.best_mean_reward)
            results.append((algo, seed, best, elapsed,
                            args.timesteps / max(elapsed, 1e-9)))
            print("%s seed %d: best eval reward %.2f in %.0f s (%.0f steps/s)"
                  % (algo, seed, best, elapsed, args.timesteps / elapsed))

            env.close()
            eval_env.close()

    print("\n" + "=" * 66)
    print("SUMMARY  (note 3: wall-clock is reported, not just timesteps)")
    print("=" * 66)
    print("%-6s %6s %14s %10s %12s" % ("algo", "seed", "best eval rew",
                                       "wall (s)", "steps/s"))
    for algo, seed, best, elapsed, fps in results:
        print("%-6s %6d %14.2f %10.0f %12.0f" % (algo, seed, best, elapsed, fps))

    print("\nper-algorithm mean over seeds:")
    for algo in args.algorithms:
        rows = [r for r in results if r[0] == algo]
        if rows:
            rewards = [r[2] for r in rows]
            print("  %-5s %8.2f +/- %.2f  over %d seeds"
                  % (algo, np.mean(rewards), np.std(rewards), len(rows)))
    print("\nEvaluate the winner properly with:  make evaluate-grid")
    print("Reward alone does not distinguish a walking policy from a standing "
          "one - that is exactly the trap the original comparison fell into.")


if __name__ == "__main__":
    main()
