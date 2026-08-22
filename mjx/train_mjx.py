"""Train the Go2 on GPU with MJX + brax PPO.

    # smoke test the wiring (works on CPU, slowly):
    python -m mjx.train_mjx --smoke

    # a real run (needs a CUDA GPU and jax[cuda]):
    python -m mjx.train_mjx --timesteps 60000000 --num-envs 4096

Requires `jax`, `mujoco-mjx` and `brax`. For GPU:

    pip install -U "jax[cuda12]" mujoco-mjx brax

This is the throughput path, not the canonical one. The CPU environment in
`envs/go2_env.py` remains what you evaluate, render and drive by hand; it uses
the unsimplified collision model, and `scene_mjx.xml` does not. Train here,
evaluate there, and expect a small gap.

Why brax's PPO rather than `ppo_from_scratch/ppo.py`: the from-scratch
implementation is NumPy/PyTorch and would move every observation off the GPU
each step, which throws away the entire reason for using MJX. Brax's PPO keeps
the rollout, the advantage computation and the update all inside one jitted
JAX program. It is the same algorithm - chapters 4 and 5 apply unchanged.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brax.envs.base import Env, State  # noqa: E402

from envs.commands import ranges_from_config  # noqa: E402
from mjx.mjx_env import Go2MJXEnv  # noqa: E402


class BraxGo2(Env):
    """Adapt Go2MJXEnv to brax's Env interface.

    Brax's wrappers (episode limit, auto-reset, domain randomisation) all expect
    `reset(rng) -> State` and `step(state, action) -> State`, with the
    environment's own bookkeeping carried in `state.info`. That is why the
    underlying environment returns a plain dict: it drops straight into `info`
    and brax's auto-reset can `jnp.where` over it like any other pytree.
    """

    # Keys in state.info owned by Go2MJXEnv. Everything else in info belongs to
    # brax's wrappers and must be passed through untouched.
    _OWN_INFO_KEYS = ("rng", "cmd", "gait_freq", "cmd_height", "phase",
                      "prev_action", "prev_joint_vel", "feet_air_time",
                      "last_contact", "step")

    def __init__(self, env: Go2MJXEnv):
        self.env = env

    @property
    def observation_size(self):
        return self.env.obs_size

    @property
    def action_size(self):
        return self.env.action_size

    @property
    def backend(self):
        return "mjx"

    def _zero_metrics(self):
        """Brax scans over the step function, so the metrics dict must have the
        SAME key set on every step - including the very first one produced by
        reset. Adding a key in step() and not in reset() is a pytree-structure
        mismatch inside jax.lax.scan, which is what this method exists to
        prevent."""
        m = {"reward": jnp.array(0.0), "height": jnp.array(0.0)}
        for name, weight in self.env.weights.items():
            if weight != 0.0:
                m["rew/" + name] = jnp.array(0.0)
        return m

    def reset(self, rng):
        s = self.env.reset(rng)
        obs = s.pop("obs")
        reward = s.pop("reward")
        done = s.pop("done")
        data = s.pop("data")
        metrics = self._zero_metrics()
        metrics["reward"] = reward
        metrics["height"] = data.qpos[2]
        return State(pipeline_state=data, obs=obs, reward=reward, done=done,
                     metrics=metrics, info=s)

    def step(self, state, action):
        # Brax's wrappers add their own keys to info (steps, truncation,
        # first_obs, ...). Pull out only the ones this environment owns, and
        # write only those back, or the wrapper's bookkeeping is discarded and
        # the scan carry changes shape.
        inner = {k: state.info[k] for k in self._OWN_INFO_KEYS}
        inner["data"] = state.pipeline_state
        inner["obs"] = state.obs
        inner["reward"] = state.reward
        inner["done"] = state.done

        out, terms = self.env.step(inner, action)

        obs = out.pop("obs")
        reward = out.pop("reward")
        done = out.pop("done")
        data = out.pop("data")

        metrics = dict(state.metrics)
        metrics["reward"] = reward
        metrics["height"] = data.qpos[2]
        # Per-term reward logging, the same diagnostic the CPU trainer provides
        # (docs/13-training-diagnostics.md). Brax averages metrics for you.
        for name, value in terms.items():
            metrics["rew/" + name] = value * self.env.dt

        info = dict(state.info)
        info.update(out)
        return state.replace(pipeline_state=data, obs=obs, reward=reward,
                             done=done, metrics=metrics, info=info)


def build_env(config_path=None, num_envs_hint=None):
    ranges = None
    weights = None
    if config_path and os.path.exists(config_path):
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cmd_cfg = cfg.get("commands", {}) or {}
        # Start from the FINAL ranges: MJX runs are long enough and parallel
        # enough that the curriculum matters much less, and brax has no
        # per-rollout callback hook to drive one.
        ranges = ranges_from_config(cmd_cfg.get("final"))
        weights = (cfg.get("reward") or {}).get("weights")
    return BraxGo2(Go2MJXEnv(command_ranges=ranges, reward_weights=weights))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/training_config.yaml")
    p.add_argument("--timesteps", type=int, default=60_000_000)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--unroll-length", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/go2_mjx_params.pkl")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run to verify the wiring. Works on CPU.")
    args = p.parse_args()

    if args.smoke:
        args.timesteps, args.num_envs = 2048, 8
        args.num_minibatches, args.unroll_length = 1, 4
        args.batch_size, args.episode_length = 8, 40

    print("jax devices:", jax.devices())
    if args.smoke:
        print("SMOKE MODE - verifying the wiring, not training anything.")
    elif jax.devices()[0].platform == "cpu":
        print("WARNING: no GPU visible. MJX on CPU is far slower than the "
              "CPU MuJoCo path in scripts/train.py. Install jax[cuda12].")

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo_train

    env = build_env(args.config)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        # Same architecture as the SB3 config, so the two are comparable.
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
    )

    times = [time.perf_counter()]

    def progress(step, metrics):
        times.append(time.perf_counter())
        reward = metrics.get("eval/episode_reward", float("nan"))
        length = metrics.get("eval/avg_episode_length", float("nan"))
        fps = step / max(times[-1] - times[0], 1e-9)
        print("step %10d  reward %8.2f  len %7.1f  %8.0f steps/s"
              % (step, reward, length, fps))

    make_inference_fn, params, _ = ppo_train.train(
        environment=env,
        num_timesteps=args.timesteps,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        # These are the same PPO hyperparameters as configs/training_config.yaml
        # where the two APIs express the same thing. Chapters 4 and 5 apply.
        learning_rate=3e-4,
        entropy_cost=0.002,
        discounting=0.99,
        gae_lambda=0.95,
        clipping_epsilon=0.2,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=5,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        normalize_observations=True,
        network_factory=network_factory,
        seed=args.seed,
        progress_fn=progress,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    from brax.io import model

    model.save_params(args.out, params)
    print("saved " + args.out)


if __name__ == "__main__":
    main()
