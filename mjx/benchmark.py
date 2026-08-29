"""Measure what this GPU can actually do, rather than quoting a paper.

    python -m mjx.benchmark                  # find the ceiling, then benchmark
    python -m mjx.benchmark --envs 1024      # benchmark one size
    python -m mjx.benchmark --max-search 8192

The MJX literature quotes 4096 parallel environments. That figure comes from
datacentre cards. On a 4 GB laptop GPU it is fiction, and quoting it would be
the same mistake as bug B20 - asserting a number the hardware cannot deliver
instead of measuring one.

So this script does two things:

1. **Finds the ceiling.** Doubles the environment count until allocation fails
   or throughput stops improving, then reports the largest size that worked.
2. **Benchmarks honestly.** Times only the steady state, after JIT compilation,
   with `block_until_ready` so that JAX's asynchronous dispatch cannot make the
   numbers look better than they are. Forgetting that is the single most common
   way JAX benchmarks end up wrong by an order of magnitude.

Reported throughput is *environment steps per second* - the same unit
`scripts/train.py` prints - so the two paths are directly comparable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set before jax is imported. Without this JAX grabs 75% of the card up
# front, which both hides genuine out-of-memory behaviour and starves anything
# else using the GPU - including, on a laptop, the desktop compositor.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def gpu_memory_mb():
    """Bytes in use on device 0, if the backend reports it."""
    try:
        stats = jax.devices()[0].memory_stats()
        return stats.get("bytes_in_use", 0) / 1e6, stats.get("bytes_limit", 0) / 1e6
    except Exception:
        return None, None


def benchmark(env, n_envs, steps=100, warmup_steps=3, seed=0):
    """Return (env_steps_per_second, peak_mb) or None if it did not fit."""
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    try:
        keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
        t0 = time.perf_counter()
        state = reset(keys)
        state["obs"].block_until_ready()
        compile_reset = time.perf_counter() - t0

        action = jnp.zeros((n_envs, env.action_size))

        t0 = time.perf_counter()
        state, _ = step(state, action)
        state["obs"].block_until_ready()
        compile_step = time.perf_counter() - t0

        for _ in range(warmup_steps):
            state, _ = step(state, action)
        state["obs"].block_until_ready()

        # Steady state. block_until_ready is what makes this a measurement
        # rather than a measurement of how fast Python can enqueue work.
        t0 = time.perf_counter()
        for _ in range(steps):
            state, _ = step(state, action)
        state["obs"].block_until_ready()
        elapsed = time.perf_counter() - t0

    except Exception as exc:  # OOM surfaces as RESOURCE_EXHAUSTED / XlaRuntimeError
        name = type(exc).__name__
        first = str(exc).strip().splitlines()[0][:90] if str(exc).strip() else ""
        print("    %6d envs -> FAILED (%s) %s" % (n_envs, name, first))
        return None

    used, limit = gpu_memory_mb()
    throughput = n_envs * steps / elapsed
    print("    %6d envs -> %10.0f env-steps/s   compile %4.1f+%4.1f s%s"
          % (n_envs, throughput, compile_reset, compile_step,
             ("   %.0f/%.0f MB" % (used, limit)) if used else ""))
    return throughput, used


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--envs", type=int, default=None,
                   help="Benchmark this size only, instead of searching.")
    p.add_argument("--start", type=int, default=64)
    p.add_argument("--max-search", type=int, default=8192)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--xml", default=None)
    args = p.parse_args()

    devices = jax.devices()
    print("jax %s   devices: %s" % (jax.__version__, devices))
    if devices[0].platform == "cpu":
        print("\nWARNING: running on CPU. MJX on CPU is much SLOWER than the\n"
              "plain CPU MuJoCo path in scripts/train.py - this benchmark is\n"
              "only meaningful on a GPU.\n")
    used, limit = gpu_memory_mb()
    if limit:
        print("device memory: %.0f MB total" % limit)

    from mjx.mjx_env import Go2MJXEnv

    t0 = time.perf_counter()
    env = Go2MJXEnv(**({"xml_path": args.xml} if args.xml else {}))
    print("model loaded in %.1f s   obs %d  act %d\n"
          % (time.perf_counter() - t0, env.obs_size, env.action_size))

    if args.envs:
        print("  benchmarking %d environments:" % args.envs)
        benchmark(env, args.envs, steps=args.steps)
        return

    print("  searching for the largest workable environment count:")
    best = None
    n = args.start
    while n <= args.max_search:
        result = benchmark(env, n, steps=args.steps)
        if result is None:
            break
        throughput, _ = result
        if best is not None and throughput < best[1] * 1.05:
            print("    (throughput stopped improving; %d is the useful ceiling)"
                  % best[0])
            break
        best = (n, throughput)
        n *= 2

    print()
    if best:
        print("BEST: %d environments at %.0f env-steps/s" % best)
        print("CPU baseline (scripts/train.py, 16 envs): ~616 env-steps/s")
        print("speedup: %.1fx" % (best[1] / 616.0))
        hours = 20e6 / best[1] / 3600
        print("20M steps would take %.2f hours (%.0f minutes)" % (hours, hours * 60))
    else:
        print("Nothing fit. Try --start 32.")


if __name__ == "__main__":
    main()
