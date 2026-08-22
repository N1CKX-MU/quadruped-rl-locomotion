# 17. Scaling with MJX

Sixteen CPU environments is the wrong shape for this problem. This chapter
measures why, and describes the path to the right one.

## 17.1 The measurement

Benchmarked on the development machine (8 physical cores, 12 logical), MuJoCo
3.12, PyTorch 2.13 CPU:

**Environment alone** (no policy, no learning):

| | steps/s |
|---|---|
| before optimisation | 392 |
| after replacing `np.cross` and caching the frame conversions | **810** |

The 2.07× came from a single profiling session and is worth recording, because
the culprit is not where you would guess:

```
   ncalls  tottime  cumtime  function
    20000    2.217    2.217  mujoco._functions.mj_step
    28012    1.757    5.156  numpy.cross
   168072    1.148    1.752  numpy._core.numeric.normalize_axis_tuple
    84036    0.806    2.986  numpy._core.numeric.moveaxis
```

**`np.cross` cost more wall-clock time than the physics.** It is a fully general
$n$-dimensional routine that calls `moveaxis` and `normalize_axis_tuple` on
every invocation, and the environment called it fourteen times per control step
on 3-element vectors. Writing out the six multiplications explicitly, and
caching the three frame conversions that the observation, the reward and the
termination check all wanted separately, doubled throughput.

**End-to-end training**, including the PPO update:

| envs | vec backend | steps/s |
|---|---|---|
| 4 | subproc | 382 |
| 6 | subproc | 486 |
| 8 | subproc | 533 |
| 10 | subproc | 571 |
| 10 | **dummy** (in-process) | 290 |
| 16 | subproc | **616** |

Two things to notice.

**Scaling is badly sublinear.** One environment alone does 810 steps/s; sixteen
in parallel do 616 *in total*. Adding fifteen environments made the system
slower per environment by a factor of twenty.

**`SubprocVecEnv` beats `DummyVecEnv` here**, despite the Windows pipe overhead,
which tells you the bottleneck is genuine CPU contention rather than
serialisation. Both numbers are worth having; `--vec-env` exposes the choice so
you can benchmark on your own machine rather than trusting this table.

## 17.2 What this means for a training run

At 616 steps/s:

| Budget | Wall clock |
|---|---|
| 1M steps | 27 minutes |
| 10M steps | 4.5 hours |
| 50M steps | 22.5 hours |

For comparison, the legged-locomotion literature (Rudin et al., *Learning to
Walk in Minutes*, 2021) trains comparable policies in **under twenty minutes**,
using 4096 parallel environments on a single GPU. The difference is not the
algorithm — it is PPO in both cases — it is that the simulation runs on the GPU
alongside the policy.

## 17.3 Why CPU parallelism runs out

Each CPU environment is a separate process running a serial rigid-body solve.
The costs that do not go away:

- **Contact solving is serial per environment** and does not vectorise across
  environments in the CPU MuJoCo build.
- **Eight physical cores.** Beyond eight workers you are timeslicing.
- **Every step crosses a process boundary.** Observations are pickled, written
  to a pipe, and read back, sixteen times per vectorised step.
- **The policy forward pass is a batch of 16.** A GPU is idle at that size; even
  a CPU is barely warmed up. Almost all the arithmetic capacity of the machine
  is unused.

The last point is the real one. The hardware is capable of thousands of
simultaneous rigid-body solves; the architecture asks it for sixteen.

## 17.4 MJX

MJX is MuJoCo re-implemented in JAX. The physics becomes a pure function that
`jax.vmap` maps across a batch and `jax.jit` compiles to GPU kernels. Instead of
$N$ processes each stepping one robot, you get one program stepping $N$ robots as
a batched tensor operation.

The consequences:

| | CPU MuJoCo | MJX |
|---|---|---|
| environments | 16 | 4096+ |
| steps/s | ~600 | $10^5$–$10^6$ |
| where the policy lives | CPU, batch 16 | GPU, batch 4096 |
| observation transfer | pickled through pipes | stays on the GPU |
| 50M steps | 22 hours | minutes |

Nothing crosses a process boundary, because nothing is in another process.

### What it costs

Three real constraints, worth knowing before starting.

**Everything must be functional and shape-static.** No Python `if` on a traced
value, no data-dependent shapes, no in-place mutation. Control flow becomes
`jnp.where`. Every episode in the batch runs for the same number of steps, so
termination is a mask rather than a break.

**The contact model is more constrained.** MJX supports a subset of MuJoCo's
collision geometry, and the number of contacts must be bounded statically.
Menagerie ships `go2_mjx.xml` and `scene_mjx.xml` precisely for this — simplified
collision geometry with a fixed contact budget. The physics is *not* identical to
the CPU model, which matters if you intend to evaluate on one and train on the
other.

**Debugging is harder.** Inside a `jit`, you cannot print an intermediate value
without `jax.debug.print`, and a shape error surfaces as a trace-time exception
far from its cause.

## 17.5 The path taken here

The CPU environment stays canonical. It is the one that renders, that
`scripts/play.py` drives, that `scripts/evaluate.py` measures, and that produces
video. It is also the one whose physics is unsimplified.

MJX is an **optional training accelerator**, and it is deliberately last in the
build order. The reasoning is the same as chapter 13's: optimising the training
loop before the reward function is correct spends GPU hours discovering that
your reward is wrong faster. The two bugs that cost the most in this project
(B1 and B17) would have been reproduced at 4096× the speed and been no easier to
find.

### Sharing the reward maths

The one design decision made *in advance* for MJX is that `envs/rewards.py` and
`envs/gait.py` are written array-generically. Every function uses `xp.where`
rather than Python branching, avoids in-place mutation, and dispatches on the
array type:

```python
def _backend(x):
    if type(x).__module__.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp
        return jnp
    return np
```

so the identical reward function runs under both backends. This matters more
than it looks: two implementations of a sixteen-term reward will diverge, and
the divergence will be discovered as an unexplained performance difference
between the CPU and GPU policies months later.

The `RewardState` dataclass exists for the same reason. The reward terms never
reach into `mjData`, so there is nothing MuJoCo-specific for them to reach into
under MJX.

### What still has to be written per backend

- state extraction — building a `RewardState` from `mjx.Data` instead of
  `mjData`
- contact detection — MJX exposes contacts differently and the geom filtering
  has to be vectorised
- the reset and command sampler — must become functional, with an explicit PRNG
  key
- the training loop — either brax's PPO, or a JAX port of
  `ppo_from_scratch/ppo.py`

That is a real piece of work, on the order of a few days, which is why it is
staged rather than assumed.

## 17.6 Cheaper things to do first

Not everything requires JAX.

**A CUDA build of PyTorch.** The environment ships `torch 2.13.0+cpu` by
default, because that is what `pip install -r requirements.txt` resolves to.
With 16 environments the policy update is a small fraction of the time, so this
buys little now — but it costs nothing and matters as the network grows:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Profile before optimising.** The 2× in §17.1 came from twenty minutes with
`cProfile` and would never have been guessed. `np.cross` on 3-vectors is a trap
that is presumably still costing other MuJoCo projects a factor of two.

**Benchmark your own machine.** `--n-envs` and `--vec-env` are flags for a
reason; the table in §17.1 is one laptop.

**Reduce what the environment computes.** `_contact_state` iterates every
contact and calls `mj_contactForce` on the ground contacts. It is currently 6% of
the step and was 3% before the `np.cross` fix removed the larger cost around it.

## 17.7 The honest summary

The current setup trains a walking policy overnight. That is entirely adequate
for developing and debugging the environment, and it is what most of this
repository's value is in.

It is roughly two orders of magnitude off the state of the art in throughput,
and the gap is architectural rather than algorithmic. If you intend to run
ablations across a dozen reward configurations, or to train on terrain, the MJX
path stops being optional.

---

**Previous:** [16. What sim-to-real would take](16-sim-to-real.md) ·
**Back to:** [index](README.md)
