# Quadruped RL Locomotion — Unitree Go2

Command-conditioned locomotion for the Unitree Go2 in MuJoCo, trained with PPO.
The robot walks forwards, backwards and sideways, turns on the spot, holds a
commanded body height, and trots at a commandable step frequency — tracking
velocity commands to **0.02 m/s** and yaw rate to **0.035 rad/s**, with a
**100% survival rate** and a measured duty factor of exactly 0.500.

The gait is a *phase schedule* the reward compares foot contacts against, so
trot, pace, bound and walk are all expressible. The shipped checkpoint is
trained on trot only — see [Results](#results) for what it does and does not
do.

There is a **book** in [`docs/`](docs/README.md) that derives all of this from
scratch: the maths of reinforcement learning, the algorithm line by line, the
robot, the reward design, and a debugging log of every defect found on the way.

```bash
git clone https://github.com/N1CKX-MU/quadruped-rl-locomotion.git
cd quadruped-rl-locomotion
make setup
make check          # 10-second sanity check; run this before training
make train
make play           # drive it yourself: WASD, Q/E, 1-5 for gaits
```

---

## What changed in v2

v1 trained a policy that walked forwards at 0.74 m/s and could do nothing else.
That was not a tuning limit. It was the sum of sixteen specific defects, none of
which produced an error message, all of which survived five documented training
runs. Four more (B17-B20) were found in v2 and are listed alongside them.

The headline one:

> **v1's "default standing pose" placed all four calf joints outside their own
> travel limits.** The pose was taken from `qpos0` (all joints at 0 rad) rather
> than the model's `home` keyframe. The Go2's calf range is
> $[-2.723, -0.838]$ rad — it does not contain zero. With an action scale of
> 0.5 rad, the policy's entire reachable target set was $[-0.5, +0.5]$, which
> does not intersect the joint's legal range at all. Every episode began in an
> illegal configuration, and for every action the policy could emit, the
> commanded knee angle was physically unreachable.

The full list is [`docs/14-debugging-log.md`](docs/14-debugging-log.md). Summary:

| # | Bug | Impact |
|---|---|---|
| B1 | Nominal pose outside the joint limits | **Defined v1's ceiling** |
| B2 | Velocity tracked in the world frame, not the body frame | Turning incoherent |
| B3 | Reward could only express forward motion | Forward-only gait |
| B4 | Gait reward maximised by a robot standing perfectly still | No gait signal |
| B5 | Curriculum made 8 blocking IPC round trips per environment step | Throughput loss |
| B6 | PD torque held constant for 40 ms | Judder |
| B7 | 25 Hz control | Judder; halved the planning horizon |
| B8 | Every episode started from one initial condition | Brittle |
| B9 | A new OpenGL context per rendered frame | Slow rendering |
| B10 | Contact detector counted foot-on-shin self-contacts | Wrong gait signal |
| B11 | Pushes fired on an exactly periodic schedule, teleporting momentum | Memorisable |
| B12 | Resuming training silently discarded the normalisation statistics | **Silent policy destruction** |
| B13 | Eval environment kept updating its own normalisation | Noisy eval curve |
| B14 | 2560 gradient steps per PPO rollout | Instability and waste |
| B15 | Observation docstring said 49; the space was 53 | Indicator |
| B16 | `mujoco.viewer` used without being imported | `render("human")` crashed |
| B17 | Swing clearance too marginal to lift a foot (found in v2) | No gait |
| B18 | Stepping reward was piecewise constant, so its gradient was zero (v2) | No gradient toward stepping |
| B19 | Stride reward made a correct trot score worse than standing still (v2) | Target behaviour was penalised |
| B20 | Command envelope asked for speeds the leg geometry cannot reach (v2) | Tracking reward saturated |

Four things were checked and found **not** to be bugs — the quaternion helper,
the torque units, the timeout bootstrapping, and the PD gains. Those are
recorded too, because a debugging log that lists only confirmed problems
misrepresents how the work goes.

### New capability

| | v1 | v2 |
|---|---|---|
| Commands | one fixed forward speed | $(v_x, v_y, \omega_z)$, resampled every 5 s |
| Directions | forward | forward, backward, strafe, turn, and combinations |
| Gait | emergent, uncontrollable | a commanded phase schedule; shipped policy trained on trot |
| Step frequency | — | commandable, 1.5–3.0 Hz |
| Command feasibility | n/a | every command clamped to a measured speed envelope |
| Body height | — | commandable, 0.27–0.34 m |
| Curriculum | open-loop ramp of one scalar | closed-loop over the full command envelope |
| Control rate | 25 Hz | 50 Hz, with the PD loop at 500 Hz |
| Reward | 8 terms, inline, unlogged | 17 terms, separate functions, each logged |
| Tests | none | 67 |

---

## How it works

**Observation** (50 dims): projected gravity, base-frame angular and linear
velocity, joint positions relative to nominal, joint velocities, previous
action, the velocity command, and a gait clock $(\sin 2\pi\phi, \cos 2\pi\phi)$.
Every dimension is justified in
[`docs/09-observations-and-actions.md`](docs/09-observations-and-actions.md).

**Action** (12 dims): joint position offsets from the Go2's `home` pose, scaled
by 0.40 rad, tracked by a PD controller ($k_p=55$, $k_d=1.4$) recomputed at
every 2 ms physics step. This is the same interface the real robot exposes.

**The gait is a clock, not an emergent property.** A scalar phase
$\phi \in [0,1)$ advances at the commanded step frequency. Each foot has an
offset $\theta_i$, and the schedule says foot $i$ should be in stance while
$(\phi + \theta_i) \bmod 1 < \beta$. The reward asks how many feet currently
match the schedule. Because the schedule is an *input*, one trained policy
serves every gait — swap the offsets and the trot becomes a pace.
See [`docs/11-gaits-and-phase.md`](docs/11-gaits-and-phase.md).

**Reward**: seventeen weighted terms, each a pure function in
[`envs/rewards.py`](envs/rewards.py), each logged separately to TensorBoard, all
weights in YAML. Full derivation and a worked reward-hacking exploit in
[`docs/10-reward-engineering.md`](docs/10-reward-engineering.md).

---

## Two PPO implementations

`scripts/train.py` uses Stable-Baselines3. `ppo_from_scratch/` contains an
annotated ~450-line PPO — rollout buffer, GAE, clipped surrogate, value
clipping, KL early stop, observation normalisation — that trains the same
environment. Every block references a numbered equation in the docs.

```bash
make train                                          # SB3
python ppo_from_scratch/train_scratch.py            # from scratch
tensorboard --logdir logs/tensorboard               # the curves should agree
```

If the two curves agree, you have verified you understand the algorithm. If they
diverge, the difference is a specific implementation detail, and finding it
teaches more than re-reading the paper.

---

## Usage

```bash
make check                    # sanity-check the environment before training
make test                     # 67 unit tests
make train                    # SB3 PPO
make train-scratch            # the from-scratch implementation
make resume CKPT=models/checkpoints/go2_ppo_2000000_steps.zip
make tensorboard

make play                     # drive it by hand
make evaluate                 # tracking error over random commands
make evaluate-grid            # tracking error swept along each command axis
make gait-analysis            # gait diagram, duty factor, phase offsets
make gait-all                 # one diagram per gait
```

On Windows, pass the venv path explicitly:

```bash
make PY=venv/Scripts/python.exe check
```

### Driving it

`make play` opens the MuJoCo viewer with keyboard control:

```
W / S      forward / backward        1..5   trot, pace, bound, walk, pronk
A / D      strafe left / right       [ / ]  step frequency down / up
Q / E      turn left / right         - / =  body height down / up
SPACE      stop                      R      reset
```

### Training multiple gaits

```yaml
commands:
  gaits: ["trot", "pace", "bound", "walk"]
```

Note the caveat in [`docs/11-gaits-and-phase.md`](docs/11-gaits-and-phase.md)
§11.8: with more than one gait you must also switch the observation from the
global clock to the per-foot clock (`clock_signal` in `envs/gait.py`), or the
policy is being asked to satisfy a schedule it cannot see.

---

## Measured performance

Development machine: 8 physical cores (12 logical), MuJoCo 3.12, PyTorch CPU.

**Environment throughput**, after profiling:

| | steps/s |
|---|---|
| before optimisation | 392 |
| after replacing `np.cross` and caching the frame conversions | **810** |

`np.cross` on 3-element vectors cost more wall-clock time than the MuJoCo
physics itself — it is a general $n$-dimensional routine that runs `moveaxis`
and `normalize_axis_tuple` on every call, and the environment invoked it
fourteen times per control step.

**End-to-end training throughput:**

| envs | backend | steps/s |
|---|---|---|
| 4 | subproc | 382 |
| 8 | subproc | 533 |
| 10 | dummy | 290 |
| 16 | subproc | **616** |

So 10M steps is about 4.5 hours on this machine. For the two-orders-of-magnitude
speedup and why it is architectural rather than algorithmic, see
[`docs/17-mjx-and-scaling.md`](docs/17-mjx-and-scaling.md).

<a name="results"></a>
**Policy performance.** 18.76M steps, 16 environments, ~8 hours. Measured over
30 random commands from the trained envelope, 8 s each:

| Metric | v2 | v1 |
|---|---|---|
| mean forward-velocity error | **0.021 m/s** | not measured (no command) |
| mean lateral-velocity error | **0.028 m/s** | **not possible** |
| mean yaw-rate error | **0.035 rad/s** | **not possible** |
| survival rate | **100%** | fell after 277 of 1000 steps |
| mean feet in contact | **2.00 / 4** | — |
| gait schedule match | **96.0%** | n/a |
| duty factor | **0.500** (reference 0.50) | n/a |

Yaw tracking is essentially exact (commanded 1.50 rad/s, achieved 1.508).
Combined commands work: $(0.80, 0, 0.80)$ gives $(0.795, -0.023, 0.765)$.

**What it does not do**, stated plainly: it does **not** change gait on command.
Asked for a pace or a bound it trots anyway, at 50% schedule match, which is
chance. That is expected and predicted — the config ships `gaits: ["trot"]`, so
the phase offsets were constant during training and the gait identity is not in
the observation. Training a multi-gait policy needs the per-foot clock
(`clock_signal` in `envs/gait.py`) and a wider `gaits` list; see
[`docs/11`](docs/11-gaits-and-phase.md) §11.8. It is also not sim-to-real ready
([`docs/16`](docs/16-sim-to-real.md)).

Full tables in [`docs/15-results.md`](docs/15-results.md).

---

## Repository layout

```
envs/
  go2_env.py        the environment (v2)
  go2_env_v1.py     the original, frozen verbatim for comparison and ablation
  rewards.py        16 reward terms, one pure function each
  gait.py           phase clock and gait definitions
  commands.py       command sampling and the adaptive curriculum
callbacks/
  curriculum.py     closed-loop command curriculum
  logging.py        per-reward-term TensorBoard logging
ppo_from_scratch/
  ppo.py            annotated PPO, cross-referenced to the docs
  train_scratch.py  trains the same environment
scripts/
  train.py          SB3 training
  check_env.py      pre-flight sanity check
  play.py           keyboard teleop
  evaluate.py       command-grid evaluation
  gait_analysis.py  gait diagrams and duty/phase measurement
tests/              67 tests: maths, gait schedule, reward terms, env, MJX parity
docs/               the book (17 chapters)
configs/            YAML: reward weights, command ranges, PPO hyperparameters
```

---

## The documentation

[`docs/README.md`](docs/README.md) is the index. In brief:

**Part I — the algorithm.** MDPs and returns; value functions and the advantage;
the policy gradient theorem derived from the log-derivative trick; actor-critic
and a full GAE derivation; TRPO to PPO with the clipped objective explained case
by case; the from-scratch code line by line; and an honest re-reading of this
repository's own PPO-vs-SAC-vs-TD3 comparison.

**Part II — the robot.** The Go2, MuJoCo, and PD control; every observation
dimension; reward engineering with a worked exploit; gaits and phase; curriculum
and domain randomisation.

**Part III — practice.** Reading a training run; the debugging log; results;
what sim-to-real would take; and scaling with MJX.

Two chapters are worth reading even if you skip the rest:
[13 (training diagnostics)](docs/13-training-diagnostics.md) before your first
long run, and [14 (the debugging log)](docs/14-debugging-log.md) for how the
bugs were actually found.

---

## Requirements

MuJoCo 3.x, Gymnasium 1.x, Stable-Baselines3 2.x, PyTorch, and the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) Go2
model (cloned by `make setup`). `requirements.txt` resolves to the CPU build of
PyTorch; install a CUDA build separately if you have a GPU.

## Acknowledgements

Robot model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
The reward structure follows the conventions established by
[legged_gym](https://github.com/leggedrobotics/legged_gym) (Rudin et al.), and
the gait-phase formulation follows the line of work on periodic reward
composition in quadruped locomotion.
