# 15. Results

This chapter separates three kinds of result, because they have very different
evidential status and mixing them is how documentation becomes marketing.

1. **Verified engineering results** — measured, reproducible, independent of how
   well any policy trains.
2. **v1's published numbers** — reproduced from `logs/` as they were reported,
   with the caveats that apply.
3. **v2 policy performance** — measured by running the evaluation scripts on a
   trained checkpoint. How to produce it, and what to expect.

## 15.1 Verified engineering results

These do not depend on a trained policy and can be reproduced in minutes.

### The nominal pose (bug B1)

```
$ python scripts/check_env.py

Nominal pose (bug B1)
---------------------
default_joint_pos : [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8]
nominal height    : 0.270 m
matches Go2 home  : YES
```

and the v1 pose, checked against the model's own joint limits:

```
FL_calf_joint    qpos0=+0.000 limits=[-2.723,-0.838] *** OUTSIDE LIMITS ***
FR_calf_joint    qpos0=+0.000 limits=[-2.723,-0.838] *** OUTSIDE LIMITS ***
RL_calf_joint    qpos0=+0.000 limits=[-2.723,-0.838] *** OUTSIDE LIMITS ***
RR_calf_joint    qpos0=+0.000 limits=[-2.723,-0.838] *** OUTSIDE LIMITS ***
```

Both directions are pinned as tests: `test_nominal_pose_is_a_standing_pose`
asserts the fix, and `test_v1_nominal_pose_would_not_stand` asserts that driving
the same PD controller toward v1's pose collapses the robot within two seconds.

### Standing stability versus PD gain

Zero action, 150 control steps, then averaged. Sag is measured from the 0.27 m
nominal pose.

| $k_p$ | $k_d$ | settled height | sag | peak torque |
|---|---|---|---|---|
| 20 | 0.5 | 0.196 m | 7.4 cm | 8.8 N·m |
| 25 | 0.6 | 0.206 m | 6.4 cm | 10.4 N·m |
| 40 | 1.0 | 0.242 m | 2.8 cm | 8.0 N·m |
| **55** | **1.4** | **0.254 m** | **1.6 cm** | — |
| 60 | 1.5 | 0.256 m | 1.4 cm | 7.1 N·m |
| 80 | 2.0 | 0.261 m | 0.9 cm | 6.8 N·m |

Height standard deviation at $k_p = 55$ is 0.0002 m — the robot stands still,
not oscillating.

### Swing clearance (bug B17)

A scripted open-loop trot, no policy involved. Measured maximum foot air time
and agreement with the commanded gait schedule:

| $k_p$ | action scale | stance extension | max air time | schedule match |
|---|---|---|---|---|
| 40 | 0.25 | no | 0.12 s | 0.61 |
| 40 | 0.25 | yes | 0.24 s | 0.61 |
| 40 | 0.35 | yes | 0.24 s | 0.74 |
| **55** | **0.30** | **yes** | **0.20 s** | **0.78** |
| 80 | 0.35 | yes | 0.30 s | 0.84 |

Interpretation in [chapter 13, §13.5](13-training-diagnostics.md). The short
version: a kinematic sweep says 0.25 rad retracts a foot 5.9 cm, which should be
ample — but while two legs retract the other two sag under the whole robot, and
the trunk drops by nearly as much as the swing foot rises.

### Speed envelope (bug B20)

Top speed of a scripted open-loop trot, best over swing amplitude and phase.
No policy, no learning:

| action scale | 1.5 Hz | 2.0 Hz | 2.5 Hz | 3.0 Hz |
|---|---|---|---|---|
| 0.30 | 0.52 | 0.62 | 0.65 | 0.69 |
| **0.40** | **0.57** | **0.94** | **1.05** | **1.08** |

A static kinematic sweep of stance-foot travel gives 0.29 m of stride at
`action_scale = 0.40`, predicting 0.87 m/s at 3 Hz. The measured 1.08 m/s is
higher, because body pitch, foot roll on the spherical contact and dynamic
effects add stride the fixed-base sweep cannot see. The sweep is a useful lower
bound and a bad ceiling.

`action_scale` stops at 0.40 because $k_p 	imes lpha = 22$ N·m against a
23.7 N·m hip and thigh actuator limit.

### Does a competent gait out-score doing nothing?

The single most important pre-training check, and the one that would have caught
all three v2 stalls. Scripted trot versus zero action, under the final reward:

| command | scripted trot | standing | winner |
|---|---|---|---|
| 0.50 m/s | $+0.02405$ | $+0.03669$ | standing |
| 0.70 m/s | $+0.03298$ | $+0.03069$ | **trot** (+7%) |
| 0.85 m/s | $+0.03785$ | $+0.02891$ | **trot** (+31%) |

The scripted controller is open loop and always runs at 0.94 m/s, so at a
command of 0.5 it overshoots and is correctly penalised. At commands near its
own speed it wins clearly.

Before the B19 fix, the trot lost at *every* command, by up to $-0.08$ per step.

### The gait mechanism, measured on a partially-trained policy

At 3M steps (before the B20 fix, so it trots on the spot rather than
travelling), `scripts/gait_analysis.py` at a trot command:

```
schedule match      : 86.8%
foot                       duty  stride Hz phase (meas)  phase (ref)
FL (front left)           0.563       1.99         0.00         0.00
FR (front right)          0.803       1.99         0.36         0.50
RL (rear left)            0.640       2.00         0.48         0.50
RR (rear right)           0.520       2.00         0.00         0.00
```

Stride frequency is exactly the commanded 2.00 Hz on all four feet, and the
diagonal pairing is correct (FL with RR near phase 0, FR with RL near 0.5). The
phase-clock mechanism of chapter 11 works: the policy reads the clock from its
observation and matches the schedule.

### The do-nothing reward baseline

What a policy that never moves collects, Monte-Carlo over the configured command
distribution:

| initial command ranges | $\sigma_v$ | do-nothing tracking score |
|---|---|---|
| $[-0.3,0.6],[-0.2,0.2],[-0.4,0.4]$ | 0.25 | **0.79** |
| $[-0.5,1.0],[-0.3,0.3],[-0.6,0.6]$ | 0.25 | 0.62 |
| **$[-0.5,1.0],[-0.3,0.3],[-0.6,0.6]$** | **0.20** | **0.57** |
| $[-1.0,1.5],[-0.7,0.7],[-1.5,1.5]$ | 0.20 | 0.30 |

The shipped configuration, after the B20 feasibility clamp, sits at **0.571**
against a promotion threshold of 0.85 — reported by `scripts/check_env.py`
before every run.

The first row is what this repository originally shipped, against a curriculum
promotion threshold of 0.85. A statue scored 0.79.

### Throughput

| | steps/s |
|---|---|
| environment alone, before optimisation | 392 |
| environment alone, after | **810** |

| envs | backend | steps/s (end to end) |
|---|---|---|
| 4 | subproc | 382 |
| 6 | subproc | 486 |
| 8 | subproc | 533 |
| 10 | subproc | 571 |
| 10 | dummy | 290 |
| 16 | subproc | **616** |

### Test suite

69 tests, all passing: quaternion transforms against MuJoCo's own
`mju_rotVecQuat`, the Euler convention, gait-schedule periodicity and duty
factors, every reward term's sign and bounds, observation shape and finiteness,
command conditioning, contact detection, and Gymnasium API compliance via SB3's
`check_env`.

Two of them assert that v1's bugs were real rather than merely suspected:
`test_v1_nominal_pose_would_not_stand` and
`test_v1_gait_reward_was_maximised_by_standing_still`. Four more
(`tests/test_mjx.py`) assert that the CPU and MJX backends compute the same
reward from the same state, which is the claim that makes sharing
`envs/rewards.py` across the two backends worth anything.

## 15.2 v1's published numbers

Reproduced from `logs/` as reported.

| Metric | v1 |
|---|---|
| Mean forward speed | 0.737 m/s |
| Peak episode speed | 0.874 m/s |
| Mean episode length | 277 steps (11.1 s at 25 Hz) |
| Evaluation episodes | 50 |
| Training | 5M steps, 8 environments |

Two caveats, neither of which makes the numbers wrong, both of which limit what
they mean.

**They measure one command.** Forward speed at a fixed forward command. There is
no lateral or angular measurement, because the policy had no lateral or angular
capability and the environment had no way to ask for one.

**Mean episode length was 277 of a possible 1000.** The robot fell, on average,
after 11 seconds. The forward-speed figure is an average over the interval
before it fell.

`logs/algorithm_comparison.md` is analysed separately in
[chapter 7](07-why-not-sac-td3.md). Its headline conclusion is not supported by
its experiment — in particular, all three algorithms were trained on the v1
environment, in which walking was kinematically unreachable (B1).

## 15.3 v2 policy performance

### How to measure it

The right measurement for a command-conditioned policy is tracking error across
the command space, not speed at one command:

```bash
make evaluate-grid                      # sweep each axis independently
make evaluate                           # random commands from the full envelope
make gait-analysis                      # duty factor, phase offsets, schedule match
make gait-all                           # one gait diagram per gait
```

`evaluate --grid` sweeps $v_x \in [-1.0, 1.5]$, $v_y \in [-0.75, 0.75]$ and
$\omega_z \in [-1.5, 1.5]$ independently, plus four diagonal commands, holding
each for 8 seconds after a 1-second settling period.

### What to report

| Metric | Why |
|---|---|
| mean $\lvert v_x \rvert$ error | forward/backward tracking |
| mean $\lvert v_y \rvert$ error | **strafing — v1 could not do this at all** |
| mean $\lvert \omega_z \rvert$ error | **turning — v1 could not do this at all** |
| survival rate | fraction of commands held without falling |
| mean feet in contact | duty factor; should be near $4\beta = 2$ for a trot |
| schedule match | the `gait_phase` reward as a percentage |
| action jerk | smoothness; comparable across policies |

The two rows in bold are the point. A forward-only policy scores well on $v_x$
and badly on the other two, so the measurement makes the v1→v2 capability
difference quantitative rather than rhetorical.

### Expected training trajectory

From [chapter 13, §13.4](13-training-diagnostics.md), for the default config at
16 environments:

| Steps | `ep_len_mean` | `rew/feet_air_time` | `tracking_lin_err` | `curriculum/level` |
|---|---|---|---|---|
| 0–200k | rising from ~100 | ~0 | ~0.45 | 0 |
| 200k–1M | 400–1000 | becoming non-zero | 0.35–0.45 | 0 |
| 1M–4M | ~1000 | clearly positive | 0.2–0.35 | climbing |
| 4M–10M | ~1000 | positive and stable | 0.1–0.2 | approaching 1.0 |

`rew/feet_air_time` becoming non-zero is the milestone that matters. It is
credited only on touchdown, so a non-zero value is the first proof that a foot
has genuinely left the ground.

### Reward shaping, verified without training

One measurement that *is* complete, and that gates everything above. Under the
final reward, a scripted open-loop trot is compared against zero action, per
control step:

| term | standing | scripted trot | diff |
|---|---|---|---|
| `gait_phase` | $+0.01500$ | $+0.02343$ | $+0.00843$ |
| `feet_clearance` | $-0.00601$ | $-0.00314$ | $+0.00288$ |
| `feet_air_time` | $\phantom{+}0.00000$ | $+0.00056$ | $+0.00056$ |
| `torques` | $-0.00074$ | $-0.00292$ | $-0.00218$ |
| `ang_vel_xy` | $-0.00000$ | $-0.00147$ | $-0.00147$ |
| `feet_slip` | $-0.00000$ | $-0.00085$ | $-0.00085$ |

Stepping pays about $+0.012$ per step gross and $+0.007$ net of its costs,
before any credit for locomotion; tracking 0.8 m/s adds a further $+0.029$.

Before the B19 fix the `feet_air_time` row read $-0.08$ per step, and a perfect
trot scored roughly four times worse than standing still. That single number is
why three successive runs stalled, and it was measurable in minutes without
training anything.

### Status

At the time of writing, a 20M-step run is in progress on the development machine
(about 8.8 hours at 631 steps/s). **The final policy numbers are not filled in
here, and should not be, until that run completes and `make evaluate-grid` has
been executed against the checkpoint.**

To fill this section in:

```bash
make evaluate-grid > docs/results-grid.txt
make evaluate      >> docs/results-grid.txt
make gait-all
```

and paste the tables. The v1 comparison is only meaningful if the v2 numbers
come from an actual evaluation run rather than from an expectation.

## 15.4 What is honestly claimed

**Established:**

- v1's nominal pose was outside the robot's joint limits, and its reachable
  target set did not intersect the calf's legal range. This is arithmetic, not
  interpretation.
- v1's gait reward was maximised, permanently, by a motionless robot.
- v1's curriculum could not command lateral or angular velocity, and its reward
  had no term for lateral tracking. A forward-only gait was the only outcome
  the environment permitted.
- Resuming v1 training discarded the observation normalisation statistics.
- The v2 environment can express a trot: a hand-written open-loop controller
  reaches 0.78 schedule match with no learning at all.
- Environment throughput doubled after profiling.
- Under the final reward, a scripted trot out-scores standing on the stepping
  terms; under the reward as it stood before B19, it did not.

**Not yet established:** that the v2 *policy* tracks commands well. That
requires the training run to finish and the evaluation to be run. Until then
this chapter says so.

---

**Previous:** [14. The debugging log](14-debugging-log.md) ·
**Next:** [16. What sim-to-real would take](16-sim-to-real.md)
