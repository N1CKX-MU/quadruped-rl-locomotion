# 13. Reading a training run

An RL run gives you no error messages. It gives you curves, and it will happily
converge to something wrong while every curve looks plausible. This chapter is
about reading them.

Read this before starting a long run. The difference between debugging in an
afternoon and debugging in a week is almost entirely knowing which trace answers
which question.

```bash
tensorboard --logdir logs/tensorboard/ --port 6006
```

## 13.1 The traces, and what each one is for

### `train/explained_variance` — the most diagnostic single number

$$
\mathrm{EV} = 1 - \frac{\operatorname{Var}[G - \hat V]}{\operatorname{Var}[G]}
$$

How much of the return's variance the critic accounts for.

| Value | Meaning |
|---|---|
| $> 0.9$ | critic is good; advantages are meaningful |
| $0.5 - 0.9$ | acceptable, still learning |
| $\approx 0$ | critic is no better than predicting the mean |
| $< 0$ | critic is *worse* than the mean. Normal for the first few updates; alarming after that |

Why it matters more than the reward curve: the policy gradient is weighted by
advantages estimated from the critic (chapter 3, §3.6). A bad critic does not
merely slow learning, it points the policy in wrong directions. If EV is near
zero after a few hundred thousand steps, nothing downstream is trustworthy and
there is no point tuning the reward.

Causes of persistently low EV, in the order worth checking: the reward has huge
unpredictable spikes (a termination penalty, an un-normalised term); $\gamma$ is
too high for the horizon the critic can actually see; `vf_coef` is too small;
there is genuine unobservable randomness — which for this environment there is,
because a push is invisible to the policy until it has already happened
(chapter 9, §9.10). Expect a permanent ceiling somewhat below 1.0 for that
reason.

### `train/approx_kl` — how far the policy moved

Equation 5.10. Should sit somewhere around `target_kl` (0.02 here).

| | Meaning |
|---|---|
| $\ll 0.01$ | the policy is barely moving. LR too low, or advantages near zero |
| $0.01 - 0.03$ | healthy |
| $> 0.05$ | too large. Expect instability shortly |
| spiking | one bad minibatch. Check `max_grad_norm` |

### `train/clip_fraction` — what fraction of samples hit the clip

Should be **non-zero**. Zero means the clip never activates, so PPO is behaving
as plain policy gradient and you have lost the trust region. 0.05–0.2 is
healthy. Above 0.3 means most of your update is being discarded — the learning
rate or the number of epochs is too high, and you are burning compute to produce
no gradient.

### `train/std` — the policy's exploration noise

Starts at $e^{-1} \approx 0.37$ and should **decrease slowly**. This is the
policy becoming confident.

| | Meaning |
|---|---|
| falls fast (below 0.1 within 1M steps) | premature convergence. Raise `ent_coef` |
| does not fall at all | `ent_coef` too high — v1's 0.01 does this |
| rises | the policy is being pushed toward randomness. Something is wrong |

### `rollout/ep_len_mean` and `episode/mean_length`

The bluntest survival measure. In this environment the truncation limit is 1000
steps (20 s), so:

- rising toward 1000: the robot is learning not to fall
- pinned at 1000: it never falls. **This is not necessarily good** — see §13.3
- falling while `curriculum/level` rises: the curriculum is outrunning the policy

### `rollout/ep_rew_mean`

The headline number, and the least informative one. It confounds every reward
term. A robot that stands still can out-score one that walks badly — that is
precisely what happened to v1 (chapter 10, §10.5). Use it to detect collapse,
not to judge quality.

### `rew/*` — the per-term breakdown

This is where the real information is, and it is the main thing v2 added over
v1. `RewardTermLoggerCallback` records every weighted term separately.

| Trace | Reads as |
|---|---|
| `rew/track_lin_vel_xy` | is it following velocity commands |
| `rew/gait_phase` | are the right feet on the ground at the right time |
| `rew/feet_clearance` | are the swing feet getting anywhere near their target height |
| `rew/feet_air_time` | **is it actually lifting its feet** |
| `rew/feet_slip` | is it skating |
| `rew/collision` | are the knees hitting the ground |
| `rew/action_rate`, `rew/joint_acceleration` | is it juddering |

`rew/feet_air_time` deserves special attention. It is credited only on
touchdown, so **exactly zero means no foot has ever landed, which means no foot
has ever left the ground.** It is the single clearest "is this a gait or a
shuffle" indicator in the whole log, and it is the trace that caught the failure
in §13.5.

### `curriculum/*`

`level` should climb steadily and settle at 1.0. Pinned at 0 means the policy
is not meeting the promotion threshold — check `tracking_score` against the
do-nothing baseline (chapter 12, §12.4) before assuming the threshold is too
strict.

### `diagnostics/tracking_lin_err`, `tracking_ang_err`

Direct, interpretable, in physical units. A well-trained policy on this task
should reach roughly 0.1–0.2 m/s of linear tracking error.

### `time/fps`

Throughput. Watch for it falling over a run — a memory leak, another process, or
a callback doing something expensive per step. v1's curriculum callback made 8
blocking IPC round trips *per environment step*; that shows up here and nowhere
else.

## 13.2 The order to read them in

When a run looks wrong:

1. **`explained_variance`.** If it is near zero, stop. Nothing else is
   meaningful until the critic works.
2. **`ep_len_mean`.** Is it surviving at all? If not, the problem is
   termination/stability, not the task.
3. **`rew/feet_air_time`.** Is it a gait or a shuffle?
4. **`rew/*` generally.** Which term is the largest? Is that the term you
   intended to be largest?
5. **`approx_kl` and `clip_fraction`.** Is the optimiser behaving?
6. **`curriculum/level` against `tracking_score`.** Is the task advancing?

## 13.3 Failure modes, and what they look like

### Standing still

**Looks like:** `ep_len_mean` pinned at the truncation limit, `ep_rew_mean`
respectable and flat, `explained_variance` high (a stationary robot is very
predictable), **`rew/feet_air_time` exactly zero**, `tracking_lin_err` stuck
around the mean commanded speed.

**Is:** the classic local optimum. The reward for doing nothing is close enough
to the reward for succeeding that the intervening risk is not worth it.

**Fix:** compute the do-nothing baseline (chapter 10, §10.4). Widen commands,
sharpen the tracking kernel, raise the dense stepping reward.

### Shuffling

**Looks like:** `rew/gait_phase` high, `rew/feet_air_time` small but non-zero,
`rew/feet_slip` large and negative, forward speed below command.

**Is:** contacts toggling on schedule with almost no swing. The gait reward is
satisfied without any real stride.

**Fix:** raise `feet_air_time`; check the swing clearance is physically
achievable (§13.5).

### Judder

**Looks like:** `rew/action_rate` and `rew/joint_acceleration` dominating the
penalties; visually a high-frequency tremor.

**Fix:** raise those penalties; check the control rate is not too low; check
`std` has not collapsed to near zero, which makes the deterministic policy's
sharp features visible.

### Policy collapse

**Looks like:** `ep_rew_mean` falls off a cliff and does not recover.
`approx_kl` spiked just beforehand.

**Is:** chapter 5, §5.1 — too large a step, worse policy, worse data, no way
back.

**Fix:** lower the learning rate, lower `n_epochs`, tighten `target_kl`, check
`max_grad_norm`.

### Critic divergence

**Looks like:** `train/value_loss` grows without bound, `explained_variance`
goes negative and stays.

**Common cause:** treating truncation as termination (chapter 4, §4.7), which
teaches the critic that states near the time limit are worthless while the
observed returns say otherwise.

## 13.4 What healthy looks like on this task

Approximate, for the default config at 16 environments:

| Steps | `ep_len_mean` | `feet_air_time` | `tracking_lin_err` | `curriculum/level` |
|---|---|---|---|---|
| 0–200k | rising from ~100 | ~0 | ~0.45 | 0 |
| 200k–1M | 400–1000 | becoming non-zero | 0.35–0.45 | 0 |
| 1M–4M | ~1000 | clearly positive | 0.2–0.35 | climbing |
| 4M–10M | ~1000 | positive and stable | 0.1–0.2 | approaching 1.0 |

`explained_variance` should be above 0.8 from a few hundred thousand steps
onward. `std` should drift from 0.37 down toward 0.2.

## 13.5 A worked diagnosis

This happened during development of v2 and is a complete example of the process.

**Symptom.** At 524k steps everything looked fine: `explained_variance` 0.91,
`ep_len_mean` pinned at 1000 (never falling), `ep_rew_mean` climbing to 40,
`rew/track_lin_vel_xy` at 77% of its maximum.

Except `rew/feet_air_time` was exactly $-0.0000$.

**Step 1 — read the zero literally.** That term is credited only on touchdown.
Exactly zero means `feet_first_contact` is never 1, which means no foot ever
leaves the ground. The robot was standing and leaning.

**Step 2 — ask what standing pays.** Twenty lines of numpy, no training
required: sample commands from the configured ranges, set the robot's achieved
velocity to zero, and evaluate the tracking terms.

```
A ROBOT THAT DOES NOTHING, under the initial command ranges:
  mean track_lin_vel_xy term : 0.739 of its maximum
  mean tracking_score        : 0.790  (curriculum threshold is 0.85)
```

The policy's 0.77 was **barely above the do-nothing floor of 0.739**. It had
learned almost nothing, and the reward function agreed that was fine.

**Step 3 — ask whether the target behaviour is even reachable.** This is the
step that is easy to skip and that separates "the reward is wrong" from "the
robot cannot do it". Drive the environment with a hand-written open-loop trot —
no policy, no learning — and measure.

First attempt: 0.12 s of maximum air time, schedule match 0.61. Barely off the
ground.

A kinematic sweep said 0.25 rad of joint offset should retract a foot 5.9 cm,
which is plenty for a trot needing 3–5 cm. So why 0.12 s?

**Because the trunk is not fixed.** While two legs retract, the other two carry
the entire robot and sag further. Measured standing sag at $k_p=40$ is 2.8 cm on
four legs, and roughly double that on two. The body drops by almost exactly as
much as the swing foot rises, and the foot never clears.

Adding stance-leg extension to the script — the stance legs push down while the
swing legs retract, which is what a real gait does — changed everything:

| $k_p$ | scale | stance push | max air time | schedule match |
|---|---|---|---|---|
| 40 | 0.25 | no | 0.12 s | 0.61 |
| 40 | 0.25 | yes | 0.24 s | 0.61 |
| 40 | 0.35 | yes | 0.24 s | 0.74 |
| 55 | 0.30 | yes | 0.20 s | **0.78** |
| 80 | 0.35 | yes | 0.30 s | 0.84 |

**Step 4 — conclude.** Two independent problems, both real:

1. *Reward geometry.* Standing collected 74% of the tracking reward. Fixed by
   widening the initial command ranges (0.739 → 0.488 do-nothing floor),
   sharpening $\sigma_v$ from 0.25 to 0.20, and tripling the `gait_phase`
   weight, which is the only dense term that pays for lifting a foot.
2. *Physical headroom.* Foot clearance at $k_p=40$, scale 0.25 was marginal — the
   policy would have had to discover simultaneous stance extension and swing
   retraction just to get off the ground. Raising to $k_p=55$, scale 0.30 gives
   the same coordination much more room. $k_p=80$ is better still on paper and
   was rejected: that stiff, the policy starts exploiting contact impulses no
   real actuator produces.

**The transferable part** is step 3. When a policy will not learn a behaviour,
script the behaviour open-loop and measure whether the environment can express
it at all. If a hand-written controller cannot do it, no amount of reward tuning
will help, and you have converted an open-ended search over hyperparameters into
a bounded question about kinematics.

## 13.5b A second diagnosis: a reward with no gradient

Both fixes above were applied, and the next run still stalled — 1.1 million
steps, `rew/feet_air_time` still zero, `rew/gait_phase` at 55% of its maximum.

55% is the tell. A robot with all four feet planted, under a trot schedule with
duty 0.5, matches the two feet that are supposed to be in stance and mismatches
the two that are supposed to be swinging: exactly 0.5. The policy was still
standing, and this time the reward geometry was not the reason.

**The reward was not wrong. It was flat.**

`gait_phase` is built on a **binary** contact flag. Measured on the standing
robot, the foot geoms sit at 9.5 mm. Raising a foot to 9.4 mm does not change
$c_i$, so it does not change the reward. The gradient is exactly zero right up
to a discontinuity — and that discontinuity is the barrier a policy that has
never stepped has to cross.

| swing foot raised by | `gait_phase` | `feet_clearance` |
|---|---|---|
| 0.0 cm | 0.50 | $-0.00333$ |
| 0.5 cm | 0.75 | $-0.00278$ |
| 2.0 cm | 0.75 | $-0.00142$ |
| 8.0 cm | 0.75 | $\phantom{-}0.00000$ |

The fix was a new term, `feet_clearance`: a smooth quadratic cost on the height
of each *swing* foot, which pulls the foot upward from the first millimetre.

The general rule, and the reason this is worth a section of its own:

> **A reward term defined on a thresholded quantity — a contact flag, a success
> indicator, any boolean — is piecewise constant. A policy cannot climb a
> staircase whose slope it cannot see.** Whenever a term is built on a boolean,
> ask what continuous quantity underlies it, and reward that as well.

The diagnostic that catches this class of bug in general: for each reward term,
ask what its **derivative with respect to the behaviour you want** is, in the
region the policy currently occupies. If the answer is zero, that term cannot
teach the behaviour no matter how heavily it is weighted.

## 13.5c A third diagnosis: score the target behaviour directly

Two fixes in, the run still would not step. At that point the useful move is not
another reward tweak — it is to stop reasoning about gradients and **measure what
the reward actually pays for the behaviour you want**.

That is a twenty-line experiment. Drive the environment with a hand-scripted
trot, drive it again with zero action, and print the per-term reward difference:

| term | standing | scripted trot | diff |
|---|---|---|---|
| `gait_phase` | $+0.01500$ | $+0.02343$ | $+0.00843$ |
| `feet_clearance` | $-0.00601$ | $-0.00314$ | $+0.00288$ |
| `feet_air_time` | $\phantom{+}0.00000$ | $+0.00056$ | $+0.00056$ |
| `torques` | $-0.00074$ | $-0.00292$ | $-0.00218$ |
| `ang_vel_xy` | $-0.00000$ | $-0.00147$ | $-0.00147$ |

Before the fix, the `feet_air_time` row read $-0.08$ per step. The term's 0.5 s
offset — borrowed from legged_gym, where the gait frequency is emergent — is
larger than the scheduled swing time of 0.25 s at the commanded 2 Hz, so it was
negative for **every correctly-timed step**. A perfect trot scored roughly four
times worse than standing still (chapter 14, B19).

The general procedure, and the one worth internalising:

> **Score the target behaviour under your reward and compare it against the
> trivial policy.** If the thing you want does not out-score doing nothing, the
> problem is not exploration, not the learning rate, and not the algorithm. No
> optimiser will find a behaviour your objective penalises.

It takes minutes, it needs no training, and it would have caught all three of
the stalls in this repository's history.

## 13.6 Sanity checks before you start

`scripts/check_env.py` runs all of these in about ten seconds:

- nominal pose matches the model's `home` keyframe (bug B1)
- the robot stands still under zero action without falling or oscillating
- the gait schedule alternates as expected
- 1000 random actions produce no NaN or infinite reward
- every configured reward term appears in the info dict
- the reward decomposition is printed, so you can see which term is loudest
  *before* spending six hours finding out

The last is worth more than it sounds. If a term you meant to be a minor
regulariser is 40% of the gross reward, you will discover it in ten seconds
rather than in a training curve.

---

**Previous:** [12. Curriculum and domain randomisation](12-curriculum-and-domain-randomization.md) ·
**Next:** [14. The debugging log](14-debugging-log.md)
