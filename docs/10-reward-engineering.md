# 10. Reward engineering

This is the chapter that matters most in practice. The algorithm is the same one
everyone uses; the reward is what makes this robot walk rather than some other
robot doing something else.

The governing principle: **the agent optimises what you wrote, not what you
meant.** Every gap between the two will be found and exploited, because finding
gaps in an objective function is precisely what an optimiser does.

## 10.1 Conventions

Three, and all three exist because of something that went wrong without them.

**Sign lives in the config, not the function.** A function named like a penalty
returns a non-negative cost; its configured weight is negative. So the sign of a
weight in `configs/training_config.yaml` always tells you whether the term is a
carrot or a stick, with no need to read the code.

**Weights are per second.** The environment computes
$r_t = \Delta t \sum_i w_i f_i(s_t)$. Changing the control rate therefore does
not silently rescale the objective. Without this, v2's move from 25 Hz to 50 Hz
would have halved every per-step reward and doubled the episode's step count,
leaving the total return unchanged but the *per-step* reward — which is what the
critic predicts — different by a factor of two.

**Every term is a separate pure function.** `envs/rewards.py` defines each as a
function of a `RewardState` snapshot, registered in `REWARD_TERMS`. This buys
three things: ablation is a config edit, per-term logging is automatic, and each
term is unit-testable in isolation at a hand-constructed state. v1 had all eight
terms inline in one 40-line method with the weights hard-coded mid-expression,
which is why a broken term survived five documented training runs.

## 10.2 The terms

### Task terms

**`track_lin_vel_xy`** — weight $+1.5$

$$
r = \exp\!\left( -\frac{\| \mathbf{v}^*_{xy} - \mathbf{v}_{xy} \|^2}{\sigma_v} \right), \qquad \sigma_v = 0.20
\tag{10.1}
$$

The main objective. Both axes, base frame. Bounded in $(0, 1]$.

**`track_ang_vel_yaw`** — weight $+0.75$

$$
r = \exp\!\left( -\frac{(\omega_z^* - \omega_z)^2}{\sigma_\omega} \right), \qquad \sigma_\omega = 0.25
\tag{10.2}
$$

Half the weight of linear tracking, which reflects that turning is the easier
half of the problem and should not dominate.

**`gait_phase`** — weight $+1.5$

$$
r = \frac{1}{4}\sum_{i=1}^{4} \Big[ c_i d_i + (1 - c_i)(1 - d_i) \Big]
\tag{10.3}
$$

with $c_i$ the actual contact and $d_i$ the scheduled one (chapter 11). "Fraction
of feet doing the right thing", in $[0,1]$.

**`feet_clearance`** — weight $-30.0$

$$
r = \sum_{i=1}^{4} (1 - d_i)\,(h_i - h^*)^2, \qquad h^* = 0.08\ \text{m}
\tag{10.3b}
$$

A cost on how far each *swing* foot is from a target height. §10.4b explains why
this term, rather than `gait_phase`, is what makes stepping discoverable.

**`feet_air_time`** — weight $+1.0$

$$
r = \mathbb{1}\big[\|\mathbf{c}\| > 0.1\big] \sum_{i=1}^{4} \min\!\big(t^{\text{air}}_i,\ t_{\text{swing}}\big) \, \mathbb{1}[\text{foot } i \text{ just landed}]
\tag{10.4}
$$

with $t_{\text{swing}} = (1 - \beta)/f$ taken from the active gait command.
Credited only on touchdown; non-negative; saturating at the scheduled swing
duration. Gated on a non-zero command, or it bribes a standing robot to fidget.

The obvious formulation — $\sum_i (t^{\text{air}}_i - 0.5)$, which is what
legged_gym uses — is **actively wrong here**, and §10.4c works through why.

**`alive`** — weight $+0.25$

A constant. Small on purpose; §10.5 explains why v1's was four times too big.

### Regularisation terms

| Term | Weight | Formula | Purpose |
|---|---|---|---|
| `lin_vel_z` | $-2.0$ | $v_z^2$ | stop the trunk bouncing |
| `ang_vel_xy` | $-0.05$ | $\|\omega_{xy}\|^2$ | stop the body wobbling |
| `orientation` | $-1.0$ | $\|g_{b,xy}\|^2$ | stay level |
| `base_height` | $-5.0$ | $(h - h^*)^2$ | hold the commanded ride height |
| `torques` | $-2\times10^{-4}$ | $\sum \tau_i^2$ | energy |
| `joint_acceleration` | $-2.5\times10^{-7}$ | $\sum ((\dot q_i - \dot q_i^{-})/\Delta t)^2$ | no judder |
| `action_rate` | $-0.01$ | $\|a - a^{-}\|^2$ | smooth commands |
| `joint_limits` | $-10.0$ | out-of-soft-range excess | do not lean on the endstops |
| `collision` | $-1.0$ | count of non-foot ground contacts | knees off the floor |
| `feet_slip` | $-0.05$ | $\sum_i c_i \|\mathbf{v}^{\text{foot}}_{i,xy}\|^2$ | no skating |
| `stand_still` | $-0.5$ | $\mathbb{1}[\|\mathbf{c}\| < 0.1] \sum_i \|q_i - q_{\text{nom},i}\|$ | actually stop when told |

Three of these deserve a note.

**`joint_acceleration`'s weight looks absurdly small** — $2.5\times10^{-7}$ —
because the quantity is enormous. At 50 Hz a 1 rad/s change in one step is
50 rad/s², squared is 2500, summed over twelve joints is $3\times10^4$. The
product lands at the same order as the other terms. Whenever a weight looks
wrong by orders of magnitude, check the units of the thing it multiplies.

**`joint_limits` uses a *soft* limit**, 5% inside the hard one. MuJoCo enforces
the hard limit for us, so without this term the policy learns a gait that leans
on the endstop — free force, in simulation, from a mechanical stop that on real
hardware is a rubber bumper and a warranty claim.

**`feet_slip` is gated on contact.** A swinging foot is *supposed* to move. Only
a loaded foot that is moving is skating.

## 10.3 Exponential kernels versus quadratic costs

Why is velocity tracking $\exp(-e^2/\sigma)$ and not $-e^2$?

**A quadratic cost is unbounded below.** Early in training $e$ is large — the
robot is on its side — so $-e^2$ dominates every other term. And the fastest way
to reduce it is to reduce $\|v\|$, i.e. to stop moving. A negative-quadratic
tracking reward actively teaches a beginner policy to freeze.

**The exponential is bounded and its gradient is where you want it.**

$$
\frac{\mathrm{d}}{\mathrm{d}e}\exp(-e^2/\sigma) = -\frac{2e}{\sigma}\exp(-e^2/\sigma)
$$

This is near zero for large $e$ (nothing to gain by flailing), maximal around
$e = \sqrt{\sigma/2}$, and near zero again at $e = 0$. The learning pressure is
concentrated exactly where the policy is close but not yet right.

**But it saturates**, and that is a real cost. At $e = 1.5$ m/s the gradient is
$\approx 4\times10^{-5}$: essentially flat. A policy that starts very far from
the command gets almost no signal. Which brings us to the failure this
repository actually hit.

## 10.4 A worked failure: the standing local optimum

This happened during the v2 run, not v1, and it is the best illustration in the
book of why reward *geometry* matters as much as reward *terms*.

At 524,000 steps the training log looked healthy: `explained_variance` 0.91,
episode length pinned at the 1000-step limit (never falling), episode return
climbing to ~40. `track_lin_vel_xy` was at 77% of its maximum.

And `feet_air_time` was exactly $0.0000$.

That number can only be zero if `feet_first_contact` is never 1 — if no foot
ever touches down, because no foot ever leaves the ground. The robot had learned
to stand, lean, and shuffle its weight, and to collect 77% of the velocity
reward while doing it.

The arithmetic explains why. Under the initial command ranges
($v_x \in [-0.3, 0.6]$, $v_y \in [-0.2,0.2]$, $\omega_z \in [-0.4,0.4]$, 10%
explicit stands) the typical commanded speed is about 0.3 m/s. A **perfectly
stationary** robot therefore has $e \approx 0.3$ and collects

$$
\exp(-0.3^2/0.25) = 0.70
$$

Monte-Carlo over the actual command distribution gives **0.739 of the maximum
tracking reward, for doing nothing at all.** The policy's 0.77 was barely above
that floor.

So the situation was: standing pays 74%, walking pays 100%, and getting from one
to the other requires crossing a region where the robot falls over and collects
the `collision` penalty and an early termination. A 26% upside does not buy
that.

**Three changes, each attacking a different part of the geometry.**

1. **Widen the initial command range** to $v_x \in [-0.5, 1.0]$,
   $v_y \in [-0.3,0.3]$, $\omega_z \in [-0.6,0.6]$. Larger commands mean larger
   $e$ for a stationary robot, so the do-nothing floor drops from 0.739 to
   0.488. This is counter-intuitive — the curriculum's *starting* point was made
   harder — but the curriculum's job is to make the task **learnable**, not
   easy, and a task where standing nearly wins is not learnable.

2. **Sharpen the kernel**, $\sigma_v: 0.25 \to 0.20$. Directly reduces what a
   near-miss earns.

3. **Raise `gait_phase` from 0.5 to 1.5.** This is the important one. Consider
   what signal each term provides *to a robot that has never stepped*:

   - `track_lin_vel_xy` rewards the *result* of walking. Undiscoverable until
     you are already walking.
   - `feet_air_time` pays at touchdown. Exactly zero if you have never lifted a
     foot. It cannot bootstrap; it can only refine.
   - `gait_phase` pays, densely and every single step, for having the right feet
     on the ground right now. A stationary robot scores 0.5 under a trot
     schedule; lifting one foot at the right moment immediately scores 0.75.

   The phase reward is the only term that provides a gradient *toward* stepping
   rather than a reward *for* stepping. That is precisely why it exists, and
   under-weighting it wasted half a million steps.

Measured effect of the change, on a random-action rollout: `gait_phase` went
from 14% to 33% of the gross reward, becoming the largest single term. The
do-nothing tracking score fell from 0.79 to 0.57 against an unchanged promotion
threshold of 0.85.

The general lesson: **compute what a do-nothing policy scores under your reward,
before you train.** It takes twenty lines of numpy and it tells you whether
there is a gradient out of the trivial solution. This is now a section of
`scripts/check_env.py`.

## 10.4b The second stall: a reward with no gradient

Fixing §10.4 was not enough. A second run reached 1.1 million steps with
`rew/feet_air_time` still at zero and `rew/gait_phase` at 55% of its maximum —
which is exactly what a fully-planted robot scores under a 50% duty schedule.
The policy was still standing.

The diagnosis is a nice one, because the reward was not *wrong* this time. It
was **flat**.

`gait_phase` compares a **binary** contact flag against the schedule:

$$
r = \tfrac14 \sum_i \big[ c_i d_i + (1-c_i)(1-d_i) \big]
$$

$c_i$ is 1 or 0. Nothing in between. Measured on the standing robot, the foot
geoms sit at 9.5 mm. Raising a foot from 9.5 mm to 9.4 mm changes $c_i$ not at
all, so it changes the reward not at all. The gradient is **exactly zero** all
the way up to a discontinuity — and that discontinuity is precisely the barrier a
policy that has never stepped has to cross.

| swing foot raised by | `gait_phase` | `feet_clearance` |
|---|---|---|
| 0.0 cm | 0.50 | $-0.00333$ |
| 0.5 cm | 0.75 | $-0.00278$ |
| 1.0 cm | 0.75 | $-0.00228$ |
| 2.0 cm | 0.75 | $-0.00142$ |
| 4.0 cm | 0.75 | $-0.00031$ |
| 8.0 cm | 0.75 | $\phantom{-}0.00000$ |

The left column is a step function. The right column is smooth, and pulls the
swing foot upward from the first millimetre.

So `gait_phase` and `feet_clearance` do different jobs, and both are needed:

- `feet_clearance` gets the foot **off the ground** — a dense, continuous
  gradient through the region where nothing else provides one.
- `gait_phase` gets it off the ground **at the right time** — it is what makes
  the motion a *gait* rather than four independent legs waving.
- `feet_air_time` makes the resulting strides **long** rather than a shuffle.

This generalises past this repository. **A reward term built on a thresholded
quantity — a contact flag, a success indicator, a boolean — is piecewise
constant, and a policy cannot climb a staircase it cannot see the slope of.**
Whenever a term is defined on a boolean, ask what continuous quantity underlies
it, and reward that too.

## 10.4c The third stall: a borrowed constant that inverted the sign

Two stalls down, the run still would not step. This time the culprit was
`feet_air_time`, and it is the most transferable of the three because the bug
was a **borrowed convention that stopped being true**.

The legged_gym form pays for air time beyond half a second:

$$
r = \sum_i \big(t^{\text{air}}_i - 0.5\big)\,\mathbb{1}[\text{foot } i \text{ just landed}]
$$

That is sensible where the gait frequency is *emergent* and the resulting
strides are long. It is wrong the moment frequency becomes a **commanded
input**. At the commanded 2 Hz with duty 0.5,

$$
t_{\text{swing}} = \frac{1-\beta}{f} = \frac{0.5}{2} = 0.25\ \text{s}
$$

so every touchdown scores $(0.25 - 0.5) \times 2.0 = -0.5$. Four feet at two
touchdowns per second is $-4.0$ per second, or $-0.08$ per control step — against
a total of $+0.03$ per step for standing perfectly still.

**A flawless trot scored roughly four times worse than doing nothing.** The
policy had not failed to find the gait. It had found it, evaluated it, and
correctly rejected it.

The fix credits air time *up to* the scheduled swing duration rather than
offsetting it by a constant. Three properties follow, all of which the old form
lacked:

- **non-negative**, so early imperfect steps are never punished — and those are
  the only route to good ones;
- **saturating**, so hanging a foot in the air earns nothing extra and cannot
  fight the phase schedule;
- **frequency-independent**, with a maximum rate of $4(1-\beta)$ per second
  whatever $f$ is, so commanding a faster gait does not inflate the reward.

Measured afterwards — scripted trot minus standing, per control step:
`gait_phase` $+0.0084$, `feet_clearance` $+0.0029$, `feet_air_time` $+0.0006$,
against `torques` $-0.0022$, `ang_vel_xy` $-0.0015$, `feet_slip` $-0.0009$.
Stepping now pays about $+0.007$ net *before* any credit for moving, and
tracking 0.8 m/s adds a further $+0.029$.

**The lesson: a borrowed hyperparameter carries the assumptions of the codebase
it came from.** The 0.5 s constant encodes "strides are long because frequency
is emergent"; import it into an environment where frequency is commanded and it
silently inverts the sign of the term for the exact behaviour you want.

And the check that catches it, which is cheap and general: **evaluate your
reward on a hand-scripted version of the target behaviour, and compare it
against the trivial policy.** If the thing you want does not out-score doing
nothing, no algorithm will find it, and you will spend days blaming exploration.

## 10.5 v1's reward, and what it actually optimised

| v1 term | Weight | Verdict |
|---|---|---|
| $2.0 \exp(-(v_x^* - v_x)^2/0.25)$ | 2.0 | world frame (B2); ignores $v_y^*$ (B3) |
| $0.5 \exp(-(\omega_z^* - \omega_z)^2/0.25)$ | 0.5 | world frame |
| alive | 0.5 | **too large** |
| $-0.5\,\lvert v_y \rvert$ | | punishes strafing unconditionally (B3) |
| $-1.0\,\|g_{xy}\|^2$ | | correct |
| $-0.01\,\|a - a^-\|^2$ | | correct |
| $-10^{-4}\sum\tau^2$ | | correct |
| $+0.1\,\lvert c_{FL}c_{RR} - c_{FR}c_{RL}\rvert$ | | **broken** (§10.6) |

Add up what a **motionless upright** v1 robot collected per step:

$$
\underbrace{2.0 \times 0.37}_{\text{tracking } (e = 0.5)} + \underbrace{0.5 \times 1.0}_{\text{yaw, } \omega = 0} + \underbrace{0.5}_{\text{alive}} + \underbrace{0.1 \times 1.0}_{\text{gait, standing on one diagonal}} = 1.84
$$

against a theoretical maximum of $2.0 + 0.5 + 0.5 + 0.1 = 3.1$. **59% of the
maximum reward, for standing still, with zero risk of the penalties that walking
incurs.**

v1's own `logs/algorithm_comparison.md` records the consequence without
identifying it: TD3 scored the *highest* mean reward (259) while achieving the
*lowest* forward speed (0.012 m/s). It had found the standing optimum and the
reward function agreed it was the best available policy. That table is a
measurement of the reward function, not of TD3.

## 10.6 The gait reward that a statue maximised

v1's eighth term:

```python
def _gait_reward(self):
    contacts = self._get_foot_contacts()
    diag1 = contacts[0] * contacts[3]   # FL & RR
    diag2 = contacts[1] * contacts[2]   # FR & RL
    return abs(diag1 - diag2)
```

Intent: reward a trot, in which diagonal pairs alternate.

What it does: returns 1.0 whenever exactly one diagonal pair is in contact.

Consider a robot standing motionless on FL and RR, feet planted, not moving at
all, forever. $\text{diag1} = 1$, $\text{diag2} = 0$, reward $= |1 - 0| = 1.0$.
**The maximum. Every step. Forever.**

The expression has **no time dependence whatsoever**. It cannot distinguish
alternation from a frozen asymmetric stance, because it only ever sees one
instant. "Alternating" is a statement about a sequence, and no function of a
single contact vector can express it.

This is the textbook shape of a reward-hacking bug, and it is instructive that
the fix is not a cleverer contact expression. No function of $\mathbf{c}$ alone
will do. You need a **reference the contact can be compared against** — a clock.
That is chapter 11.

Pinned as a test, so the point survives:

```python
def test_v1_gait_reward_was_maximised_by_standing_still():
    frozen = np.array([1.0, 0.0, 0.0, 1.0])
    assert v1_gait_reward(frozen) == 1.0            # the maximum

    # Against the v2 schedule, the same frozen robot averages 0.5:
    assert np.mean(scores) == pytest.approx(0.5, abs=0.05)
```

## 10.7 Reward hacking, generally

Every exploit in this repository fits one of three patterns.

**A reward for a proxy rather than the thing.** The gait reward proxied
"alternation" with "asymmetry". Asymmetry is cheaper than alternation, so
asymmetry is what you get.

**A reward whose trivial solution is nearly as good as the real one.** The
standing optimum. Whenever the do-nothing policy scores most of the maximum,
expect the do-nothing policy.

**A penalty that conflicts with a command.** v1 penalised all lateral velocity
while (nominally) accepting a lateral velocity command. The policy cannot
satisfy both, so it satisfies the one with the larger weight and the command
becomes decorative.

Four defences, all cheap:

1. **Compute the do-nothing baseline analytically** before training.
2. **Log every term separately.** This is what `RewardTermLoggerCallback` is for.
   The 524k-step failure was diagnosable in thirty seconds because
   `rew/feet_air_time` was plotted on its own; in v1's aggregate reward it would
   have been invisible.
3. **Unit-test each term's sign and bounds** at a hand-constructed state.
4. **Watch the robot.** A reward of 40 tells you nothing about whether the gait
   looks like a gait. `scripts/play.py` exists partly for this.

## 10.8 Tuning, in order

If you are changing weights, change them in this order.

1. **Get it to survive.** Termination penalties, `orientation`, `base_height`.
   Until episodes last more than a couple of seconds, nothing else matters.
2. **Get it to step.** `gait_phase`. Watch `rew/feet_air_time` become non-zero —
   that is the signal that the robot has genuinely left the ground.
3. **Get it to track.** `track_lin_vel_xy`, `track_ang_vel_yaw`, and the
   curriculum. Watch `diagnostics/tracking_lin_err`.
4. **Make it look right.** `action_rate`, `joint_acceleration`, `feet_slip`,
   `torques`. These cost performance and buy realism; tune them last, and only
   as far as you need.

Change one weight at a time, by a factor of two or three, not by 10%. RL runs
are noisy enough that a 10% change is unmeasurable without many seeds.

## 10.9 Summary

The reward is where the domain knowledge lives. PPO is off-the-shelf; the
sixteen terms above are the actual engineering.

Two numbers are worth computing before every run, and neither requires training
anything:

- What does a **do-nothing** policy score? (If it is most of the maximum, stop.)
- What does a **perfect** policy score? (If your terms cannot sum to much more
  than the do-nothing baseline, stop.)

---

**Previous:** [9. Observations and actions](09-observations-and-actions.md) ·
**Next:** [11. Gaits, phase, and periodic control](11-gaits-and-phase.md)
