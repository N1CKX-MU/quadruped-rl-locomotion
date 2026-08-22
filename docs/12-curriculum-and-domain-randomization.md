# 12. Curriculum and domain randomisation

Two mechanisms that both change the distribution of episodes the policy sees,
for opposite reasons. A curriculum narrows the task so it can be learned at all;
domain randomisation widens the world so what is learned generalises.

## 12.1 Why a curriculum

Sample the full command envelope from step one — $v_x$ up to 1.5 m/s, $v_y$ to
0.7, yaw to 1.5 rad/s — and an untrained policy falls over in under a second on
almost every episode. Nearly all of the data is about falling, the tracking
reward is saturated at its floor for every command (chapter 10, §10.3: the
exponential's gradient at $e = 1.5$ is $\approx 4\times10^{-5}$), and there is
no signal distinguishing a slightly better fall from a slightly worse one.

Start with commands the policy can nearly satisfy and the tracking reward is in
its informative region, where the gradient is largest.

## 12.2 Why v1's curriculum did not work

```python
def _on_step(self):
    progress = min(1.0, self.num_timesteps / self.warmup_steps)
    current_vel = self.start_vel + (self.max_vel - self.start_vel) * progress
    for env_idx in range(self.training_env.num_envs):
        self.training_env.env_method("set_cmd_vel", (current_vel, 0.0, 0.0),
                                     indices=[env_idx])
```

Three separate problems, one of them not about curricula at all.

**It was open loop.** The ramp is a function of wall-clock timesteps and nothing
else. If the policy was struggling, the curriculum widened anyway. If it was
ahead of schedule, it was held back. A curriculum that does not measure the
student is a timer.

**A scalar cannot describe a 3-D command space.** It ramped forward speed only.
$v_y$ and $\omega_z$ were fixed at zero for the entire run, which is a large part
of why v1 could only walk forwards — the *curriculum itself* guaranteed the
policy never saw a lateral or turning command.

**It performed 8 blocking IPC round trips per environment step.** This is the
one that is not a curriculum bug but cost the most. Under `SubprocVecEnv`, each
`env_method` call with `indices=[i]` pickles the arguments, writes to a pipe,
and **blocks** waiting for the worker's acknowledgement. The loop did that once
per environment, on every single step, to send a value that had not changed
since the previous step. At 8 environments and 25 Hz that is 200 blocking round
trips per simulated second, per environment — pure overhead against a physics
step that costs about 1 ms.

## 12.3 The v2 curriculum

Closed loop, over the ranges rather than a scalar, and touching the workers at
most once per rollout.

**The signal.** Each environment reports a normalised tracking score in $[0,1]$
in its info dict:

$$
\text{score} = \tfrac12 \exp\!\left(-\frac{e_\text{lin}^2}{\sigma_v}\right) + \tfrac12 \exp\!\left(-\frac{e_\text{ang}^2}{\sigma_\omega}\right)
\tag{12.1}
$$

**The controller.** A scalar level $\ell \in [0,1]$, updated once per rollout:

$$
\ell \leftarrow
\begin{cases}
\min(1, \ell + \Delta) & \text{score} \ge 0.85 \\
\max(0, \ell - \Delta) & \text{score} < 0.55 \\
\ell & \text{otherwise}
\end{cases}
\tag{12.2}
$$

with $\Delta = 0.05$. The dead band between 0.55 and 0.85 stops the level
oscillating on rollout-to-rollout noise.

**The ranges.** $\ell$ linearly interpolates every range from `initial` to
`final`:

$$
[\,\text{lo}, \text{hi}\,]_\ell = (1-\ell)\,[\,\text{lo}, \text{hi}\,]_\text{init} + \ell\,[\,\text{lo}, \text{hi}\,]_\text{final}
$$

| | initial ($\ell=0$) | final ($\ell=1$) |
|---|---|---|
| $v_x$ (m/s) | $[-0.5, 1.0]$ | $[-1.0, 1.5]$ |
| $v_y$ (m/s) | $[-0.3, 0.3]$ | $[-0.7, 0.7]$ |
| $\omega_z$ (rad/s) | $[-0.6, 0.6]$ | $[-1.5, 1.5]$ |
| $f$ (Hz) | $[1.8, 2.2]$ | $[1.5, 3.0]$ |
| $h^*$ (m) | $[0.29, 0.31]$ | $[0.27, 0.34]$ |

**The IPC.** In-process accumulation on every step; a single broadcast
`env_method` call at rollout end, and only when the level actually changed:

```python
def _on_step(self):
    for info in self.locals.get("infos", ()):
        score = info.get("tracking_score")
        if score is not None:
            self._scores.append(score)      # no IPC
    return True

def _push_ranges(self, force=False):
    if not force and self._pushed_level == level:
        return
    self.training_env.env_method("set_command_ranges", self.curriculum.ranges)
```

One call per 8192 steps instead of eight calls per step: a factor of roughly
65,000 fewer round trips.

## 12.4 Setting the thresholds, and the trap in them

The promotion threshold cannot be picked by taste. It has to be compared against
what a **do-nothing policy** scores under the same command ranges, because a
threshold below that number promotes a robot that has learned nothing.

Monte-Carlo over the command sampler, with the robot's achieved velocity set to
zero:

| initial ranges | $\sigma_v$ | do-nothing tracking score |
|---|---|---|
| $[-0.3, 0.6],[-0.2,0.2],[-0.4,0.4]$ | 0.25 | **0.79** |
| $[-0.5, 1.0],[-0.3,0.3],[-0.6,0.6]$ | 0.25 | 0.62 |
| $[-0.5, 1.0],[-0.3,0.3],[-0.6,0.6]$ | 0.20 | **0.57** |
| $[-1.0, 1.5],[-0.7,0.7],[-1.5,1.5]$ | 0.20 | 0.30 |

The first row is what this repository originally shipped, against a promotion
threshold of 0.85. A stationary robot scored 0.79 — within 0.06 of promotion,
having never taken a step. And with a decay threshold of 0.65, standing was
*above* the demotion line too, so the curriculum considered a statue to be
making acceptable progress.

The fix was to make the starting task **harder**, not easier: widen the initial
ranges and sharpen the kernel until the do-nothing baseline is 0.57, leaving
real headroom to 0.85. This is the counter-intuitive part of curriculum design
and worth stating as a rule:

> A curriculum's job is to make the task **learnable**, not easy. If the trivial
> policy nearly satisfies the starting task, there is no gradient out of it, and
> the curriculum has created the local optimum it was supposed to avoid.

Chapter 10, §10.4 has the full incident.

## 12.5 Stand-still commands

10% of sampled commands are an explicit zero, and any command whose magnitude
falls below 0.15 is snapped to zero.

Without this, "stop" is a measure-zero event in a continuous range and is
effectively never sampled. The result is a robot that marches on the spot when
told to stop — a policy that has learned "commanded speed maps to stepping rate"
and has never been shown that zero is special.

The snapping matters too. In the ambiguous band around zero the `gait_phase`
term (using a stepping schedule) and the `stand_still` term (penalising
deviation from nominal) pull in opposite directions. Snapping to a true stand
switches the gait schedule to `stand`, so both terms agree.

## 12.6 Mid-episode resampling

Commands are resampled every 5 seconds within an episode, not just at reset.

Two reasons. First, a policy can infer a constant command from its own early
motion and stop reading the observation — it becomes open loop with a
20-second memory. Second, the transitions themselves need training: what the
robot does in the half-second after the command changes is exactly what you feel
when driving it with `scripts/play.py`, and it is not trained by a policy that
only ever sees step changes at reset.

## 12.7 Domain randomisation

The opposite operation. The curriculum narrows what is asked; randomisation
widens the world it is asked in.

The formal justification is chapter 1's POMDP caveat. Friction, payload and
actuator gains are hidden state. A policy trained at one value learns a control
law tuned to it. Randomise across episodes and the policy must find a law that
works for the whole distribution — which is a robust law, and one that has a
chance of working on a system whose true value is outside the training range.

What is randomised, per episode:

| Quantity | Range | Standing in for |
|---|---|---|
| ground and foot friction | $\times[0.4, 1.4]$ | surface, wear, dust |
| trunk mass | $+[-1.0, +2.0]$ kg | battery, sensors, payload |
| $k_p$ | $\times[0.8, 1.2]$ | motor constant, driver |
| $k_d$ | $\times[0.8, 1.2]$ | friction, unmodelled damping |
| initial joint angles | $\pm 0.1$ rad | — |
| initial height | $[-0.02, +0.05]$ m | — |
| initial roll/pitch | $\pm 0.05$ rad | — |
| initial yaw | $[-\pi, \pi)$ | heading invariance |
| initial base velocity | $\pm 0.3$ m/s, $\pm 0.3$ rad/s | — |
| initial joint velocity | $\pm 0.5$ rad/s | — |
| initial gait phase | $[0, 1)$ | — |

Two implementation points that are easy to get wrong.

**Randomise from the model defaults, never from the current value.** Otherwise
the randomisation compounds across episodes and drifts:

```python
self.model.body_mass[:] = self.default_body_mass          # restore first
self.model.body_mass[self.base_body_id] = (
    self.default_body_mass[self.base_body_id] + self.np_random.uniform(-1.0, 2.0))
```

**Reset the gains even when randomisation is off**, so that
`randomize_dynamics: false` really means nominal dynamics rather than "whatever
the last randomised episode left behind".

### Initial-state randomisation is the underrated one

v1 reset to a single state and perturbed joint angles by $\pm 0.05$ rad. A
policy trained that way has never seen the states it visits after a stumble, so
it cannot recover from one. Randomising the whole initial condition — height,
attitude, both velocities, phase — is cheap and is most of what "push recovery"
actually is.

Random initial **yaw** deserves its own mention: it is the second line of
defence against heading-correlated behaviour, complementing the choice of
projected gravity over the quaternion (chapter 9, §9.2).

Random initial **phase** stops every episode starting on the same foot, which
would let the policy key off the episode timer rather than the observed clock.

## 12.8 Pushes

```python
f = self.np_random.uniform(-40, 40, size=3)
f[2] *= 0.25                       # mostly horizontal
self.data.xfrc_applied[self.base_body_id, :3] = f
self._push_remaining = 0.15        # seconds
```

Every 3–7 seconds, a randomly-timed force of up to 40 N held for 0.15 s. Against
a 15.2 kg robot that is $\approx 0.4$ m/s of velocity change — a solid shove.

Contrast v1:

```python
if self.step_count > 0 and self.step_count % 200 == 0:
    push = self.np_random.uniform(-3.0, 3.0, size=3)
    self.data.qvel[0:3] += push
```

Two problems. It fired on an **exactly periodic schedule** the policy could
memorise — after enough episodes, "step 200" is a predictable event and the
policy braces for it rather than learning general recovery. And it wrote
directly into `qvel`, teleporting momentum: an instantaneous velocity change of
up to 3 m/s with no force, no impulse through the contacts, and no
correspondence to anything physical. A real push acts through the trunk over
some duration and the legs feel it through the ground.

Randomised timing means the policy cannot anticipate. A real force means the
recovery it learns is a recovery from something that can actually happen.

## 12.9 How much is too much

Randomisation is not free. Every dimension you randomise increases the variance
of the return, which increases the variance of the advantage estimates, which
slows learning. Over-randomise and the policy converges to the most conservative
behaviour that survives the worst case — a robot that walks slowly and stiffly
because it is braced for a friction coefficient of 0.4 at all times.

The heuristic: randomise a parameter if you do not know its true value to better
than the width of the range. Friction is genuinely unknown to within a factor of
two, so $\times[0.4, 1.4]$ is honest. Gravity is known to five significant
figures, so randomising it is noise for its own sake.

Since sim-to-real is out of scope for this version, the ranges here are modest —
enough to prevent brittleness, not enough to pay the full conservatism tax.
Chapter 16 lists what would need widening for real transfer.

## 12.10 What to watch

| Trace | Healthy | Unhealthy |
|---|---|---|
| `curriculum/level` | climbs steadily, reaches 1.0, stays | pinned at 0 (nothing is learning); oscillating (dead band too narrow) |
| `curriculum/tracking_score` | climbs past 0.85 | flat near the do-nothing baseline — check §12.4 |
| `curriculum/max_vel_x` | follows the level | — |
| `episode/mean_length` | rises toward the truncation limit | falling as the level rises: the curriculum is outrunning the policy |

The failure worth naming: level climbing while `episode/mean_length` falls. The
curriculum is promoting on a tracking score achieved in the brief interval
before the robot falls over. Lower the promotion threshold's dead band, or add
episode length as a second promotion criterion.

---

**Previous:** [11. Gaits, phase, and periodic control](11-gaits-and-phase.md) ·
**Next:** [13. Reading a training run](13-training-diagnostics.md)
