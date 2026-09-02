# 14. The debugging log

Twenty defects, with the reasoning that found each one and the evidence that
confirmed it. Sixteen were in the v1 environment and training script; the last
four (B17-B20) were found in v2 and are included because they were found the
same way and illustrate the method better than anything else here.

This is the most useful chapter in the book, for a reason that is worth stating
plainly: **none of these bugs produced an error message.** v1 trained. It
converged. It produced a robot that walked at 0.74 m/s, five documented
training runs, a GIF, and a README with sim-to-real recommendations. Every bug
below survived all of that.

The chapter is organised by impact. Each entry gives the symptom, the evidence,
the fix, and — where it generalises — the lesson.

A note on method. The bugs were not found by reading the code top to bottom.
They were found by asking, of each design decision, *what would have to be true
for this to be right?* and then checking. That question is what turns a code
review into a measurement.

---

## B1 — The nominal pose was not a pose the robot can hold

**Severity: this is the one that defined v1's ceiling.**

### The code

```python
mujoco.mj_resetData(self.model, self.data)
mujoco.mj_forward(self.model, self.data)
self.default_joint_pos = self.data.qpos[7:].copy()
```

`mj_resetData` sets `qpos = model.qpos0`. For the Menagerie Go2, `qpos0` has all
twelve joints at exactly 0 rad.

The model also ships a `home` keyframe — the actual standing pose — which was
never loaded:

```xml
<key name="home" qpos="0 0 0.27  1 0 0 0   0 0.9 -1.8   0 0.9 -1.8   0 0.9 -1.8   0 0.9 -1.8"/>
```

### The evidence

Every control step, v1 computed `target_pos = default_joint_pos + action * 0.5`,
so the PD controller's reachable target set per joint was $[-0.5, +0.5]$ rad.
Checking that against the model's own joint limits:

```
FL_hip_joint     qpos0=+0.000 limits=[-1.047,+1.047] OK
FL_thigh_joint   qpos0=+0.000 limits=[-1.571,+3.491] OK
FL_calf_joint    qpos0=+0.000 limits=[-2.723,-0.838] *** OUTSIDE LIMITS ***
FR_calf_joint    qpos0=+0.000 limits=[-2.723,-0.838] *** OUTSIDE LIMITS ***
RL_calf_joint    qpos0=+0.000 limits=[-2.723,-0.838] *** OUTSIDE LIMITS ***
RR_calf_joint    qpos0=+0.000 limits=[-2.723,-0.838] *** OUTSIDE LIMITS ***
```

Two facts follow, and both are worse than "the pose was suboptimal".

**The reachable target set and the joint's legal range did not intersect.**
$[-0.5, +0.5] \cap [-2.723, -0.838] = \varnothing$. For *every action the policy
could possibly emit*, the commanded knee angle was outside the knee's physical
travel — by at least 0.34 rad.

**Every episode began in an illegal configuration.** `qpos0` puts the calf
joints at 0, itself outside $[-2.723, -0.838]$. So every reset placed the robot
in a state violating its own joint limits, and MuJoCo's limit constraints spent
the first steps of every episode shoving the joints back while the PD controller
pulled the other way.

The Go2's calf range does not contain zero. There is no configuration of this
robot with a straight knee. v1 spent five training runs asking for one.

### The fix

```python
self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
if self._home_key_id < 0:
    raise RuntimeError("The MJCF has no 'home' keyframe. ...")
self.default_joint_pos = self.model.key_qpos[self._home_key_id, 7:].copy()
```

and `mj_resetDataKeyframe` instead of `mj_resetData` in `reset()`.

### Confirmed by test

`tests/test_env.py` asserts both directions — that the fix is right, and that
the bug was real:

```python
def test_nominal_pose_is_a_standing_pose(env):
    assert np.allclose(env.default_joint_pos, np.tile([0.0, 0.9, -1.8], 4))

def test_v1_nominal_pose_would_not_stand():
    env.default_joint_pos = np.zeros(12)      # v1's pose
    ...
    assert fell, "expected the all-zeros pose to collapse"
```

The second test passes: driving the same PD controller toward the all-zeros pose
collapses the robot within two seconds.

### Lesson

**When you load a robot model, print the nominal pose and check it against the
joint limits before anything else.** It is three lines. Nothing errors if you
skip it; the simulation runs, the policy trains, the curve goes up, and you
publish a number that is a fraction of what the robot can do.

`scripts/check_env.py` now does this as its first section.

---

## B2 — Velocity was tracked in the world frame

### The code

```python
base_lin_vel = self.data.qvel[0:3]                       # WORLD frame
lin_vel_error = (self.cmd_vel[0] - base_lin_vel[0]) ** 2
```

and identically in `_get_obs`.

For a free joint, `qvel[0:3]` is the linear velocity **in the fixed world
frame**. It is not the robot's forward velocity.

### Why it survived

v1 never randomised the initial heading and never commanded a yaw rate. The
robot always faced $+x$, where world-frame and base-frame velocity coincide. The
bug was invisible right up until someone asked for a turn — which is to say,
until v2.

The irony: v1 *had* a correct `_quat_rotate_inverse`, verified here against
MuJoCo's `mju_rotVecQuat`. It was used for gravity and never for velocity.

### The fix

Rotate everything into the base frame:

```python
def _refresh_frame_cache(self):
    q = self.data.qpos[3:7]
    self._lin_vel_b = self.quat_rotate_inverse(q, self.data.qvel[0:3])
    self._ang_vel_b = self.quat_rotate_inverse(q, self.data.qvel[3:6])
    self._proj_gravity = self.quat_rotate_inverse(q, self._GRAVITY_DIR)
```

### Lesson

A frame error is invisible in the symmetric case that your tests happen to
cover. Randomising the initial yaw is cheap and makes the class of bug
impossible to hide.

---

## B3 — The reward could only express "go forward"

Three separate defects that combine into "it only does a forward gait".

**Only $v_x$ was tracked.** `cmd_vel[1]` appeared nowhere in the reward.

**Lateral motion was punished unconditionally.** `r_lateral = -abs(v_y) * 0.5`.
So a hypothetical sideways command was not merely untracked — it was actively
penalised. The two objectives were in direct contradiction and the penalty won.

**The command was constant for the entire run.** A policy shown one command has
no reason to *condition* on it, so it collapses to an open-loop gait. The
command occupied three of the 53 observation dimensions and carried no
information.

And the curriculum (B5) ramped only forward speed, so $v_y$ and $\omega_z$ were
pinned at zero for every step of every run.

### The fix

`envs/commands.py`: three-dimensional commands, resampled at reset and every 5 s
within an episode, with an explicit stand-still mode. `envs/rewards.py`:

```python
err = xp.sum(xp.square(s.cmd[:2] - s.lin_vel_b[:2]))   # both axes
return xp.exp(-err / s.lin_vel_sigma)
```

### Lesson

"The robot only walks forward" was never a capability limit. The environment
never asked it to do anything else. Before concluding a policy *cannot* do
something, check that the reward makes it *expressible*.

---

## B4 — The gait reward had no notion of time

### The code

```python
def _gait_reward(self):
    contacts = self._get_foot_contacts()
    diag1 = contacts[0] * contacts[3]   # FL & RR
    diag2 = contacts[1] * contacts[2]   # FR & RL
    return abs(diag1 - diag2)
```

Intent: reward a trot, in which diagonal pairs alternate.

### The evidence

A robot standing motionless on FL and RR: $\text{diag1}=1$, $\text{diag2}=0$,
reward $=|1-0|=1.0$. **The maximum, every step, forever, without moving.**

The expression has no time dependence at all. It sees one instant, and
"alternating" is a property of a sequence. No function of a single contact
vector can express it.

### The fix

The fix is not a cleverer contact expression — none exists. It is to introduce a
**reference the contact can be compared against**: a phase clock (chapter 11).

$$
r = \tfrac14\textstyle\sum_i \big[ c_i d_i + (1-c_i)(1-d_i) \big]
$$

Now a frozen robot averages exactly 0.5, and there is no stationary
configuration that scores above it.

### Confirmed by test

```python
def test_v1_gait_reward_was_maximised_by_standing_still():
    frozen = np.array([1.0, 0.0, 0.0, 1.0])
    assert v1_gait_reward(frozen) == 1.0          # v1: the maximum
    assert np.mean(scores) == pytest.approx(0.5)  # v2: chance
```

### Lesson

The canonical shape of reward hacking: a **proxy** (contact asymmetry) stood in
for the **property** (alternation), and the proxy was cheaper to satisfy. Ask of
every reward term: what is the cheapest state that maximises this?

---

## B5 — The curriculum made 8 blocking IPC round trips per environment step

### The code

```python
def _on_step(self):
    progress = min(1.0, self.num_timesteps / self.warmup_steps)
    current_vel = self.start_vel + (self.max_vel - self.start_vel) * progress
    for env_idx in range(self.training_env.num_envs):
        self.training_env.env_method("set_cmd_vel", (current_vel, 0.0, 0.0),
                                     indices=[env_idx])
```

### The evidence

`_on_step` runs on **every environment step**. Under `SubprocVecEnv`, each
`env_method(..., indices=[i])` pickles its arguments, writes to a pipe, and
**blocks** for the worker's reply. The loop does that once per environment.

At 8 environments: 8 blocking round trips per step, to send a value that had not
changed since the previous step. Against a physics step costing about 1 ms, this
is pure overhead — and it is invisible in every trace except `time/fps`.

Two design bugs on top of the performance one: the ramp was **open loop** (a
function of timesteps, not of whether the policy could cope), and a **scalar
cannot describe a 3-D command space**.

### The fix

Accumulate in-process on every step; one broadcast call per rollout, and only
when the level actually changed:

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

One call per 8192 steps rather than eight per step: about 65,000× fewer round
trips. And the curriculum is now closed loop on measured tracking performance.

### Lesson

Callbacks that fire per step are on the hottest path in the system. Anything
that crosses a process boundary belongs in `_on_rollout_end`.

---

## B6 — The PD controller was a zero-order hold

### The code

```python
torques = self.kp * (target_pos - current_pos) - self.kd * current_vel
self.data.ctrl[:] = np.clip(torques, -ctrl_max, ctrl_max)
for _ in range(self.frame_skip):        # 20 steps = 40 ms
    mujoco.mj_step(self.model, self.data)
```

### The evidence

$\tau$ is constant for 40 ms. The $-k_d \dot q$ term — the entire damping
mechanism — is evaluated once from a velocity that becomes up to 40 ms stale.
During a foot strike, when joint velocities reverse in a few milliseconds, the
"damping" term can be actively *driving* the oscillation it exists to suppress.

This is not what a PD controller is. Real hardware runs the inner loop at
500–1000 Hz while the policy runs at 50.

### The fix

```python
for _ in range(self.decimation):
    torque = kp * (target_pos - self.data.qpos[7:]) - kd * self.data.qvel[6:]
    torque = np.clip(torque, -self.torque_limits, self.torque_limits)
    self.data.ctrl[:] = torque
    mujoco.mj_step(self.model, self.data)
```

### Note

v1's **gains** ($k_p=40$, $k_d=1$) were reasonable and were not the problem. It
is worth resisting the urge to change everything when a system underperforms;
change what the evidence implicates. (The gains were eventually raised to
$k_p=55$ for an unrelated, separately measured reason — B17.)

---

## B7 — Control ran at 25 Hz

`frame_skip=20` at a 2 ms timestep. The field standard for legged RL is 50 Hz.

A 2 Hz trot gives 25 control decisions per stride at 50 Hz and only 12 at 25 Hz.
Twelve decision points is not many for placing four feet.

Fixed by `decimation: 10`.

**The consequence that is easy to miss:** changing the control rate changes the
meaning of $\gamma$. At $\gamma = 0.99$ the effective horizon is 100 steps —
4 seconds at 25 Hz, 2 seconds at 50 Hz. Doubling the control rate *halved* the
planning horizon. Two seconds is the right number for locomotion, so this
happened to be fine, but it had to be checked rather than assumed.

---

## B8 — Reset produced a single initial condition

```python
mujoco.mj_resetData(self.model, self.data)
noise = self.np_random.uniform(-0.05, 0.05, size=self.n_actuators)
self.data.qpos[7:] = self.default_joint_pos + noise
```

Joint angles perturbed by $\pm 0.05$ rad; nothing else. No base height,
attitude, or velocity noise. No heading randomisation.

A policy trained from one initial state has never seen the states it visits
*after a stumble*, so it cannot recover from one. Randomising the full initial
condition is cheap and is most of what "push recovery" actually consists of.
Random initial yaw additionally makes B2-class frame bugs impossible to hide.

Fixed: height, roll/pitch, yaw (uniform on $[-\pi,\pi)$), both base velocities,
joint velocities, and the initial gait phase are all randomised.

---

## B9 — A new OpenGL context per rendered frame

```python
elif self.render_mode == "rgb_array":
    renderer = mujoco.Renderer(self.model, height=480, width=640)
    ...
    renderer.close()
```

Constructing and destroying a `mujoco.Renderer` — and with it a GL context —
on every single frame. Fixed by creating it lazily once and reusing it.

---

## B10 — Foot contact detection counted the wrong contacts

```python
for i in range(self.data.ncon):
    c = self.data.contact[i]
    for foot_idx, gid in enumerate(self.foot_geom_ids):
        if c.geom1 == gid or c.geom2 == gid:
            contacts[foot_idx] = 1.0
```

Two failures. It never checked the **other** geom, so foot-on-shin and
foot-on-foot self-contact counted as ground contact. And it used the mere
*existence* of a contact pair, so a foot grazing the ground scored the same as
one bearing a quarter of the robot's weight.

Fixed by requiring the counterpart to be the floor geom and the normal force to
exceed 1 N (against a 149 N robot):

```python
if g1 == self.floor_geom_id:   other = g2
elif g2 == self.floor_geom_id: other = g1
else:                          continue
mujoco.mj_contactForce(self.model, self.data, i, force)
if abs(force[0]) < force_threshold: continue
```

### The silent fallback, which was worse

```python
if len(self.foot_geom_ids) != 4:
    self.foot_geom_ids = list(range(self.model.ngeom - 4, self.model.ngeom))
```

"The last four geoms" is not a foot detector. Had the names ever failed to
resolve, the gait reward would have silently rewarded contacts on arbitrary
bodies, and nothing would have indicated it. v2 raises instead.

**Lesson:** a fallback that silently changes what is being optimised is worse
than a crash. Misconfiguration should stop the program.

---

## B11 — Pushes were deterministic and unphysical

```python
if self.step_count > 0 and self.step_count % 200 == 0:
    push = self.np_random.uniform(-3.0, 3.0, size=3)
    self.data.qvel[0:3] += push
```

**Exactly periodic**, so the policy could memorise "step 200" and brace rather
than learn general recovery. And it wrote directly into `qvel`, teleporting up
to 3 m/s of velocity with no force, no impulse through the contacts, and no
correspondence to anything that can physically happen.

Fixed: a real force through `xfrc_applied`, up to 40 N, held for 0.15 s, at
randomised intervals of 3–7 s, mostly horizontal.

---

## B12 — Resuming training silently destroyed the policy

**The nastiest bug in the original repository, because nothing errors.**

```python
env = SubprocVecEnv([...])
env = VecNormalize(env, norm_obs=True, ...)     # FRESH statistics
...
if args.resume:
    model = PPO.load(args.resume, env=env)
```

`VecNormalize` maintains running mean and variance of the observation. The
checkpoint's companion `.pkl` was saved by `CheckpointCallback` and **never
loaded**. So on resume the statistics reset to mean 0, variance 1, and the
policy — trained on observations normalised by statistics accumulated over
millions of steps — received inputs on a completely different scale.

Nothing raises. The run continues. The loss simply gets worse, and it looks like
the policy "forgot".

### The fix

```python
stats_path = args.vecnormalize or _guess_vecnormalize_path(args.resume)
if stats_path and os.path.exists(stats_path):
    env = VecNormalize.load(stats_path, env.venv)
    env.training = True
    model.set_env(env)
else:
    raise SystemExit("Could not find the VecNormalize statistics ...")
```

Note the `raise`. Resuming without the statistics is never what anyone wants, so
it should be impossible rather than merely inadvisable.

### Lesson

**Normalisation statistics are part of the model.** The from-scratch
implementation makes this structural — `PPO.save` writes `obs_rms` into the same
file as the weights, so they cannot be separated.

The same failure appears at inference time: `scripts/play.py` and
`scripts/evaluate.py` both load the statistics and warn loudly if they cannot
find them.

---

## B13 — The evaluation environment updated its own normalisation

```python
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
```

`training` defaults to `True`, so the eval environment kept updating its own
running statistics between evaluations. `EvalCallback` syncs the training env's
statistics across before each evaluation, so the effect is bounded — but it means
each evaluation measures a slightly different normalisation, adding noise to the
one curve that is supposed to be the clean measurement.

Fixed with `eval_env.training = False`.

---

## B14 — 2560 gradient steps per rollout

| | v1 | v2 |
|---|---|---|
| rollout | $8 \times 2048 = 16384$ | $16 \times 512 = 8192$ |
| `batch_size` | 64 | 2048 |
| `n_epochs` | 10 | 5 |
| **gradient steps per rollout** | **2560** | **20** |

PPO's entire premise (chapter 5) is that the policy stays near the one that
collected the data. No clip range keeps a policy near $\theta_\text{old}$ across
2560 updates. In practice the clip fraction saturates, most samples contribute
no gradient, and the compute is spent for nothing — a stability risk and a large
waste simultaneously.

Also fixed in the same pass: `ent_coef` 0.01 → 0.002 (v1's value pays the policy
to stay noisy, which for a legged robot means it never settles into the
low-variance limit cycle a trot requires), plus a linear LR schedule, a
`target_kl` early stop, value clipping, and `log_std_init = -1.0`.

---

## B15 — The observation docstring said 49; the space was 53

Cosmetic on its own. Worth recording because of what it indicates: the
observation had been edited without anyone re-deriving its size. Where the
documentation and the code disagree, one of them has been changed without
thought, and it is worth finding out which.

---

## B16 — `mujoco.viewer` was used without being imported

```python
import mujoco
...
self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
```

`mujoco.viewer` is a submodule and is not imported by `import mujoco`. This
would raise `AttributeError` on any call to `render("human")` — which suggests
that path had never been exercised, despite `make evaluate-render` existing.

---

## B17 — Swing clearance was marginal (found in v2, not v1)

Included because it was found the same way and is the best illustration of the
method.

**Symptom.** After the v2 rewrite, a 524k-step run had `rew/feet_air_time`
exactly zero: the robot was standing and leaning, never lifting a foot.

**The step that mattered.** Rather than tuning rewards further, drive the
environment with a **hand-written open-loop trot** and ask whether the behaviour
is expressible at all.

It was not, quite. A kinematic sweep said 0.25 rad of joint offset retracts a
foot 5.9 cm — plenty for a trot needing 3–5 cm. But the trunk is not fixed:
while two legs retract, the other two carry the whole robot and sag further, and
the body drops by nearly as much as the swing foot rises.

| $k_p$ | scale | stance extension | max air time | schedule match |
|---|---|---|---|---|
| 40 | 0.25 | no | 0.12 s | 0.61 |
| 40 | 0.25 | yes | 0.24 s | 0.61 |
| 55 | 0.30 | yes | 0.20 s | **0.78** |
| 80 | 0.35 | yes | 0.30 s | 0.84 |

Fixed by raising $k_p$ to 55 and the action scale to 0.30 — enough headroom for
the policy to extend the stance legs and retract the swing legs simultaneously.
$k_p = 80$ scores better still and was rejected: that stiff, the policy starts
exploiting contact impulses no real actuator produces.

**Lesson, and the most transferable thing in this chapter:** when a policy will
not learn a behaviour, **script the behaviour open-loop and measure whether the
environment can express it.** If a hand-written controller cannot do it, no
amount of reward tuning will. It converts an open-ended search over
hyperparameters into a bounded question about kinematics.

---

## B18 — The stepping reward had no gradient (found in v2)

With B17 fixed, the next run still stalled: 1.1 million steps,
`rew/feet_air_time` still exactly zero, `rew/gait_phase` at 55% of its maximum.

55% is the diagnostic number. A robot with all four feet planted, under a trot
schedule with duty 0.5, matches the two feet scheduled for stance and mismatches
the two scheduled for swing — exactly 0.5, every step. The policy was standing.

This time the reward was not *wrong*. It was **flat**.

`gait_phase` compares a **binary** contact flag against the schedule:

$$
r = \tfrac14 \sum_i \big[ c_i d_i + (1-c_i)(1-d_i) \big]
$$

$c_i \in \lbrace 0, 1\rbrace $; there is nothing in between. On the standing robot the foot
geoms sit at 9.5 mm. Raising a foot to 9.4 mm leaves $c_i$ unchanged and
therefore leaves the reward unchanged. **The gradient is exactly zero all the
way up to a discontinuity** — and that discontinuity is precisely the barrier a
policy that has never stepped must cross. Weighting the term more heavily does
not help: three times zero is still zero.

| swing foot raised by | `gait_phase` | `feet_clearance` |
|---|---|---|
| 0.0 cm | 0.50 | $-0.00333$ |
| 0.5 cm | 0.75 | $-0.00278$ |
| 1.0 cm | 0.75 | $-0.00228$ |
| 2.0 cm | 0.75 | $-0.00142$ |
| 4.0 cm | 0.75 | $-0.00031$ |
| 8.0 cm | 0.75 | $\phantom{-}0.00000$ |

### The fix

A new term, `feet_clearance`, weight $-30$:

$$
r = \sum_{i=1}^{4} (1 - d_i)(h_i - h^*)^2, \qquad h^* = 0.08\ \text{m}
$$

Foot height is continuous, so this pulls the swing foot upward from the first
millimetre. It is gated by $(1 - d_i)$, so it applies only to feet the schedule
wants in the air and vanishes entirely under a stand command.

The three stepping terms now decompose cleanly:

- `feet_clearance` gets a foot **off the ground** (dense, continuous).
- `gait_phase` gets it off the ground **at the right time** (the timing).
- `feet_air_time` makes the resulting stride **a real one** (credited at
  touchdown).

### Lesson

> **A reward term defined on a thresholded quantity — a contact flag, a success
> indicator, any boolean — is piecewise constant. A policy cannot climb a
> staircase whose slope it cannot see.**

The general diagnostic: for each reward term, ask what its derivative with
respect to the behaviour you want is, *in the region the policy currently
occupies*. If it is zero, that term cannot teach the behaviour at any weight.
Whenever a term is built on a boolean, find the continuous quantity underneath
it and reward that too.

---

## B19 — The stride reward made a correct trot worse than standing (found in v2)

The third and final stall, and the most instructive of the three, because the
bug was a **borrowed convention that stopped being true** once the environment
changed.

### The term

```python
def feet_air_time(s):
    moving = xp.where(xp.linalg.norm(s.cmd) > 0.1, 1.0, 0.0)
    return xp.sum((s.feet_air_time - 0.5) * s.feet_first_contact) * moving
```

Straight from legged_gym: pay for air time beyond half a second, credited at
touchdown, so that long strides beat a shuffle.

### The evidence

Work out what a **perfect** trot scores. At the commanded 2 Hz with duty 0.5,
the scheduled swing time per foot is

$$
t_\text{swing} = \frac{1 - \beta}{f} = \frac{0.5}{2} = 0.25\ \text{s}
$$

Every touchdown therefore scores $(0.25 - 0.5) \times 2.0 = -0.5$. Four feet at
two touchdowns per second is eight events, so $-4.0$ per second, and at
$\Delta t = 0.02$:

$$
-0.08\ \text{per control step}
$$

against a total reward of $+0.03$ per step for standing perfectly still.

**The reward function made a flawless trot score roughly four times worse than
doing nothing.** The policy was not failing to find the gait. It had found it,
evaluated it, and correctly rejected it.

The 0.5 s offset is a sensible constant in legged_gym, where the gait frequency
is *emergent* and the resulting strides are long. It became wrong the moment
this repository made gait frequency a **commanded input**: the constant no
longer bears any relation to the swing duration the schedule is asking for.

### The fix

Credit air time up to the scheduled swing duration, rather than offsetting it by
a constant:

$$
r = \mathbb{1}\big[\Vert \mathbf{c}\Vert > 0.1\big] \sum_{i=1}^{4} \min\big(t^\text{air}_i,\ t_\text{swing}\big) \mathbb{1}[\text{foot } i \text{ just landed}]
$$

with $t_\text{swing} = (1 - \beta)/f$ computed from the **active gait command**.

Three properties, all of which the old form lacked:

- **Non-negative.** Early, imperfect steps are never punished — and those are
  the only route to good ones.
- **Saturating.** Hanging a foot in the air beyond its scheduled swing earns
  nothing extra, so it cannot fight the phase schedule.
- **Frequency-independent.** The maximum rate is $4(1-\beta)$ per second
  regardless of $f$, so commanding a faster gait does not inflate the reward.

### Measured effect

Scripted open-loop trot versus standing still, under the actual environment
reward, per control step:

| term | standing | trot | diff |
|---|---|---|---|
| `gait_phase` | $+0.01500$ | $+0.02343$ | $+0.00843$ |
| `feet_clearance` | $-0.00601$ | $-0.00314$ | $+0.00288$ |
| `feet_air_time` | $\phantom{+}0.00000$ | $+0.00056$ | $+0.00056$ |
| `torques` | $-0.00074$ | $-0.00292$ | $-0.00218$ |
| `ang_vel_xy` | $-0.00000$ | $-0.00147$ | $-0.00147$ |
| `feet_slip` | $-0.00000$ | $-0.00085$ | $-0.00085$ |

Stepping now pays about $+0.012$ per step gross and $+0.007$ net of its costs,
*before* any credit for actually moving. Tracking 0.8 m/s is worth a further
$+0.029$, so a competent trot beats standing by roughly $+0.036$ per step —
more than doubling the standing reward.

(The scripted controller in that table only reaches 0.041 m/s, so it collects
almost none of the tracking reward and still loses narrowly overall. That is a
limitation of a hand-written open-loop gait, not of the reward.)

### Lesson

**A borrowed hyperparameter carries the assumptions of the codebase it came
from.** The 0.5 s offset encodes "strides are long because the frequency is
emergent". Import it into an environment where frequency is commanded and it
silently inverts the sign of the term for the exact behaviour you are trying to
produce.

The general check, and it is cheap: **evaluate your reward on a hand-scripted
version of the target behaviour and compare it against the trivial policy.** If
the behaviour you want does not score higher than doing nothing, no algorithm
will find it, and you will spend days blaming exploration.

---

## B20 — The command envelope asked for speeds the robot cannot reach (found in v2)

With B19 fixed, the policy finally learned the gait: at 3M steps
`scripts/gait_analysis.py` reported **86.8% schedule match** and a stride
frequency of exactly 2.00 Hz on all four feet, with the diagonal pairing
correct. It was trotting on the spot.

Achieved forward velocity: **0.064 m/s**, against a command of 0.8.

### The evidence

Speed comes from stride length times step frequency. During stance the foot is
planted and the *body* moves over it, so the fore-aft travel available to a
planted foot is the stride length. That is a kinematic question, answerable
without training anything — the same sweep used for B17, rotated ninety degrees:

| action scale | stance foot travel | implied v at 2 Hz |
|---|---|---|
| 0.25 | 16.2 cm | 0.32 m/s |
| 0.30 | 19.9 cm | 0.40 m/s |
| 0.40 | 28.5 cm | 0.57 m/s |
| 0.50 | 34.4 cm | 0.69 m/s |

The config at the time used `action_scale = 0.30` and a command envelope
reaching **1.5 m/s**. At 2 Hz that needs a 75 cm stride from a 43 cm leg.

So a large part of the command distribution was unreachable, and — this is the
part that matters — an unreachable command is not merely unhelpful. The
exponential tracking kernel *saturates*: at an error of 1.5 m/s its gradient is
about $4\times10^{-5}$ (chapter 10, §10.3). The policy receives no useful signal
for those commands at all, so the command channel becomes noise in the
observation, and the best available behaviour is to ignore it.

This is bug B1's shape again — the command space contained points the robot
could not reach — but quantitative rather than absolute.

### The correction to the correction

The static sweep is a **lower bound**, and taking it as a hard cap would have
been a second mistake in the opposite direction. Driving a scripted open-loop
trot and simply measuring the top speed it reaches:

| action scale | 1.5 Hz | 2.0 Hz | 2.5 Hz | 3.0 Hz |
|---|---|---|---|---|
| 0.30 | 0.52 | 0.62 | 0.65 | 0.69 |
| 0.40 | 0.57 | **0.94** | 1.05 | **1.08** |

The real robot goes considerably faster than the static sweep predicts, because
body pitch, foot roll on the spherical contact, and dynamic effects all add
stride that a fixed-base kinematic sweep cannot see. At `action_scale = 0.40`
the measured ceiling is about 1.08 m/s, not the 0.87 the sweep implies.

Trusting the static number would have capped the robot below its ability;
trusting the aspirational number starved it of gradient. **Measure.**

### The fix

Three parts.

**Raise the action scale to 0.40.** This buys stride length directly. It stops
there because $k_p \times \alpha = 55 \times 0.40 = 22$ N·m, and the hip and
thigh actuators are limited to 23.7 N·m — beyond that the PD offset alone
saturates the actuator.

**Clamp every sampled command to what its own sampled frequency can deliver:**

$$
v_\max = \min\big(\kappa f,\ v_\text{ceiling}\big) \cdot m,
\qquad \kappa = 0.40\ \tfrac{\text{m/s}}{\text{Hz}},\ v_\text{ceiling} = 1.0,\ m = 0.85
$$

Both constants come from the measured table. Sampling speed and frequency
independently, as the code did, generates pairs like "1.0 m/s at 1.5 Hz" that
no gait can satisfy.

**Set the configured range at the limit rather than above it.** The clamp means
a wider range would not be *wrong* — it would just be a number that never means
anything, which is exactly the kind of latent inconsistency this whole exercise
is about removing.

### Confirmed by measurement

`scripts/check_env.py` now prints the speed envelope and the do-nothing baseline
before any training happens:

```
frequency static bound  sampler limit
    1.5 Hz        0.44           0.51
    2.0 Hz        0.58           0.68
    3.0 Hz        0.87           0.85

configured |vx| range: 0.85 m/s
verdict             : every configured command is reachable
do-nothing tracking score : 0.571   (curriculum promotes above 0.85)
```

And the decisive check — does a competent gait actually out-score doing nothing?

| command | scripted trot | standing | winner |
|---|---|---|---|
| 0.50 m/s | $+0.02405$ | $+0.03669$ | standing |
| 0.70 m/s | $+0.03298$ | $+0.03069$ | **trot** |
| 0.85 m/s | $+0.03785$ | $+0.02891$ | **trot** |

The scripted controller is open-loop and always runs at 0.94 m/s, so at a
command of 0.5 it overshoots and is correctly penalised. At commands near its
own speed it wins by up to 31%. That is the reward behaving exactly as intended.

### Lesson

**A command space is part of the action space, and it needs the same
reachability audit.** Before training a command-conditioned policy, ask what
the extremes of the command distribution physically require, and check the
robot can do it. Then check by measurement rather than by kinematics alone,
because a static bound will be conservative in a way that quietly costs you
performance.

---

## Things that were checked and were *not* bugs

Recording these matters as much as recording the bugs. A debugging log that only
lists confirmed problems gives a false impression of how the work actually goes.

**`_quat_rotate_inverse` was correct.** Verified against MuJoCo's
`mju_rotVecQuat` over 200 random quaternions (`tests/test_math.py`). The bug was
that it was never applied to velocities, not that it was wrong.

**Writing torques into `data.ctrl` was correct.** The Menagerie Go2 uses `motor`
actuators with `ctrlrange` in newton-metres, so the units line up. This looks
like a bug — many MJCF models use position servos, where it would be one — and
is not.

**Timeout bootstrapping was handled.** Chapter 4, §4.7 describes how treating
truncation as termination biases the critic. SB3's vectorised wrappers set
`TimeLimit.truncated` automatically and PPO acts on it, so v1 was fine here
despite doing its own truncation inside the environment.

**The PD gains were reasonable.** $k_p = 40$, $k_d = 1$ produce 2.8 cm of
standing sag, which is about right for a compliant legged robot. They were
eventually raised for the separate, measured reason in B17 — not because they
were wrong.

---

## Summary

| # | Bug | Impact |
|---|---|---|
| B1 | Nominal pose outside the joint limits | **Defined v1's ceiling** |
| B2 | World-frame velocity in reward and observation | Turning incoherent |
| B3 | Reward could only express forward motion | Forward-only gait |
| B4 | Gait reward maximised by standing still | No gait signal |
| B5 | 8 blocking IPC round trips per step | Large throughput loss |
| B6 | PD held constant for 40 ms | Judder |
| B7 | 25 Hz control | Judder; halved horizon |
| B8 | Single initial condition | Brittle |
| B9 | GL context per frame | Slow rendering |
| B10 | Contact detector counted self-contacts | Wrong gait signal |
| B11 | Periodic, momentum-teleporting pushes | Memorisable, unphysical |
| B12 | Resume lost normalisation statistics | **Silent policy destruction** |
| B13 | Eval env normalisation not frozen | Noisy eval curve |
| B14 | 2560 gradient steps per rollout | Instability and waste |
| B15 | Docstring/space mismatch | Indicator |
| B16 | `mujoco.viewer` not imported | `render("human")` crashed |
| B17 | Marginal swing clearance (v2) | No foot lift |
| B18 | Stepping reward was piecewise constant (v2) | **No gradient toward stepping** |
| B19 | Stride reward made a correct trot score worse than standing (v2) | **Target behaviour was penalised** |
| B20 | Command envelope asked for unreachable speeds (v2) | Tracking reward saturated; command ignored |

The through-line: **not one of these announced itself.** Every one had to be
found by asking what would have to be true for the code to be right, and then
measuring it.

---

**Previous:** [13. Reading a training run](13-training-diagnostics.md) ·
**Next:** [15. Results](15-results.md)
