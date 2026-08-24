# 8. The Go2, MuJoCo, and PD control

This chapter is about the physical system. It is independent of which RL
algorithm you use, and getting it wrong makes the algorithm irrelevant — as v1
demonstrated.

## 8.1 The robot

The Unitree Go2 is a 15.2 kg quadruped with twelve actuated joints, three per
leg. The model used here is `unitree_go2/scene.xml` from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

Mass distribution, from the model:

| Body | Mass (kg) | ×4 legs |
|---|---|---|
| trunk (`base`) | 6.921 | — |
| hip | 0.678 | 2.712 |
| thigh | 1.152 | 4.608 |
| calf | 0.241 | 0.964 |
| **total** | | **15.206** |

Just under half the mass is in the trunk. That matters: the legs are light
relative to the body, so their swing dynamics barely perturb the trunk, and a
quasi-static view of the body with fast legs is a reasonable mental model.

### Joints and their conventions

Each leg has **hip** (abduction/adduction, rotates the leg sideways), **thigh**
(hip flexion, swings the leg fore-aft), and **calf** (knee).

Joint ordering in `qpos[7:]` and in every action vector is
**FL, FR, RL, RR**, each as `(hip, thigh, calf)`.

| Joint | Range (rad) | Torque limit (N·m) |
|---|---|---|
| hip (all four) | $[-1.047, +1.047]$ | 23.7 |
| thigh (front) | $[-1.571, +3.491]$ | 23.7 |
| thigh (rear) | $[-0.524, +4.538]$ | 23.7 |
| calf (all four) | $[-2.723, -0.838]$ | 45.4 |

Two things to notice.

**The knee is stronger.** 45.4 N·m against 23.7 for the others, because the knee
carries the robot's weight through the largest moment arm.

**The calf range does not contain zero.** $[-2.723, -0.838]$ — the knee is
*always* bent. There is no configuration of this robot with a straight rear
leg segment. Remember this; it is about to matter a great deal.

## 8.2 MuJoCo state layout

`nq = 19`, `nv = 18`. The asymmetry is quaternions: orientation needs four
numbers to represent but has only three degrees of freedom.

```
qpos[0:3]   base position (world frame), metres
qpos[3:7]   base orientation, unit quaternion, (w, x, y, z)
qpos[7:19]  twelve joint angles, radians

qvel[0:3]   base linear velocity, WORLD frame
qvel[3:6]   base angular velocity
qvel[6:18]  twelve joint velocities
```

`qvel[0:3]` is world-frame. That single fact is bug B2: v1 fed it straight into
both the reward and the observation as though it were the robot's forward
velocity. It is not — it is the velocity in the fixed world frame, which stops
corresponding to "forward" the instant the robot yaws.

The conversion is one line:

$$
v_b = R(q)^{\top} \, v_w
$$

implemented as `quat_rotate_inverse` in `envs/go2_env.py`, using the identity

$$
q^{*} v q = v - 2w (u \times v) + 2\, u \times (u \times v), \qquad u = (q_x, q_y, q_z)
\tag{8.1}
$$

v1 had this function, correctly implemented, and used it only for gravity.

## 8.3 The `home` keyframe, and the bug that defined v1

MuJoCo's `mj_resetData` sets `qpos = model.qpos0`. For this model, `qpos0` is
whatever the MJCF's body positions and joint defaults imply — here, **all twelve
joints at exactly 0 rad**, with the trunk at $z = 0.445$ m.

The model also defines a keyframe:

```xml
<keyframe>
  <key name="home"
       qpos="0 0 0.27  1 0 0 0   0 0.9 -1.8   0 0.9 -1.8   0 0.9 -1.8   0 0.9 -1.8"
       ctrl="0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8"/>
</keyframe>
```

That is the standing pose: hip 0, thigh $+0.9$, calf $-1.8$, trunk 27 cm up.

v1 did this:

```python
mujoco.mj_resetData(self.model, self.data)
mujoco.mj_forward(self.model, self.data)
self.default_joint_pos = self.data.qpos[7:].copy()   # <- all zeros
```

and then, every control step:

```python
target_pos = self.default_joint_pos + action * self.action_scale   # 0 + a*0.5
```

So the PD controller's target was $0 \pm 0.5$ rad on every joint. Compare
against the calf's legal range:

```
reachable calf target:  [-0.5, +0.5]
calf joint limit:       [-2.723, -0.838]
```

**These sets do not intersect.** Not "the pose was suboptimal", not "the policy
had to work harder" — for every action the policy could possibly emit, the
commanded knee angle was outside the knee's physical travel, by at least 0.34
rad and at most 1.3.

It gets worse. `qpos0` itself has the calf joints at 0, which is *also* outside
$[-2.723, -0.838]$. So v1 reset every single episode into a configuration that
violated the robot's own joint limits, and MuJoCo's limit constraints
immediately began shoving them back while the PD controller pulled the other
way. Every episode began with the actuators fighting the endstops.

The check is three lines and is now `scripts/check_env.py`'s first section:

```
default_joint_pos : [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8]
nominal height    : 0.270 m
matches Go2 home  : YES
```

The fix is one line:

```python
self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
self.default_joint_pos = self.model.key_qpos[self._home_key_id, 7:].copy()
```

The lesson generalises past this repository: **when you load a robot model, print
the nominal pose and check it against the joint limits before you do anything
else.** Nothing errors when you get this wrong. The simulation runs, the
training curve goes up, papers get written.

## 8.4 Actuators: torque, not position

Menagerie's Go2 uses `motor` actuators:

```xml
<motor class="abduction" name="FL_hip" joint="FL_hip_joint"/>
```

with `ctrlrange` in newton-metres. So `data.ctrl` is **torque**, and writing a
torque there is correct — a point worth stating because it is the kind of thing
that looks like a bug and is not. v1 got this right.

Some other MJCF models use `position` actuators, where `data.ctrl` is a target
angle in radians and MuJoCo runs its own internal servo. Writing a torque into
one of those produces a robot that behaves bizarrely with no error message.
Always check which kind your model has.

## 8.5 PD control

The policy outputs **position targets**, not torques:

$$
q^*_t = q_\text{nom} + \alpha \, a_t, \qquad \alpha = 0.40\ \text{rad}
$$

and a joint-space PD law converts them:

$$
\tau = k_p (q^* - q) - k_d \dot q
\tag{8.2}
$$

clipped to the actuator limits.

### Why not have the policy output torque directly?

Three reasons, in increasing order of importance.

**It matches the hardware.** A real Go2's low-level controller takes position
targets with gains. A torque policy would have to be re-derived for deployment.

**It is a better-conditioned action space.** A position target is a *setpoint*;
the PD law handles the fast, stiff dynamics of tracking it. The policy operates
on the slow, task-relevant timescale. A torque policy has to learn the stiff
part too, from scratch, at 50 Hz.

**It has built-in stability.** Equation 8.2 with positive gains is a passive
spring-damper: even a badly wrong target produces a bounded, dissipative force.
A badly wrong torque produces whatever it produces. This makes early training
survivable.

### Gain selection, measured rather than guessed

At zero action the PD targets the nominal pose exactly, so how much the robot
sags under its own weight is a direct measurement of whether the gains are
sensible. Measured on this model (150 control steps, then average):

| $k_p$ | $k_d$ | settled height | sag from 0.27 m | peak torque |
|---|---|---|---|---|
| 20 | 0.5 | 0.196 m | 7.4 cm | 8.8 N·m |
| 25 | 0.6 | 0.206 m | 6.4 cm | 10.4 N·m |
| 40 | 1.0 | 0.242 m | 2.8 cm | 8.0 N·m |
| **55** | **1.4** | **0.254 m** | **1.6 cm** | — |
| 60 | 1.5 | 0.256 m | 1.4 cm | 7.1 N·m |
| 80 | 2.0 | 0.261 m | 0.9 cm | 6.8 N·m |

v2 uses $k_p = 55$, $k_d = 1.4$. Sag matters more than it first appears: it is
subtracted directly from the foot clearance available during swing, because
while two legs retract the other two carry the whole robot and sag further
(bug B17). $k_p = 80$ scores better still on paper and was rejected — that
stiff, the policy starts exploiting contact impulses no real actuator could
produce.

Note that $k_p = 40$, $k_d = 1.0$ are **v1's values**, and they were never the
problem. The gains were raised for the separate, separately-measured reason
above. When a system underperforms there is a temptation to change everything;
the discipline is to change only what the evidence implicates.

The action scale interacts with the gains through the torque limit:
$k_p 	imes lpha = 55 	imes 0.40 = 22$ N·m against a 23.7 N·m hip and thigh
limit, so at $|a| = 1$ the PD offset alone very nearly saturates the actuator.
That is what caps $lpha$ at 0.40, and hence caps stride length and top speed
(bug B20).

### The bug that *was* in the PD controller

v1 computed the torque once per control step and then ran the physics forward
with it frozen:

```python
torques = self.kp * (target_pos - current_pos) - self.kd * current_vel
self.data.ctrl[:] = np.clip(torques, -ctrl_max, ctrl_max)
for _ in range(self.frame_skip):        # 20 steps = 40 ms
    mujoco.mj_step(self.model, self.data)
```

For 40 milliseconds, $\tau$ is a constant. The $-k_d \dot q$ term — the entire
damping mechanism — is evaluated once and then held, which means it is
computed from a velocity that is up to 40 ms stale. During a foot strike, when
joint velocities change sign in a few milliseconds, the "damping" term can be
actively *driving* the oscillation it exists to suppress.

v2 recomputes inside the loop:

```python
for _ in range(self.decimation):
    torque = kp * (target_pos - self.data.qpos[7:]) - kd * self.data.qvel[6:]
    torque = np.clip(torque, -self.torque_limits, self.torque_limits)
    self.data.ctrl[:] = torque
    mujoco.mj_step(self.model, self.data)
```

This is what real hardware does: the low-level PD loop runs at 500–1000 Hz while
the policy runs at 50. Simulating it any other way simulates a controller nobody
deploys.

## 8.6 Timing

| | v1 | v2 |
|---|---|---|
| physics timestep | 2 ms | 2 ms |
| decimation | 20 | 10 |
| control rate | 25 Hz | 50 Hz |
| PD evaluations per control step | 1 | 10 |
| PD rate | 25 Hz | 500 Hz |
| episode length | 1000 steps = 40 s | 1000 steps = 20 s |

50 Hz is the field standard for legged RL. The reason is the stride: a trot at
2 Hz gives 25 control steps per stride at 50 Hz, and only 12 at 25 Hz. Twelve
decision points is not many for placing four feet.

There is a consequence that is easy to miss: **changing the control rate changes
the meaning of $\gamma$.** At $\gamma = 0.99$, the effective horizon is
$1/(1-\gamma) = 100$ steps — 4 seconds at 25 Hz, 2 seconds at 50 Hz. Doubling
the control rate halved the planning horizon. Two seconds is the right number
for locomotion, so this was a happy consequence rather than a problem, but it
was a consequence that had to be checked rather than assumed.

## 8.7 Contacts

MuJoCo generates contacts each step in `data.contact`. To know whether a foot is
loaded you must ask two questions, and v1 asked neither:

1. **Is the other geom the floor?** v1 counted any contact involving a foot
   geom, which includes foot-on-shin and foot-on-foot self-contact.
2. **Does the contact carry load?** v1 used the *existence* of a contact pair.
   A foot grazing the ground and a foot bearing a quarter of the robot's weight
   scored identically.

v2 asks both:

```python
if g1 == self.floor_geom_id:   other = g2
elif g2 == self.floor_geom_id: other = g1
else:                          continue
mujoco.mj_contactForce(self.model, self.data, i, force)
if abs(force[0]) < force_threshold:  continue
```

`mj_contactForce` returns a 6-vector in the contact frame; `force[0]` is the
normal component. The threshold is 1 N against a 149 N robot.

v1 also had a fallback if the foot geoms could not be found by name:

```python
if len(self.foot_geom_ids) != 4:
    self.foot_geom_ids = list(range(self.model.ngeom - 4, self.model.ngeom))
```

"the last four geoms" is not a foot detector. If it had ever triggered, the gait
reward would have silently rewarded contacts on arbitrary bodies. v2 raises
instead: a misconfiguration should stop the program, not quietly change what is
being optimised.

Friction on the floor is `[1.0, 0.005, 0.0001]` — sliding, torsional, rolling.
Only the first matters much here, and it is randomised in $[0.4, 1.4]$ during
training (chapter 12).

## 8.8 The full control loop

```
policy(obs)             ->  a in [-1,1]^12          50 Hz
q* = q_nom + 0.25 a                                 50 Hz
    repeat 10x:
        tau = 40 (q* - q) - 1.0 qdot               500 Hz
        clip tau to +/-23.7 (hip, thigh), +/-45.4 (calf)
        mj_step                                     500 Hz
observe, reward, terminate                          50 Hz
```

Four numbers to keep in mind: 50 Hz policy, 500 Hz PD and physics, 0.25 rad of
action authority, and a nominal pose of $(0, 0.9, -1.8)$ per leg. Everything in
the next three chapters is built on top of them.

---

**Previous:** [7. Why PPO and not SAC or TD3](07-why-not-sac-td3.md) ·
**Next:** [9. Observations and actions](09-observations-and-actions.md)
