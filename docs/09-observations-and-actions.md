# 9. Observations and actions

The observation is the policy's entire world. Every design decision here is a
decision about what the robot is *able* to know, and therefore about what
behaviours are even representable.

## 9.1 The vector

Fifty numbers, in this order (`Go2Env._get_obs`):

| Slice | Block | Dim | Scale | Frame |
|---|---|---|---|---|
| `0:3` | projected gravity $g_b$ | 3 | 1.0 | base |
| `3:6` | angular velocity $\omega_b$ | 3 | 0.25 | base |
| `6:9` | linear velocity $v_b$ | 3 | 2.0 | base |
| `9:21` | joint positions $q - q_\text{nom}$ | 12 | 1.0 | joint |
| `21:33` | joint velocities $\dot q$ | 12 | 0.05 | joint |
| `33:45` | previous action $a_{t-1}$ | 12 | 1.0 | — |
| `45:48` | command $(v_x^*, v_y^*, \omega_z^*)$ | 3 | (2, 2, 0.25) | base |
| `48:50` | gait clock $(\sin 2\pi\phi, \cos 2\pi\phi)$ | 2 | 1.0 | — |

For comparison, v1's 53:

| Block | Dim | Problem |
|---|---|---|
| orientation quaternion | 4 | carries absolute yaw |
| angular velocity | 3 | world frame, unscaled |
| linear velocity | 3 | **world frame** (bug B2), unscaled |
| joint positions | 12 | absolute, not relative to nominal |
| joint velocities | 12 | unscaled ($O(20)$) |
| previous action | 12 | fine |
| foot contacts | 4 | binary, from a broken detector (B10) |
| command | 3 | constant for the entire run (B3) |

The docstring said 49; the space was 53. Nobody had recounted.

## 9.2 Projected gravity, not the quaternion

The first block is

$$
g_b = R(q)^{\top} \begin{pmatrix} 0 \\ 0 \\ -1 \end{pmatrix}
$$

the world gravity direction expressed in the base frame. Upright, it is
$(0, 0, -1)$; tilted, magnitude moves into the $xy$ components.

**Why not the quaternion?** Because a quaternion encodes yaw, and yaw is a
*commanded* quantity in this task, not a state to be corrected. Feed absolute
heading to the policy and it can — will — correlate behaviour with a direction
that has no physical meaning. Nothing about walking north differs from walking
east, but a network given the heading has no way to know that, and it will
happily fit spurious structure in whatever heading distribution the training
episodes happened to contain.

Projected gravity is precisely the yaw-free part of the attitude. This is
verified as a test:

```python
def test_projected_gravity_is_yaw_invariant():
    for yaw in np.linspace(-np.pi, np.pi, 17):
        q = Go2Env._euler_to_quat(0.0, 0.0, yaw)
        assert np.allclose(Go2Env.quat_rotate_inverse(q, g), g, atol=1e-12)
```

**Why not roll and pitch angles?** They would also be yaw-free. But extracting
them requires `atan2` and `asin`, which have a singularity at pitch $= \pm\pi/2$
and produce a discontinuity in roll as you pass through it. The robot does reach
those attitudes while falling over — exactly the states where the value function
most needs to be well-behaved. $g_b$ is smooth everywhere and costs one
quaternion rotation.

Three numbers with a redundant constraint ($\|g_b\| = 1$) rather than two
independent angles is a good trade.

Reset randomisation makes yaw uniform on $[-\pi, \pi)$ regardless, which is a
second, independent line of defence against heading-correlated behaviour.

## 9.3 Everything in the base frame

The velocities are rotated into the robot's own frame:

$$
v_b = R^{\top} v_w, \qquad \omega_b = R^{\top} \omega_w
$$

This is bug B2 fixed. v1 fed `qvel[0:3]` — world frame — into both the reward
and the observation. The consequences compound:

- The policy is asked to make $v_x^{\text{world}}$ equal the command. If the
  robot is facing east, "forward" in the reward means "east", not "the way I am
  pointing".
- Since v1 randomised nothing about the initial heading and never commanded a
  turn, the robot always faced $+x$ and the bug was invisible. It became a bug
  the moment anyone asked for a yaw rate.
- The lateral penalty $-|v_y^{\text{world}}|$ punished the robot for the world
  drifting sideways relative to it, which after any yaw is not the same thing as
  crabbing.

Commands are in the base frame too, which is what makes them intuitive: $v_x^*$
means "forward, from the robot's point of view", the same way a joystick works.

## 9.4 Joint positions relative to nominal

$$
\tilde q = q - q_\text{nom}
$$

Two reasons.

**Zero means "standing".** The network's natural output near initialisation is
zero, and now zero is a meaningful, useful pose rather than an arbitrary one.
Combined with the 0.01 output-layer gain (chapter 6), the untrained policy
stands.

**It matches the action.** The action is also an offset from $q_\text{nom}$, so
$\tilde q$ and $a$ live in the same space. The policy's job — "given where the
joints are relative to nominal, where should I put them relative to nominal" —
becomes a mapping between comparable quantities, which is easier to learn than
a mapping between an absolute and a relative frame.

## 9.5 Scaling

Each block is multiplied by a constant so it lands roughly in $[-1, 1]$:

| Block | Typical raw range | Scale | Scaled |
|---|---|---|---|
| $g_b$ | $[-1, 1]$ | 1.0 | $[-1, 1]$ |
| $\omega_b$ | $\pm 4$ rad/s | 0.25 | $\pm 1$ |
| $v_b$ | $\pm 0.5$ m/s | 2.0 | $\pm 1$ |
| $\tilde q$ | $\pm 1$ rad | 1.0 | $\pm 1$ |
| $\dot q$ | $\pm 20$ rad/s | 0.05 | $\pm 1$ |
| command $v$ | $\pm 1.5$ m/s | 2.0 | $\pm 3$ |
| command $\omega$ | $\pm 1.5$ rad/s | 0.25 | $\pm 0.4$ |

"But `VecNormalize` already normalises the observation" — true, and this is
still worth doing, for three reasons:

1. `VecNormalize` needs to *learn* the statistics. During the first thousands of
   steps the observation is unnormalised, and those are the steps that determine
   which basin the policy falls into.
2. Its running statistics drift as the policy's behaviour changes. Fixed scales
   do not.
3. It is a deployment liability. A policy whose correctness depends on a running
   statistic accumulated over 8 million steps is a policy you must ship a pickle
   file with — and lose it and the policy is worthless (bug B12). Fixed scales
   are in the source code.

## 9.6 The previous action

$a_{t-1}$, unscaled. Two jobs.

**It makes the action-rate penalty learnable.** The reward includes
$-\|a_t - a_{t-1}\|^2$. Without $a_{t-1}$ in the observation, that term is a
function of something the policy cannot see, so it appears as unexplainable
noise in the reward and the policy can only reduce it by reducing its variance
globally.

**It carries actuator state.** The real joint has not reached $a_{t-1}$'s target
yet — that is what the PD controller is still doing. $a_{t-1}$ tells the policy
where the controller is currently headed, which is genuine state information
that $q$ and $\dot q$ do not fully capture.

## 9.7 The gait clock

$$
(\sin 2\pi\phi, \; \cos 2\pi\phi)
$$

Two numbers for one scalar $\phi \in [0,1)$, because $\phi$ is **circular**:
$\phi = 0.999$ and $\phi = 0.001$ are adjacent in time but maximally distant as
real numbers. A network fed raw $\phi$ would see a discontinuity once per
stride. The $(\sin, \cos)$ embedding is the standard fix: continuous, unique,
and bounded.

This block is the whole reason v2 can control the gait. Chapter 11 develops it.
The short version: the reward compares actual foot contacts against a schedule
derived from $\phi$, and the policy can only satisfy that schedule if it knows
where in the stride it is. Without the clock in the observation, the phase
reward is unlearnable noise.

Note the global clock is in the observation, not the per-foot phases. Per-foot
phases would be eight numbers and are a deterministic function of $\phi$ and the
gait — but the gait *identity* is not in the observation, which is a deliberate
limitation of the current version: with `gaits: ["trot"]` there is only one, so
the offsets are constant and the network absorbs them. To train a genuinely
multi-gait policy you must add the per-foot clock (`clock_signal` in
`envs/gait.py` returns exactly this) or a one-hot gait ID. That is a two-line
change and is noted in chapter 11.

## 9.8 What is deliberately absent

**Foot contacts.** v1 included four binary contact flags. v2 does not, for two
reasons: the detector was unreliable (B10), and contact state is largely implied
by $q$, $\dot q$ and the clock. More importantly, on hardware a contact flag is
a thresholded force estimate that is noisy and often wrong, so a policy that
leans on it transfers badly. legged_gym omits it as well.

**Absolute position and heading.** $x$, $y$, and yaw are not in the observation.
The task is velocity tracking, not navigation. Including position would let the
policy learn behaviours that depend on where it happens to be in the arena.

**Terrain information.** Out of scope for this version. A height-map scan around
the feet is the standard addition for rough terrain.

**Observation history.** A stack of the last $N$ observations is the standard way
to give a feedforward policy some memory of unmodelled dynamics, and it is the
first thing to add for sim-to-real (chapter 16).

## 9.9 The action space

$$
a \in [-1, 1]^{12}, \qquad q^* = q_\text{nom} + 0.25\, a
$$

**Why position targets, not torques** — chapter 8, §8.5.

**Why 0.25 rad?** It bounds how far the policy can move a joint in one control
step. v1 used 0.5. Smaller is a real constraint on what the policy can do, and
smaller is better here for a reason worth stating: the action scale sets the
**maximum stiffness of the closed loop**. With $k_p = 40$ and $\alpha = 0.25$,
the largest torque the policy can command from the nominal pose is
$40 \times 0.25 = 10$ N·m, well inside the 23.7 N·m limit. Raise $\alpha$ and
the policy gains the authority to saturate the actuators every step, which in
simulation produces impressive-looking motion that no real robot can execute.

Note the interaction with bug B1. v1's action scale of 0.5 was not itself
wrong — combined with a nominal calf angle of 0 rad it produced a reachable
target set of $[-0.5, 0.5]$ against a joint limit of $[-2.723, -0.838]$: an
empty intersection. Fixing the nominal pose is what makes a *smaller* action
scale viable.

**Clipping.** The environment clips the incoming action to $[-1,1]$ before using
it. Nothing guarantees a Gaussian policy respects the box, and an unclipped
action of 5.0 would command a joint 1.25 rad from nominal.

Note the asymmetry with `ppo_from_scratch/ppo.py`, which stores the *unclipped*
action while sending the clipped one — see chapter 6, §6.4. The environment must
clip for safety; the algorithm must remember what it actually sampled.

## 9.10 Is this Markov?

Chapter 1 argued the Markov property is a design constraint on the observation,
not a given. Auditing the choices here:

| Hidden variable | Handled by |
|---|---|
| stride phase | in the observation (the clock) |
| commanded velocity | in the observation |
| actuator lag | partly, via $a_{t-1}$ |
| ground friction | randomised across episodes (chapter 12) |
| payload mass | randomised across episodes |
| an in-progress push | **not observable** |

The last row is the honest gap. A push is applied to the trunk for 0.15 s and
the policy cannot see it — it can only infer it from the resulting acceleration,
one step late. This makes the process formally non-Markov in the observation.

It is left that way deliberately, because it is realistic: a real robot cannot
see a shove coming either. The consequence is that the value function has
irreducible variance around push events, which shows up as a small permanent
ceiling on `explained_variance`. That is the correct trade, but it should be a
known trade rather than an accident.

---

**Previous:** [8. The Go2, MuJoCo, and PD control](08-the-robot.md) ·
**Next:** [10. Reward engineering](10-reward-engineering.md)
