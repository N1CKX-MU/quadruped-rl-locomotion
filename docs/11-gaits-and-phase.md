# 11. Gaits, phase, and periodic control

Locomotion is a limit cycle. Nothing in the MDP formulation of chapter 1 prefers
periodic solutions, and a policy handed a velocity-tracking reward will find
whatever aperiodic, twitching motion tracks velocity most cheaply. This chapter
is about putting the periodicity in on purpose.

## 11.1 What a gait is

A **gait** is a repeating pattern of foot contacts. Two numbers describe it.

**Duty factor** $\beta$ — the fraction of a stride cycle that a given foot spends
in stance (on the ground). $\beta = 0.5$ means half stance, half swing.
$\beta = 0.75$ means three-quarters stance. $\beta > 0.5$ for all four feet
implies at least two feet are always down; $\beta < 0.5$ implies there are
moments of flight.

**Phase offsets** $\theta_i$ — where in the cycle each foot's stance begins,
measured in fractions of a cycle from a reference foot.

With the foot ordering (FL, FR, RL, RR) used throughout this repository:

| Gait | Offsets $(\theta_{FL}, \theta_{FR}, \theta_{RL}, \theta_{RR})$ | $\beta$ | Description |
|---|---|---|---|
| **trot** | $(0, \tfrac12, \tfrac12, 0)$ | 0.5 | diagonal pairs together |
| **pace** | $(0, \tfrac12, 0, \tfrac12)$ | 0.5 | lateral pairs together |
| **bound** | $(0, 0, \tfrac12, \tfrac12)$ | 0.5 | front pair, then rear pair |
| **walk** | $(0, \tfrac12, \tfrac14, \tfrac34)$ | 0.75 | four-beat; three feet always down |
| **pronk** | $(0,0,0,0)$ | 0.5 | all four together |
| **stand** | $(0,0,0,0)$ | 1.0 | never lift a foot |

The **trot** is the default and by far the most stable for a robot of the Go2's
proportions. Its diagonal support line passes close to the centre of mass, so
the unsupported moment during each half-cycle is small. The **pace** puts both
support feet on one side and rolls the body. The **bound** pitches it.

Real animals switch gaits with speed — walk, trot, canter, gallop — for energetic
reasons (Hoyt & Taylor, 1981: horses select the gait that minimises metabolic
cost at each speed). This repository trains the trot by default and provides the
others as commandable alternatives; automatic speed-dependent gait selection is
not implemented, and §11.8 says what it would take.

## 11.2 Why a gait must be imposed rather than hoped for

The obvious approach is to reward velocity tracking, penalise energy, and let a
gait emerge. This does sometimes work, and it produces gaits that are
technically efficient and visibly wrong: asymmetric limps, three-legged
shuffles, a hop with one leg dragging. All of these can track a velocity
command at a reasonable torque cost.

There are three concrete reasons to specify the gait instead.

**The reward has no term for "periodic".** Velocity tracking is a statement
about a single instant. Energy is a sum over instants. Neither has any
preference for a repeating structure, so nothing selects one.

**Exploration cannot find the limit cycle.** A trot is a coordinated
four-limb pattern. Random exploration around a standing pose reaches it only by
a coincidence of twelve joints, and the intermediate states — one foot lifted,
the robot toppling — are punished. This is exactly the barrier described in
chapter 10, §10.4.

**Emergent gaits are not controllable.** If the gait is whatever the optimiser
found, you cannot ask for a different one, or a different step frequency,
without retraining.

## 11.3 The phase clock

Introduce a scalar $\phi \in [0, 1)$ advancing at the commanded step frequency
$f$:

$$
\phi_{t+1} = \big(\phi_t + f \Delta t\big) \bmod 1
\tag{11.1}
$$

At $f = 2$ Hz and $\Delta t = 0.02$ s, $\phi$ advances by 0.04 per control step
and completes a cycle in 25 steps.

Each foot's **local phase** is $\phi$ shifted by its offset, and the schedule
says a foot should be in stance while its local phase is inside the duty window:

$$
d_i(\phi) = \mathbb{1}\Big[ \big(\phi + \theta_i\big) \bmod 1 < \beta \Big]
\tag{11.2}
$$

That is the entire mechanism. Eleven lines in `envs/gait.py`:

```python
def advance_phase(phase, frequency, dt):
    return float((phase + frequency * dt) % 1.0)

def desired_contact(phase, offsets, duty):
    local_phase = xp.mod(phase + offsets, 1.0)
    return xp.where(local_phase < duty, 1.0, 0.0)
```

A useful sanity check, and a test: averaged over a full cycle, each foot's
desired contact must equal the duty factor exactly.

$$
\frac{1}{1}\int_0^1 d_i(\phi)\mathrm{d}\phi = \beta
$$

```python
def test_duty_factor_is_the_stance_fraction():
    samples = np.array([desired_contact(p, offsets, duty)
                        for p in np.linspace(0, 1, 1000, endpoint=False)])
    assert np.allclose(samples.mean(axis=0), duty, atol=2e-3)
```

## 11.4 The phase reward

$$
r_\text{gait} = \frac{1}{4} \sum_{i=1}^{4} \Big[ c_i d_i + (1 - c_i)(1 - d_i) \Big]
\tag{11.3}
$$

$c_i$ is the measured contact, $d_i$ the scheduled one. The bracket is 1 when
they agree and 0 when they do not, so $r_\text{gait}$ is the fraction of feet
currently doing the right thing. Bounded in $[0, 1]$.

Compare to v1's $|c_{FL}c_{RR} - c_{FR}c_{RL}|$ (chapter 10, §10.6), which a
motionless robot maximised forever. The difference is not a better contact
expression — no function of $\mathbf{c}$ alone can express alternation, because
alternation is a property of a sequence. The difference is the introduction of
$\mathbf{d}$: a **time-varying reference**. Once you have that, the reward
becomes a trivial agreement measure and the interesting content has moved into
the clock.

What a frozen robot now scores: under a trot schedule with $\beta = 0.5$, a
robot standing on all four feet matches the two feet that are supposed to be in
stance and mismatches the two that are supposed to be swinging, every step, for
$r_\text{gait} = 0.5$. A robot frozen on one diagonal pair also averages 0.5.
There is no stationary configuration that scores above 0.5, and every step in
the right direction pays immediately — lifting one correct foot takes it to
0.75.

### It is necessary but not sufficient

Equation 11.3 is built on a **binary** contact flag, so it is piecewise
constant: it does not change at all until a foot physically breaks contact. For
a policy that has never stepped, the gradient toward stepping is exactly zero.

That is why the reward set also contains `feet_clearance`, a smooth cost on
swing-foot height (chapter 10, §10.4b). The two do different jobs: clearance
gets the foot off the ground, the phase reward gets it off the ground *at the
right time*. A 1.1M-step run stalled with only the latter.

### Why this term is weighted so heavily

`gait_phase` has weight 1.5, equal to velocity tracking, because it is the term
that supplies the *timing*. Nothing else in the reward set says when a foot
should be down:

- `track_lin_vel_xy` rewards the **result** of walking, which is undiscoverable
  until you already walk.
- `feet_air_time` pays at touchdown, so it is exactly zero for a robot that has
  never stepped.
- `feet_clearance` pays for raising a swing foot, but says nothing about *which*
  foot or *when* — on its own it would produce four independent legs waving.

So the three stepping terms decompose cleanly: clearance gets a foot off the
ground, the phase reward gets it off the ground at the right moment, and air
time makes the resulting stride a real one. Under-weighting the phase reward
cost half a million steps of training in this repository; omitting the clearance
term cost another million.

## 11.5 The clock has to be in the observation

$$
o_{48:50} = \big(\sin 2\pi\phi, \quad \cos 2\pi\phi\big)
$$

Without this the phase reward is unlearnable. From the policy's point of view,
two states with identical joint angles, velocities and attitude would receive
different rewards for the same action depending on an invisible variable — which
is the definition of a non-Markov observation, and it appears to the learner as
irreducible noise.

The $(\sin, \cos)$ embedding rather than raw $\phi$: the phase is **circular**.
$\phi = 0.99$ and $\phi = 0.01$ are 0.02 apart in time and 0.98 apart as real
numbers. A network fed raw $\phi$ sees a discontinuity once per stride, right at
the point where two feet are swapping roles.

Only the *global* clock is in the observation, not the per-foot phases. With
`gaits: ["trot"]` the offsets are constant, so the network absorbs them into its
weights. §11.8 covers what changes for multiple gaits.

## 11.6 Step frequency as a command

$f$ appears in equation 11.1 and is sampled per episode from
$[1.5, 3.0]$ Hz. Because it changes how fast the clock the policy observes
advances, the policy learns to step faster or slower on demand, with no
architectural change.

There is real physics behind the range. A pendulum of length $L$ has natural
frequency $\sqrt{g/L}/2\pi$; for the Go2's 0.213 m thigh plus 0.213 m calf,
$L \approx 0.3$ m gives $\approx 0.9$ Hz for a full swing, so a 2 Hz stride
(0.5 s per cycle, 0.25 s of swing per foot at $\beta = 0.5$) is a comfortable
multiple. Below about 1.2 Hz the robot has to hold each foot up unnaturally
long; above about 3.5 Hz the swing becomes torque-limited.

`scripts/play.py` binds `[` and `]` to step frequency so you can feel this.

## 11.7 Measuring what you got

`scripts/gait_analysis.py` records a rollout and reports:

- **duty factor** per foot, versus the commanded $\beta$
- **stride frequency** per foot, from touchdown intervals, versus commanded $f$
- **phase offsets** relative to FL, versus the reference $\theta_i$
- **schedule match** — the fraction of steps where $c_i = d_i$, which is
  numerically the same quantity as equation 11.3

and plots the commanded schedule directly above the measured contacts, so
disagreement is visible rather than inferred.

The interpretations worth knowing:

| Symptom | Likely cause |
|---|---|
| duty factor well above the reference | shuffling: feet toggle on cue without real swing. Raise `feet_air_time` |
| schedule match below 0.7 | `gait_phase` under-weighted, or $f$ outside the trained range |
| measured offsets do not match the reference | the policy is running a *different* gait from the one commanded |
| match high but velocity tracking poor | the policy is marching on the spot |

Each of these is printed as a note by the script, so you do not have to remember
the table.

## 11.8 Training more than one gait

The config ships `gaits: ["trot"]`. To train a policy that can trot, pace and
bound on command:

```yaml
commands:
  gaits: ["trot", "pace", "bound", "walk"]
```

That alone is not sufficient, and it is worth being clear about why. With one
gait the offsets $\theta_i$ are constant, so the policy can infer the whole
schedule from $\phi$ alone. With several, the same $\phi$ implies different
$\mathbf{d}$ depending on which gait is active, and the gait identity is
**not currently in the observation**. The policy would be trying to satisfy a
schedule it cannot see.

The fix is to give it the per-foot clock instead of the global one.
`envs/gait.py` already provides it:

```python
def clock_signal(phase, offsets):
    local = 2.0 * xp.pi * xp.mod(phase + offsets, 1.0)
    return xp.concatenate([xp.sin(local), xp.cos(local)])
```

Eight numbers instead of two, and the gait becomes fully specified by the
observation. `Go2Env._get_obs` and `obs_dim` change by two lines. The
alternative — a one-hot gait ID — works too but generalises worse, since the
clock representation lets you interpolate to offsets never seen in training.

An honest caveat: pace and bound are genuinely harder than trot, and mixing them
in from the start slows trot learning. Train trot first, then fine-tune with the
wider gait set.

## 11.9 What this does not do

**No automatic gait selection.** The gait is commanded, not chosen. A policy that
picks its own gait by speed would need either a discrete action for the gait or a
higher-level policy; the standard trick is to make $f$ and $\beta$ continuous
actions and let the energy penalty select them, which is close to what Hoyt and
Taylor observed in horses.

**No foot-placement planning.** The schedule says *when* a foot should be down,
never *where*. The policy decides placement implicitly through joint targets. On
flat ground this is fine. On rough terrain you want a Raibert-style heuristic —
place the foot at

$$
\mathbf{p}_i^* = \mathbf{p}_i^{\text{hip}} + \frac{\beta T}{2}\mathbf{v} + k(\mathbf{v} - \mathbf{v}^*)
$$

(hip position, plus half a stance duration of travel, plus a velocity-error
correction) — as a reference for a foot-position reward.

**No aerial-phase control.** With $\beta = 0.5$ and a real trot, there are
moments with no feet down. Nothing here explicitly manages flight time; it falls
out of the dynamics.

---

**Previous:** [10. Reward engineering](10-reward-engineering.md) ·
**Next:** [12. Curriculum and domain randomisation](12-curriculum-and-domain-randomization.md)
