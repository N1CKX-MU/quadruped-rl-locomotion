# 16. What sim-to-real would take

Sim-to-real is **out of scope** for this version. This chapter is an honest
enumeration of what stands between the current policy and a real Go2, so that
the gap is documented rather than hand-waved.

The v1 README claimed the policy "directly transfers" the PD architecture and
the network. That is true of the *architecture*. It is not true of the *policy*,
and this chapter explains why.

## 16.1 The blocking problem: the actor sees what a real robot cannot

The observation includes base linear velocity $v_b$ (dimensions 6:9).

A real Go2 cannot measure it. There is no sensor for the velocity of the trunk
through the world. What the hardware has is:

- an IMU: linear **acceleration** and angular velocity, both noisy and drifting
- joint encoders: position and velocity, accurate
- foot force estimates: derived, noisy

Getting $v_b$ from those requires integrating accelerometer data, which drifts
without bound, or a state estimator fusing IMU with leg kinematics during stance
(this is what Unitree's own stack does). Either way, what the robot actually has
is an *estimate* with error that grows during flight phases and resets on
touchdown.

A policy trained on ground-truth $v_b$ and deployed on an estimate will behave
differently in exactly the situations where the estimate is worst — flight
phases, slips, and impacts. Which are the situations that matter.

### The standard fix: asymmetric actor–critic

Chapter 2, §2.6 introduced this. Give the actor only what hardware can provide,
and give the critic everything:

| | Actor (deployed) | Critic (training only) |
|---|---|---|
| projected gravity | yes | yes |
| angular velocity | yes | yes |
| **linear velocity** | **no** | yes |
| joint positions and velocities | yes | yes |
| previous action | yes | yes |
| command, gait clock | yes | yes |
| **friction, payload, push force** | **no** | **yes** |
| observation history (3–5 frames) | yes | yes |

The critic's job is prediction, and it is discarded after training, so
privileged information costs nothing and removes variance that would otherwise
appear as noise in every advantage estimate.

The actor loses information and must compensate with **history**. A stack of the
last 3–5 observations lets it infer velocity from the integrated
kinematics — the same information a state estimator would use, learned instead of
hand-derived.

In SB3 this needs a custom policy class with two feature extractors; the
from-scratch implementation makes it easier, since `ActorCritic.actor` and
`.critic` are already independent networks and only need different inputs.

## 16.2 Actuator dynamics

The current model assumes an ideal torque source: commanded torque appears
instantly, exactly.

A real Unitree motor has rotor inertia, gearbox friction and backlash, current
control bandwidth of roughly 1 kHz, and a torque–speed envelope — the available
torque falls as joint velocity rises, which matters precisely during fast swing.

Three levels of fidelity:

1. **Torque-speed limiting.** $\tau_{\max}(\dot q) = \tau_0 (1 - \dot q / \dot q_{\max})$.
   One line, captures the first-order effect.
2. **First-order lag.** $\tau_{\text{applied}} = \tau_{\text{applied}} + (\tau_{\text{cmd}} - \tau_{\text{applied}}) \Delta t / T$
   with $T \approx 5$ ms. Another line.
3. **A learned actuator network.** Hwangbo et al. (2019) trained a small MLP
   mapping recent position-error and velocity history to measured torque, from
   bench data on the real motors, and used it in place of the ideal model. This
   was the single largest contributor to their successful transfer.

Levels 1 and 2 are cheap and worth doing regardless. Level 3 requires hardware
you have to already own.

## 16.3 Latency

The current loop is instantaneous: observe, act, apply, all at the same instant.

A real loop has sensor read time, network or bus transport to the compute, policy
inference, and transport back to the motor drivers. Total is typically 5–20 ms,
i.e. **0.25 to 1 full control steps at 50 Hz**.

A policy trained with zero latency and deployed with 20 ms is acting on
information one step stale, and the resulting phase lag is exactly what turns a
stable limit cycle into an oscillation.

The fix is to simulate it — buffer observations by a randomised 1–2 control
steps — and to randomise the amount, so the policy learns a control law robust
across the range rather than one tuned to a specific lag.

## 16.4 Observation noise

`obs_noise_scale` exists in the config and defaults to 0. For real transfer it
needs to be non-zero and, more importantly, **per-channel**, because the sensors
differ enormously in quality:

| Channel | Realistic noise |
|---|---|
| joint position | $\sigma \approx 0.005$ rad (encoders are good) |
| joint velocity | $\sigma \approx 0.5$ rad/s (differentiated, noisy) |
| projected gravity | $\sigma \approx 0.02$, plus slow drift |
| angular velocity | $\sigma \approx 0.1$ rad/s, plus bias |
| linear velocity | dominated by estimator error — see §16.1 |

A single scalar applied to a normalised observation, which is what the current
implementation does, is a placeholder rather than a model.

## 16.5 Contact and terrain

MuJoCo's contact model is a soft constraint solve. Real contact involves
compliant rubber feet, surface texture, and stick–slip transitions that no rigid
solver reproduces exactly.

The practical mitigations: randomise friction widely (currently
$\times[0.4, 1.4]$; for transfer, $[0.25, 1.75]$ is more typical), randomise
restitution and contact stiffness, and — most importantly — **train on rough
terrain**. A policy trained only on a plane learns foot placements that assume
the ground is exactly where it expects. Even mild procedural roughness forces it
to be robust to a few centimetres of height error, and that robustness is most
of what real ground looks like.

Terrain is explicitly out of scope for this version. The repository already
contains `scripts/generate_terrain.py` and `configs/terrain_rough.xml` from v1,
which are a starting point.

## 16.6 Mass and inertia

Currently randomised: trunk mass, $+[-1, +2]$ kg.

Not randomised, and worth adding: centre-of-mass offset (a real robot carries
its battery slightly off-centre), link inertias (the CAD-derived tensors in the
MJCF are idealised), and per-link masses.

The centre-of-mass offset is the one that matters most, because a constant
attitude bias is exactly the kind of error a policy trained on a symmetric robot
has no mechanism to correct.

## 16.7 What is already right

Worth stating, because the list above is long.

**The action interface.** The policy outputs joint position targets fed to a PD
controller — exactly what Unitree's low-level API accepts. No re-derivation
needed.

**The PD gains are realistic.** $k_p = 55$, $k_d = 1.4$ are in the range real
Go2 controllers use, and were chosen from a sag measurement (chapter 8, §8.5)
rather than picked to make simulation easy. A policy trained against
$k_p = 500$ would look great in sim and be undeployable.

**The control rate matches.** 50 Hz is what real quadruped policies run at.

**Torque limits are enforced.** 23.7 / 45.4 N·m, from the model, clipped every
substep. The policy cannot learn a gait that requires impossible torque.

**The nominal pose is the manufacturer's.** After B1, `default_joint_pos` is the
model's own `home` keyframe.

**Domain randomisation exists** and is structured to be widened rather than
added from scratch.

## 16.8 The order to do it in

If you were to take this to hardware:

1. **Remove $v_b$ from the actor**, add observation history, add an asymmetric
   critic. Nothing else matters until the policy runs on observable quantities.
2. **Add latency** (1–2 randomised control steps) and per-channel observation
   noise.
3. **Add actuator lag and a torque–speed envelope.**
4. **Widen domain randomisation** — friction, CoM offset, link masses.
5. **Add rough terrain.**
6. **Then** consider an actuator network, which requires bench data from the
   real motors.

Steps 1–5 are simulation work and are perhaps a week. Step 6 needs hardware.

Expect the policy to get *worse* in simulation at every step. That is the point:
a policy that scores lower in a harder, more honest simulation is more likely to
work outside it.

## 16.9 The honest summary

The current policy would not walk on a real Go2, primarily because of §16.1 — it
depends on a velocity signal the robot does not have. Everything else on the
list is a degradation; that one is a wall.

None of this is a criticism of the current version. Sim-to-real was scoped out
deliberately, and the design decisions reflect that: the actor is *allowed*
privileged information, which is why asymmetric actor–critic buys nothing here
(chapter 2, §2.6). The moment you remove $v_b$, most of this chapter becomes the
next piece of work.

---

**Previous:** [15. Results](15-results.md) ·
**Next:** [17. Scaling with MJX](17-mjx-and-scaling.md)
