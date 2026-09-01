"""MJX (GPU) version of the Go2 locomotion environment.

Why this exists
---------------
The CPU environment tops out around 616 steps/s end to end on an 8-core machine
(docs/17-mjx-and-scaling.md). MJX is MuJoCo re-implemented in JAX: the physics
becomes a pure function that ``jax.vmap`` maps across a batch and ``jax.jit``
compiles to GPU kernels, so instead of N processes each stepping one robot you
get one program stepping N robots as a batched tensor op. Four thousand parallel
environments is routine.

What is shared with the CPU environment
---------------------------------------
The reward maths and the gait schedule, exactly - ``envs/rewards.py`` and
``envs/gait.py`` are written array-generically and dispatch on the array type.
This is not a stylistic choice. Two independent implementations of a sixteen-term
reward *will* diverge, and the divergence surfaces months later as an
unexplained performance gap between the CPU and GPU policies. There is a test
(``tests/test_mjx.py``) asserting that the two backends compute the same total.

What is necessarily different
-----------------------------
* Everything is functional. No mutation, no Python ``if`` on a traced value, no
  data-dependent shapes. Control flow is ``jnp.where``.
* Episodes do not end early. Termination is a mask; a "done" environment is
  reset by ``jnp.where`` on the next step rather than by breaking out of a loop.
* Contacts come from fixed-size arrays. MJX pre-allocates ``ncon`` slots and
  marks inactive ones with a positive ``dist``, so contact detection is a
  masked reduction rather than a loop over ``data.ncon``.
* The model is ``scene_mjx.xml``, which has simplified collision geometry and a
  bounded contact budget. The physics is therefore NOT identical to the CPU
  model - evaluate on the CPU environment, not this one.

Robustness, and where each piece lives
--------------------------------------
The CPU environment's robustness features are all present, but split across two
files by necessity:

* **State-level**, in this file: actuator-gain randomisation, external pushes
  and observation noise. These vary per environment simply by living in the
  state dict, which ``vmap`` already maps over.
* **Model-level**, in ``mjx/domain_randomization.py``: ground friction and link
  masses. These live in ``mjx.Model``, which is *shared* across the batch, so
  per-environment variation needs a batched model plus matching ``in_axes``.
  ``mjx/train_mjx.py`` passes it to brax as ``randomization_fn``.

The push logic is branchless. A Python ``if remaining > 0`` would be a
ConcretizationTypeError under jit and - worse - silently wrong under vmap if it
ever did trace, because one environment's timing would decide the whole batch's.

Episode truncation is reported as a separate ``truncated`` flag rather than
folded into ``done``, so a caller can bootstrap across it instead of treating it
as terminal (chapter 4, 4.7). brax's EpisodeWrapper applies its own limit on
top; this one makes the environment correct when driven directly.

The defaults are asserted equal to the CPU environment's by tests/test_mjx.py,
which caught real drift once already (action_scale and the B20 feasibility
clamp).

Status: verified on GPU (RTX 3050, WSL2). Peak 6,579 env-steps/s at 2048
environments - see docs/17.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from envs import gait as gait_mod
from envs import rewards as reward_mod
from envs.commands import CommandRanges
from envs.paths import resolve_asset_path

DEFAULT_XML = "mujoco_menagerie/unitree_go2/scene_mjx.xml"
FOOT_GEOM_NAMES = ("FL", "FR", "RL", "RR")

# Observation scales, identical to the CPU environment so a policy trained here
# can be evaluated there.
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_JOINT_VEL = 0.05
OBS_SCALE_CMD = jnp.array([2.0, 2.0, 0.25])


def quat_rotate_inverse(q, v):
    """Express a world-frame vector in the base frame. wxyz convention.

    Same identity as Go2Env.quat_rotate_inverse:
        q* v q = v - 2w (u x v) + 2 u x (u x v)
    """
    w = q[0]
    u = q[1:4]
    t = 2.0 * jnp.cross(u, v)
    return v - w * t + jnp.cross(u, t)


class Go2MJXEnv:
    """Batched Go2 locomotion under MJX.

    ``reset`` and ``step`` are pure functions of (rng) and (state, action). Map
    them over a batch with ``jax.vmap`` and compile with ``jax.jit``; see
    ``mjx/train_mjx.py`` for the brax PPO wiring, or the ``__main__`` block at
    the bottom of this file for a minimal example.
    """

    def __init__(
        self,
        xml_path: str = DEFAULT_XML,
        decimation: int = 10,
        action_scale: float = 0.40,
        kp: float = 55.0,
        kd: float = 1.4,
        episode_length: int = 1000,
        min_base_height: float = 0.12,
        max_pitch_roll: float = 0.8,
        reset_noise_scale: float = 0.1,
        command_ranges: CommandRanges | None = None,
        command_resample_steps: int = 250,   # 5 s at 50 Hz
        stand_probability: float = 0.10,
        max_speed_per_hz: float = 0.40,
        max_speed: float = 1.0,
        feasibility_margin: float = 0.85,
        gait: str = "trot",
        reward_weights: dict | None = None,
        lin_vel_sigma: float = 0.20,
        ang_vel_sigma: float = 0.25,
        # Robustness, mirroring envs/go2_env.py
        randomize_dynamics: bool = False,
        kp_range: tuple = (0.8, 1.2),
        kd_range: tuple = (0.8, 1.2),
        push_enabled: bool = True,
        push_interval_s: tuple = (3.0, 7.0),
        push_duration_s: float = 0.15,
        push_force: tuple = (-40.0, 40.0),
        obs_noise_scale: float = 0.0,
    ):
        self.mj_model = mujoco.MjModel.from_xml_path(resolve_asset_path(xml_path))
        # Named `sys` because brax's DomainRandomizationVmapWrapper swaps it
        # per batch element (env.unwrapped.sys = ...). `model` stays as a
        # readable alias.
        self.sys = mjx.put_model(self.mj_model)

        self.decimation = decimation
        self.physics_dt = float(self.mj_model.opt.timestep)
        self.dt = self.physics_dt * decimation
        self.action_scale = action_scale
        self.kp = kp
        self.kd = kd
        self.episode_length = episode_length
        self.min_base_height = min_base_height
        self.max_pitch_roll = max_pitch_roll
        self.reset_noise_scale = reset_noise_scale
        self.command_resample_steps = command_resample_steps
        self.stand_probability = stand_probability
        # Speed feasibility, mirroring envs/commands.py. Without this the GPU
        # path reintroduces bug B20: speed and gait frequency sampled
        # independently produce commands no gait can satisfy, the exponential
        # tracking kernel saturates, and the command becomes noise.
        self.max_speed_per_hz = max_speed_per_hz
        self.max_speed = max_speed
        self.feasibility_margin = feasibility_margin

        # Nominal pose from the 'home' keyframe (bug B1 - see docs/14).
        key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id < 0:
            raise RuntimeError("The MJCF has no 'home' keyframe.")
        self.init_qpos = jnp.array(self.mj_model.key_qpos[key_id])
        self.default_joint_pos = self.init_qpos[7:]

        self.n_joints = self.mj_model.nu
        joint_ids = self.mj_model.actuator_trnid[:, 0]
        ranges = self.mj_model.jnt_range[joint_ids]
        span = ranges[:, 1] - ranges[:, 0]
        self.soft_joint_limits = jnp.array(
            [ranges[:, 0] + 0.05 * span, ranges[:, 1] - 0.05 * span]
        ).T
        self.torque_limits = jnp.array(self.mj_model.actuator_ctrlrange[:, 1])

        self.floor_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.foot_geom_ids = jnp.array([
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in FOOT_GEOM_NAMES
        ])
        # Precomputed on the host. Anything derived from model metadata must be
        # a concrete array before tracing begins - calling int() on a value
        # inside a jitted function is a ConcretizationTypeError.
        # Sphere radius of the foot geoms, used to place the robot ON the floor
        # at reset rather than through it.
        self.foot_radius = float(self.mj_model.geom_size[
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                              FOOT_GEOM_NAMES[0])][0])

        self.foot_body_ids = jnp.array([
            int(self.mj_model.geom_bodyid[
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)])
            for n in FOOT_GEOM_NAMES
        ])
        base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        # Needed by the push force and by domain randomisation, so it is an
        # attribute rather than a local.
        self.base_body_id = base_id
        foot_set = set(int(g) for g in self.foot_geom_ids)
        self.trunk_geom_ids = jnp.array([
            g for g in range(self.mj_model.ngeom)
            if self.mj_model.geom_bodyid[g] == base_id and g not in foot_set
            and self.mj_model.geom_contype[g] != 0
        ])

        offsets, duty = gait_mod.gait_params(gait)
        self.gait_offsets = jnp.array(offsets)
        self.gait_duty = float(duty)

        self.ranges = command_ranges or CommandRanges()
        self.weights = reward_mod.resolve_weights(reward_weights)
        self.lin_vel_sigma = lin_vel_sigma
        self.ang_vel_sigma = ang_vel_sigma

        self.randomize_dynamics = randomize_dynamics
        self.kp_range = kp_range
        self.kd_range = kd_range
        self.push_enabled = push_enabled
        self.push_interval_s = push_interval_s
        self.push_duration_s = push_duration_s
        self.push_force = push_force
        self.obs_noise_scale = obs_noise_scale
        self.max_episode_steps = episode_length

        self.obs_size = 3 + 3 + 3 + self.n_joints * 3 + 3 + 2
        self.action_size = self.n_joints

    @property
    def model(self):
        """Alias for ``sys``. Brax requires the name ``sys``; the rest of this
        repository talks about models."""
        return self.sys

    # ------------------------------------------------------------------ #
    #  Commands                                                           #
    # ------------------------------------------------------------------ #

    def sample_command(self, rng):
        """Draw (vx, vy, yaw, frequency, height). Functional: takes a PRNG key."""
        r = self.ranges
        k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
        vx = jax.random.uniform(k1, minval=r.lin_vel_x[0], maxval=r.lin_vel_x[1])
        vy = jax.random.uniform(k2, minval=r.lin_vel_y[0], maxval=r.lin_vel_y[1])
        yaw = jax.random.uniform(k3, minval=r.ang_vel_yaw[0], maxval=r.ang_vel_yaw[1])
        freq = jax.random.uniform(k4, minval=r.gait_frequency[0],
                                  maxval=r.gait_frequency[1])
        height = jax.random.uniform(k5, minval=r.base_height[0],
                                    maxval=r.base_height[1])
        # Explicit stand-still mode, plus snapping of near-zero commands. Both
        # are jnp.where rather than `if`, because this runs inside jit.
        # Feasibility clamp (bug B20), as jnp ops because this runs under jit:
        #     v_max = min(max_speed_per_hz * f, max_speed) * margin
        v_max = jnp.minimum(self.max_speed_per_hz * freq, self.max_speed)
        v_max = v_max * self.feasibility_margin
        vx = jnp.clip(vx, -v_max, v_max)
        # Lateral strides are shorter than fore-aft ones; half is the same
        # conservative allowance the CPU sampler uses.
        vy = jnp.clip(vy, -0.5 * v_max, 0.5 * v_max)

        stand = jax.random.uniform(k6) < self.stand_probability
        cmd = jnp.array([vx, vy, yaw])
        small = jnp.linalg.norm(cmd) < 0.15
        cmd = jnp.where(stand | small, jnp.zeros(3), cmd)
        return cmd, freq, height

    # ------------------------------------------------------------------ #
    #  Contacts (fixed-size, masked)                                      #
    # ------------------------------------------------------------------ #

    def _contacts(self, data):
        """Foot contacts, trunk contact, and a count of non-foot ground contacts.

        MJX allocates a fixed number of contact slots and marks inactive ones
        with dist > 0, so this is a masked reduction over a constant-shape array
        rather than the CPU version's loop over data.ncon.
        """
        contact = data._impl.contact
        geom = contact.geom                      # (ncon, 2)
        active = contact.dist < 0.0              # (ncon,)

        g1, g2 = geom[:, 0], geom[:, 1]
        with_floor = (g1 == self.floor_geom_id) | (g2 == self.floor_geom_id)
        other = jnp.where(g1 == self.floor_geom_id, g2, g1)
        ground = active & with_floor

        # (4, ncon) -> (4,)
        foot_hits = (other[None, :] == self.foot_geom_ids[:, None]) & ground[None, :]
        foot_contact = jnp.any(foot_hits, axis=1).astype(jnp.float32)

        is_foot = jnp.any(other[None, :] == self.foot_geom_ids[:, None], axis=0)
        undesired = jnp.sum(ground & ~is_foot).astype(jnp.float32)

        is_trunk = jnp.any(other[None, :] == self.trunk_geom_ids[:, None], axis=0)
        trunk = jnp.any(ground & is_trunk)
        return foot_contact, undesired, trunk

    def _feet_vel_xy(self, data):
        """Horizontal foot velocity, world frame, shape (4, 2).

        Approximated from the geom's linear velocity via the body Jacobian is
        expensive under MJX; the CPU environment uses mj_objectVelocity. Here we
        use the parent body's linear velocity, which is close enough for the
        slip penalty (the foot is a small sphere rigidly attached to the calf).
        """
        return data.cvel[self.foot_body_ids][:, 3:5]

    # ------------------------------------------------------------------ #
    #  Observation                                                        #
    # ------------------------------------------------------------------ #

    def _obs(self, data, prev_action, cmd, phase, rng=None):
        quat = data.qpos[3:7]
        proj_g = quat_rotate_inverse(quat, jnp.array([0.0, 0.0, -1.0]))
        ang_b = quat_rotate_inverse(quat, data.qvel[3:6])
        lin_b = quat_rotate_inverse(quat, data.qvel[0:3])
        phase_rad = 2.0 * jnp.pi * phase
        obs = jnp.concatenate([
            proj_g,
            ang_b * OBS_SCALE_ANG_VEL,
            lin_b * OBS_SCALE_LIN_VEL,
            data.qpos[7:] - self.default_joint_pos,
            data.qvel[6:] * OBS_SCALE_JOINT_VEL,
            prev_action,
            cmd * OBS_SCALE_CMD,
            jnp.array([jnp.sin(phase_rad), jnp.cos(phase_rad)]),
        ])
        if self.obs_noise_scale > 0.0 and rng is not None:
            obs = obs + jax.random.normal(rng, obs.shape) * self.obs_noise_scale
        return obs

    # ------------------------------------------------------------------ #
    #  reset / step                                                       #
    # ------------------------------------------------------------------ #

    def reset(self, rng):
        (rng, k_joint, k_h, k_yaw, k_rp, k_v, k_w, k_jv, k_ph, k_cmd,
         k_kp, k_kd, k_push, k_obs) = jax.random.split(rng, 14)

        qpos = self.init_qpos
        n = self.reset_noise_scale
        qpos = qpos.at[7:].add(
            jax.random.uniform(k_joint, (self.n_joints,), minval=-n, maxval=n))
        qpos = qpos.at[2].add(jax.random.uniform(k_h, minval=-0.02, maxval=0.05))
        yaw = jax.random.uniform(k_yaw, minval=-jnp.pi, maxval=jnp.pi)
        rp = jax.random.uniform(k_rp, (2,), minval=-0.05, maxval=0.05)
        qpos = qpos.at[3:7].set(_euler_to_quat(rp[0], rp[1], yaw))

        qvel = jnp.zeros(self.mj_model.nv)
        qvel = qvel.at[0:3].set(jax.random.uniform(k_v, (3,), minval=-0.3, maxval=0.3))
        qvel = qvel.at[3:6].set(jax.random.uniform(k_w, (3,), minval=-0.3, maxval=0.3))
        qvel = qvel.at[6:].set(
            jax.random.uniform(k_jv, (self.n_joints,), minval=-0.5, maxval=0.5))

        data = mjx.make_data(self.sys).replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.sys, data)

        # Place the robot ON the ground, not through it.
        #
        # The keyframe height is exactly the standing height, so ANY downward
        # perturbation - the height noise, or a joint angle that extends a leg -
        # buries a foot in the floor. Measured before this fix: 59 of 128 reset
        # states had a foot below z=0, with penetrations up to 3.9 cm, and MJX's
        # reduced-iteration solver diverged to NaN on roughly 1 in 128 of them
        # within a few steps. Even where it did not diverge, the episode began
        # with a large spurious contact impulse.
        #
        # So: measure the lowest foot after posing, and lift the base by however
        # much it takes to clear the ground.
        foot_z = data.geom_xpos[self.foot_geom_ids, 2] - self.foot_radius
        lift = jnp.maximum(0.0, 0.002 - jnp.min(foot_z))
        qpos = qpos.at[2].add(lift)
        data = mjx.forward(self.sys, data.replace(qpos=qpos))

        phase = jax.random.uniform(k_ph)
        cmd, freq, height = self.sample_command(k_cmd)
        prev_action = jnp.zeros(self.n_joints)

        # Actuator-gain randomisation. These live in STATE rather than in the
        # model, so they vary per environment under vmap without needing a
        # batched mjx.Model. Friction and link masses are model fields and are
        # randomised instead through brax's randomization_fn - see
        # mjx/domain_randomization.py.
        rand = 1.0 if self.randomize_dynamics else 0.0
        kp_scale = 1.0 + rand * (jax.random.uniform(
            k_kp, minval=self.kp_range[0], maxval=self.kp_range[1]) - 1.0)
        kd_scale = 1.0 + rand * (jax.random.uniform(
            k_kd, minval=self.kd_range[0], maxval=self.kd_range[1]) - 1.0)

        # Push schedule. Randomised timing, so the policy cannot memorise it
        # the way it could v1's every-200-steps rule (bug B11).
        push_countdown = jax.random.uniform(
            k_push, minval=self.push_interval_s[0], maxval=self.push_interval_s[1])

        return dict(
            data=data,
            rng=rng,
            cmd=cmd,
            gait_freq=freq,
            cmd_height=height,
            phase=phase,
            prev_action=prev_action,
            prev_joint_vel=data.qvel[6:],
            feet_air_time=jnp.zeros(4),
            last_contact=jnp.zeros(4),
            step=jnp.array(0, dtype=jnp.int32),
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            push_countdown=push_countdown,
            push_remaining=jnp.array(0.0),
            obs=self._obs(data, prev_action, cmd, phase, k_obs),
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            truncated=jnp.array(0.0),
        )

    def _update_push(self, data, rng, countdown, remaining):
        """External shove, as a real force for a real duration (bug B11).

        Branchless: every environment in the batch evaluates the same graph and
        the decisions are `jnp.where`. A Python `if` on `remaining > 0` would be
        a ConcretizationTypeError under jit, and worse, would be *silently*
        wrong under vmap if it ever did trace - one environment's timing would
        decide the whole batch's.
        """
        if not self.push_enabled:
            return data, countdown, remaining

        k_force, k_next = jax.random.split(rng)
        pushing = remaining > 0.0
        due = jnp.logical_and(jnp.logical_not(pushing), countdown <= 0.0)

        force = jax.random.uniform(
            k_force, (3,), minval=self.push_force[0], maxval=self.push_force[1])
        # Mostly horizontal: a vertical yank is not a physical shove.
        force = force.at[2].multiply(0.25)

        applied = jnp.where(due, force, jnp.where(pushing,
                                                  data.xfrc_applied[self.base_body_id, :3],
                                                  jnp.zeros(3)))
        xfrc = data.xfrc_applied.at[self.base_body_id, :3].set(applied)

        new_remaining = jnp.where(
            due, self.push_duration_s, jnp.maximum(remaining - self.dt, 0.0))
        # Reschedule when a push ends; otherwise keep counting down.
        ended = jnp.logical_and(pushing, new_remaining <= 0.0)
        fresh = jax.random.uniform(
            k_next, minval=self.push_interval_s[0], maxval=self.push_interval_s[1])
        new_countdown = jnp.where(
            jnp.logical_or(due, ended), fresh,
            jnp.where(pushing, countdown, countdown - self.dt))

        return data.replace(xfrc_applied=xfrc), new_countdown, new_remaining

    def step(self, state, action):
        action = jnp.clip(action, -1.0, 1.0)
        target = self.default_joint_pos + action * self.action_scale

        rng, k_push, k_obs, k_cmd = jax.random.split(state["rng"], 4)

        data, push_countdown, push_remaining = self._update_push(
            state["data"], k_push, state["push_countdown"],
            state["push_remaining"])

        # Per-environment actuator gains (domain randomisation).
        kp = self.kp * state["kp_scale"]
        kd = self.kd * state["kd_scale"]

        def pd_step(data, _):
            torque = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
            torque = jnp.clip(torque, -self.torque_limits, self.torque_limits)
            data = mjx.step(self.sys, data.replace(ctrl=torque))
            return data, torque

        # scan rather than a Python loop: one compiled kernel, not `decimation`
        # copies of the physics graph.
        data, torques = jax.lax.scan(
            pd_step, data, None, length=self.decimation)
        last_torque = torques[-1]

        phase = jnp.mod(state["phase"] + state["gait_freq"] * self.dt, 1.0)
        desired = gait_mod.desired_contact(phase, self.gait_offsets, self.gait_duty)

        contact, undesired, trunk = self._contacts(data)
        first_contact = ((contact > 0.5) & (state["last_contact"] < 0.5)).astype(
            jnp.float32)
        air_at_landing = state["feet_air_time"] * first_contact
        feet_air_time = jnp.where(contact > 0.5, 0.0,
                                  state["feet_air_time"] + self.dt)

        quat = data.qpos[3:7]
        rstate = reward_mod.RewardState(
            lin_vel_b=quat_rotate_inverse(quat, data.qvel[0:3]),
            ang_vel_b=quat_rotate_inverse(quat, data.qvel[3:6]),
            proj_gravity=quat_rotate_inverse(quat, jnp.array([0.0, 0.0, -1.0])),
            base_height=data.qpos[2],
            joint_pos=data.qpos[7:],
            joint_vel=data.qvel[6:],
            joint_vel_prev=state["prev_joint_vel"],
            default_joint_pos=self.default_joint_pos,
            soft_joint_limits=self.soft_joint_limits,
            torques=last_torque,
            action=action,
            prev_action=state["prev_action"],
            contact=contact,
            desired_contact=desired,
            feet_air_time=air_at_landing,
            feet_first_contact=first_contact,
            feet_vel_xy=self._feet_vel_xy(data),
            feet_height=data.geom_xpos[self.foot_geom_ids][:, 2],
            cmd=state["cmd"],
            cmd_base_height=state["cmd_height"],
            undesired_contacts=undesired,
            dt=self.dt,
            lin_vel_sigma=self.lin_vel_sigma,
            ang_vel_sigma=self.ang_vel_sigma,
            target_swing_time=(1.0 - self.gait_duty)
            / jnp.maximum(state["gait_freq"], 1e-3),
        )
        total, terms = reward_mod.compute(rstate, self.weights)
        reward = total * self.dt

        proj_gz = rstate.proj_gravity[2]
        done = (
            trunk
            | (data.qpos[2] < self.min_base_height)
            | (proj_gz > -jnp.cos(self.max_pitch_roll))
        ).astype(jnp.float32)

        step = state["step"] + 1
        # Mid-episode command resampling, as a masked update. k_cmd was split
        # from the state RNG at the top of step(), alongside the push and
        # observation-noise keys - reusing one key for several draws would
        # correlate the pushes with the commands.
        resample = (step % self.command_resample_steps) == 0
        new_cmd, new_freq, new_height = self.sample_command(k_cmd)
        cmd = jnp.where(resample, new_cmd, state["cmd"])
        gait_freq = jnp.where(resample, new_freq, state["gait_freq"])
        cmd_height = jnp.where(resample, new_height, state["cmd_height"])

        # Episode time limit. Reported separately from `done` so the caller can
        # bootstrap across it rather than treating it as a terminal state - the
        # distinction chapter 4, 4.7 is about. brax's EpisodeWrapper also
        # applies its own limit; this one makes the environment correct when
        # driven directly.
        truncated = (step >= self.max_episode_steps).astype(jnp.float32)

        return dict(
            data=data,
            rng=rng,
            cmd=cmd,
            gait_freq=gait_freq,
            cmd_height=cmd_height,
            phase=phase,
            prev_action=action,
            prev_joint_vel=data.qvel[6:],
            feet_air_time=feet_air_time,
            last_contact=contact,
            step=step,
            kp_scale=state["kp_scale"],
            kd_scale=state["kd_scale"],
            push_countdown=push_countdown,
            push_remaining=push_remaining,
            obs=self._obs(data, action, cmd, phase, k_obs),
            reward=reward,
            done=done,
            truncated=truncated,
        ), terms


def _euler_to_quat(roll, pitch, yaw):
    cr, sr = jnp.cos(roll * 0.5), jnp.sin(roll * 0.5)
    cp, sp = jnp.cos(pitch * 0.5), jnp.sin(pitch * 0.5)
    cy, sy = jnp.cos(yaw * 0.5), jnp.sin(yaw * 0.5)
    return jnp.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


if __name__ == "__main__":
    # Minimal batched smoke test. On CPU JAX this is slow; it exists to prove
    # the environment traces, jits and vmaps, not to benchmark anything.
    import time

    env = Go2MJXEnv()
    batch = 8
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    t0 = time.perf_counter()
    state = reset(keys)
    print("reset  : obs %s  compile %.1fs" % (state["obs"].shape,
                                              time.perf_counter() - t0))

    action = jnp.zeros((batch, env.action_size))
    t0 = time.perf_counter()
    state, terms = step(state, action)
    print("step   : compile %.1fs" % (time.perf_counter() - t0))

    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        state, terms = step(state, action)
    state["obs"].block_until_ready()
    el = time.perf_counter() - t0
    print("throughput: %.0f env-steps/s over %d envs (CPU JAX)"
          % (n * batch / el, batch))
    print("reward    : %s" % state["reward"])
    print("height    : %s" % state["data"].qpos[:, 2])
    print("terms     : %s" % sorted(terms))
