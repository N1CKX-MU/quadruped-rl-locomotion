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

Status: verified to run, vmap and jit correctly on CPU JAX. It has not been
benchmarked on a GPU by the author; the throughput claims in chapter 17 for MJX
are from the literature, not from this machine.
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
        action_scale: float = 0.30,
        kp: float = 55.0,
        kd: float = 1.4,
        episode_length: int = 1000,
        min_base_height: float = 0.12,
        max_pitch_roll: float = 0.8,
        reset_noise_scale: float = 0.1,
        command_ranges: CommandRanges | None = None,
        command_resample_steps: int = 250,   # 5 s at 50 Hz
        stand_probability: float = 0.10,
        gait: str = "trot",
        reward_weights: dict | None = None,
        lin_vel_sigma: float = 0.20,
        ang_vel_sigma: float = 0.25,
    ):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.model = mjx.put_model(self.mj_model)

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
        self.foot_body_ids = jnp.array([
            int(self.mj_model.geom_bodyid[
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)])
            for n in FOOT_GEOM_NAMES
        ])
        base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
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

        self.obs_size = 3 + 3 + 3 + self.n_joints * 3 + 3 + 2
        self.action_size = self.n_joints

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

    def _obs(self, data, prev_action, cmd, phase):
        quat = data.qpos[3:7]
        proj_g = quat_rotate_inverse(quat, jnp.array([0.0, 0.0, -1.0]))
        ang_b = quat_rotate_inverse(quat, data.qvel[3:6])
        lin_b = quat_rotate_inverse(quat, data.qvel[0:3])
        phase_rad = 2.0 * jnp.pi * phase
        return jnp.concatenate([
            proj_g,
            ang_b * OBS_SCALE_ANG_VEL,
            lin_b * OBS_SCALE_LIN_VEL,
            data.qpos[7:] - self.default_joint_pos,
            data.qvel[6:] * OBS_SCALE_JOINT_VEL,
            prev_action,
            cmd * OBS_SCALE_CMD,
            jnp.array([jnp.sin(phase_rad), jnp.cos(phase_rad)]),
        ])

    # ------------------------------------------------------------------ #
    #  reset / step                                                       #
    # ------------------------------------------------------------------ #

    def reset(self, rng):
        rng, k_joint, k_h, k_yaw, k_rp, k_v, k_w, k_jv, k_ph, k_cmd = \
            jax.random.split(rng, 10)

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

        data = mjx.make_data(self.model).replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.model, data)

        phase = jax.random.uniform(k_ph)
        cmd, freq, height = self.sample_command(k_cmd)
        prev_action = jnp.zeros(self.n_joints)

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
            obs=self._obs(data, prev_action, cmd, phase),
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
        )

    def step(self, state, action):
        action = jnp.clip(action, -1.0, 1.0)
        target = self.default_joint_pos + action * self.action_scale

        def pd_step(data, _):
            torque = self.kp * (target - data.qpos[7:]) - self.kd * data.qvel[6:]
            torque = jnp.clip(torque, -self.torque_limits, self.torque_limits)
            data = mjx.step(self.model, data.replace(ctrl=torque))
            return data, torque

        # scan rather than a Python loop: one compiled kernel, not `decimation`
        # copies of the physics graph.
        data, torques = jax.lax.scan(
            pd_step, state["data"], None, length=self.decimation)
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
        # Mid-episode command resampling, as a masked update.
        rng, k = jax.random.split(state["rng"])
        resample = (step % self.command_resample_steps) == 0
        new_cmd, new_freq, new_height = self.sample_command(k)
        cmd = jnp.where(resample, new_cmd, state["cmd"])
        gait_freq = jnp.where(resample, new_freq, state["gait_freq"])
        cmd_height = jnp.where(resample, new_height, state["cmd_height"])

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
            obs=self._obs(data, action, cmd, phase),
            reward=reward,
            done=done,
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
