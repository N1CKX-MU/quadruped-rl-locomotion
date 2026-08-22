"""Unitree Go2 locomotion environment (v2).

What changed from v1, and why
-----------------------------
Every numbered item below is a bug fixed or a capability added; each is
explained at length in docs/14-debugging-log.md. Read that chapter alongside
this file.

B1  The nominal pose is now taken from the model's ``home`` keyframe, not from
    ``qpos0``. v1 took ``qpos0``, which is all twelve joints at 0 rad - legs
    straight and splayed, a pose the robot cannot stand in. With an action
    scale of 0.5 rad the policy could not reach the real standing calf angle of
    -1.8 rad from there, so it was being asked to walk out of a pose it could
    never leave. This single line is the root cause of v1's ceiling.
B2  All velocities used by the reward and the observation are rotated into the
    base frame. ``qvel[0:3]`` is world-frame; tracking it means "forward" stops
    meaning forward the moment the robot yaws.
B3  Commands are three-dimensional and resampled (see envs/commands.py), and
    both linear axes are tracked, so strafing and turning are expressible.
B6  The PD law is evaluated at every physics substep instead of once per
    control step. Holding a "PD" torque constant for 40 ms makes the damping
    term meaningless exactly when it is needed.
B7  Control runs at 50 Hz (decimation 10 at a 2 ms physics step) instead of
    25 Hz.
B8  Resets randomise base height, attitude, velocity and gait phase, not just
    joint angles.
B9  A single renderer is created lazily and reused, instead of building and
    destroying an OpenGL context per frame.
B10 Foot contact requires contact *with the floor* above a force threshold,
    rather than any contact touching a foot geom (which counted foot-on-shin
    self-contact).
B11 Pushes are applied as forces through ``xfrc_applied`` over a randomised
    interval and duration, rather than teleporting momentum into ``qvel`` on a
    fixed 200-step schedule the policy could memorise.
B16 ``mujoco.viewer`` is imported explicitly; v1's ``render("human")`` would
    have raised AttributeError.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer  # B16: v1 used mujoco.viewer without importing it

from envs import gait as gait_mod
from envs import rewards as reward_mod
from envs.commands import Command, CommandRanges, CommandSampler


class Go2Env(gym.Env):
    """Command-conditioned locomotion for the Unitree Go2.

    Observation (50 dims), each block scaled to roughly unit magnitude so the
    policy sees a well-conditioned input even before VecNormalize:

        projected gravity, base frame        3
        base angular velocity, base frame    3
        base linear velocity, base frame     3
        joint positions minus nominal       12
        joint velocities                    12
        previous action                     12
        command [vx, vy, yaw_rate]           3
        gait clock [sin 2*pi*phi, cos ...]   2
                                            --
                                            50

    Two deliberate choices worth defending:

    * Projected gravity replaces v1's raw quaternion. A quaternion carries yaw,
      and yaw is a *commanded* quantity here - feeding absolute yaw in lets the
      policy correlate behaviour with a heading that has no physical meaning.
      Projected gravity is the yaw-free part of the attitude, which is exactly
      the part that matters for staying upright.
    * The gait clock is in the observation. Without it the policy cannot know
      which part of the stride it is in, and the phase reward would look like
      noise. With it, the gait becomes a commandable input.

    Action (12 dims), in [-1, 1]: offsets from the nominal joint pose, scaled by
    ``action_scale`` and fed to a joint-space PD controller. The policy sets
    *targets*, not torques - the same interface a real Go2 exposes.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Foot ordering used consistently across gait.py, rewards.py and this file.
    FOOT_GEOM_NAMES = ("FL", "FR", "RL", "RR")

    # Observation scales. Chosen so each block lands in roughly [-1, 1] for
    # typical motion: joint velocities reach ~20 rad/s, angular rates ~4 rad/s.
    OBS_SCALE_ANG_VEL = 0.25
    OBS_SCALE_LIN_VEL = 2.0
    OBS_SCALE_JOINT_VEL = 0.05
    OBS_SCALE_CMD = np.array([2.0, 2.0, 0.25], dtype=np.float64)

    # World-frame gravity direction, rotated into the base frame each step.
    _GRAVITY_DIR = np.array([0.0, 0.0, -1.0])

    def __init__(
        self,
        xml_path="mujoco_menagerie/unitree_go2/scene.xml",
        decimation=10,
        action_scale=0.30,
        kp=55.0,
        kd=1.4,
        max_episode_steps=1000,
        # Termination
        min_base_height=0.12,
        max_pitch_roll=0.8,
        # Reset randomisation
        reset_noise_scale=0.1,
        # Commands
        command_ranges=None,
        stand_probability=0.10,
        command_resample_interval_s=5.0,
        gaits=("trot",),
        # Reward
        reward_weights=None,
        lin_vel_sigma=0.20,
        ang_vel_sigma=0.25,
        # Robustness
        randomize_dynamics=False,
        push_enabled=True,
        push_interval_s=(3.0, 7.0),
        push_duration_s=0.15,
        push_force=(-40.0, 40.0),
        obs_noise_scale=0.0,
        render_mode=None,
        render_size=(640, 480),
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.decimation = int(decimation)
        self.physics_dt = float(self.model.opt.timestep)
        self.dt = self.physics_dt * self.decimation  # control period (B7: 0.02 s)
        self.metadata["render_fps"] = int(round(1.0 / self.dt))

        self.action_scale = float(action_scale)
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_episode_steps = int(max_episode_steps)
        self.min_base_height = float(min_base_height)
        self.max_pitch_roll = float(max_pitch_roll)
        self.reset_noise_scale = float(reset_noise_scale)
        self.randomize_dynamics = bool(randomize_dynamics)
        self.obs_noise_scale = float(obs_noise_scale)
        self.render_mode = render_mode
        self.render_size = tuple(render_size)

        self.push_enabled = bool(push_enabled)
        self.push_interval_s = tuple(push_interval_s)
        self.push_duration_s = float(push_duration_s)
        self.push_force = tuple(push_force)

        # ---- B1: nominal pose from the 'home' keyframe -------------------- #
        self._home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if self._home_key_id < 0:
            raise RuntimeError(
                "The MJCF has no 'home' keyframe. v2 relies on it for the nominal "
                "standing pose; without it the policy is asked to walk out of a "
                "pose the robot cannot stand in (see docs/14-debugging-log.md, B1)."
            )
        self.default_joint_pos = self.model.key_qpos[self._home_key_id, 7:].copy()
        self.nominal_base_height = float(self.model.key_qpos[self._home_key_id, 2])

        # ---- Indices ------------------------------------------------------ #
        self.n_joints = self.model.nu
        self.actuated_joint_ids = self.model.actuator_trnid[:, 0].copy()
        joint_ranges = self.model.jnt_range[self.actuated_joint_ids].copy()
        # Soften by 5% of the travel on each side, so the policy is discouraged
        # from leaning on the hard endstop that MuJoCo enforces for it.
        span = joint_ranges[:, 1] - joint_ranges[:, 0]
        self.soft_joint_limits = np.stack(
            [joint_ranges[:, 0] + 0.05 * span, joint_ranges[:, 1] - 0.05 * span],
            axis=1,
        )
        self.torque_limits = self.model.actuator_ctrlrange[:, 1].copy()

        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        if self.floor_geom_id < 0:
            raise RuntimeError("scene.xml is missing a geom named 'floor'.")

        self.foot_geom_ids = []
        for name in self.FOOT_GEOM_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise RuntimeError(
                    "Foot geom %r not found. v1 silently fell back to 'the last "
                    "four geoms', which is how you end up rewarding contacts on "
                    "the wrong bodies." % name
                )
            self.foot_geom_ids.append(gid)
        self.foot_geom_ids = np.array(self.foot_geom_ids, dtype=np.int32)

        # B10: everything on the robot that is NOT a foot. Ground contact with
        # any of these is a collision (knee, shin, hip, trunk).
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        foot_set = set(self.foot_geom_ids.tolist())
        self.non_foot_geom_ids = np.array(
            [
                g
                for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] != 0
                and g not in foot_set
                and self.model.geom_contype[g] != 0
            ],
            dtype=np.int32,
        )
        # Trunk-only subset: touching the ground with these ends the episode.
        self.trunk_geom_ids = np.array(
            [g for g in self.non_foot_geom_ids if self.model.geom_bodyid[g] == self.base_body_id],
            dtype=np.int32,
        )

        # ---- Commands and gait -------------------------------------------- #
        self.command_sampler = CommandSampler(
            ranges=command_ranges or CommandRanges(),
            stand_probability=stand_probability,
            resample_interval_s=command_resample_interval_s,
            gaits=tuple(gaits),
        )
        self.command = Command()
        self.gait_phase = 0.0
        self._gait_offsets, self._gait_duty = gait_mod.gait_params(self.command.gait)

        # ---- Reward -------------------------------------------------------- #
        self.reward_weights = reward_mod.resolve_weights(reward_weights)
        self.lin_vel_sigma = float(lin_vel_sigma)
        self.ang_vel_sigma = float(ang_vel_sigma)

        # ---- Domain randomisation baselines -------------------------------- #
        self.default_geom_friction = self.model.geom_friction.copy()
        self.default_body_mass = self.model.body_mass.copy()
        self._kp_scale = 1.0
        self._kd_scale = 1.0

        # ---- Spaces --------------------------------------------------------- #
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
        self.obs_dim = 3 + 3 + 3 + self.n_joints * 3 + 3 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # ---- Episode state -------------------------------------------------- #
        self.step_count = 0
        self.prev_action = np.zeros(self.n_joints)
        self._current_action = np.zeros(self.n_joints)
        self._prev_joint_vel = np.zeros(self.n_joints)
        self._last_torques = np.zeros(self.n_joints)
        self._feet_air_time = np.zeros(4)
        self._last_contact = np.zeros(4)
        self._time_since_command = 0.0
        self._push_countdown = 0.0
        self._push_remaining = 0.0
        self._reward_terms = {}
        self._tracking_scores = []

        # Per-step frame-conversion cache (see _refresh_frame_cache).
        self._lin_vel_b = np.zeros(3)
        self._ang_vel_b = np.zeros(3)
        self._proj_gravity = self._GRAVITY_DIR.copy()

        self._viewer = None
        self._renderer = None  # B9: created once, reused
        self._camera = None

    # ------------------------------------------------------------------ #
    #  Public control surface (used by callbacks, play.py, evaluate.py)   #
    # ------------------------------------------------------------------ #

    def set_command_ranges(self, ranges):
        """Replace the sampler's ranges. Called by the curriculum callback."""
        if isinstance(ranges, dict):
            ranges = CommandRanges(**ranges)
        self.command_sampler.ranges = ranges

    def set_command(self, lin_vel_x=None, lin_vel_y=None, ang_vel_yaw=None,
                    gait=None, gait_frequency=None, base_height=None):
        """Override the live command. Used for teleop and for evaluation grids."""
        c = self.command
        if lin_vel_x is not None:
            c.lin_vel_x = float(lin_vel_x)
        if lin_vel_y is not None:
            c.lin_vel_y = float(lin_vel_y)
        if ang_vel_yaw is not None:
            c.ang_vel_yaw = float(ang_vel_yaw)
        if gait_frequency is not None:
            c.gait_frequency = float(gait_frequency)
        if base_height is not None:
            c.base_height = float(base_height)
        if gait is not None:
            c.gait = str(gait)
        self._apply_gait(c.gait)
        self._time_since_command = 0.0

    def set_cmd_vel(self, cmd_vel):
        """Backwards-compatible shim for v1 scripts and the old callback API."""
        self.set_command(lin_vel_x=cmd_vel[0], lin_vel_y=cmd_vel[1],
                         ang_vel_yaw=cmd_vel[2])

    def get_command(self):
        return self.command

    def _last_tracking_score(self):
        """Mean tracking score since the last call, then reset the accumulator.

        Exists so a trainer can read the curriculum signal with one broadcast
        ``env_method`` call per rollout instead of inspecting every info dict.
        Returns None when no steps have been taken since the last call.
        """
        if not self._tracking_scores:
            return None
        score = float(np.mean(self._tracking_scores))
        self._tracking_scores.clear()
        return score

    def _apply_gait(self, name):
        self._gait_offsets, self._gait_duty = gait_mod.gait_params(name)

    # ------------------------------------------------------------------ #
    #  Kinematic helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cross3(a0, a1, a2, b0, b1, b2):
        """Explicit 3-vector cross product.

        This exists purely for speed, and the speed is not marginal. Profiling
        the step loop showed ``np.cross`` consuming more wall-clock time than
        the MuJoCo physics itself: it is a fully general n-dimensional routine
        that runs ``moveaxis`` and ``normalize_axis_tuple`` on every call, and
        we call it fourteen times per control step on 3-element vectors. Writing
        out the six multiplications roughly doubles environment throughput.
        """
        return (a1 * b2 - a2 * b1,
                a2 * b0 - a0 * b2,
                a0 * b1 - a1 * b0)

    @staticmethod
    def quat_rotate_inverse(quat, vec):
        """Express a world-frame vector in the body frame. wxyz convention.

        Identity used:  q* v q  =  v - 2w (u x v) + 2 u x (u x v),  u = (x,y,z).
        Verified against MuJoCo's own mju_rotVecQuat in tests/test_math.py -
        this helper was correct in v1 too; the bug was that it was never applied
        to the velocities (B2).
        """
        w = float(quat[0])
        ux, uy, uz = float(quat[1]), float(quat[2]), float(quat[3])
        vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
        tx, ty, tz = Go2Env._cross3(ux, uy, uz, vx, vy, vz)
        tx, ty, tz = 2.0 * tx, 2.0 * ty, 2.0 * tz
        cx, cy, cz = Go2Env._cross3(ux, uy, uz, tx, ty, tz)
        return np.array([vx - w * tx + cx, vy - w * ty + cy, vz - w * tz + cz])

    # The three frame conversions below are recomputed once per control step and
    # cached, rather than on every access. The observation, the reward state and
    # the termination check all want them, and recomputing meant seven quaternion
    # rotations per step where three suffice.

    def _refresh_frame_cache(self):
        q = self.data.qpos[3:7]
        self._lin_vel_b = self.quat_rotate_inverse(q, self.data.qvel[0:3])
        self._ang_vel_b = self.quat_rotate_inverse(q, self.data.qvel[3:6])
        self._proj_gravity = self.quat_rotate_inverse(q, self._GRAVITY_DIR)

    def _base_lin_vel(self):
        return self._lin_vel_b

    def _base_ang_vel(self):
        return self._ang_vel_b

    def _projected_gravity(self):
        return self._proj_gravity

    # ------------------------------------------------------------------ #
    #  Contacts (B10)                                                     #
    # ------------------------------------------------------------------ #

    def _contact_state(self, force_threshold=1.0):
        """Return ``(foot_contacts, undesired_contact_count, trunk_touching)``.

        A foot counts as loaded only when it is in contact *with the floor* and
        the normal force exceeds a threshold. v1 flagged a foot on any contact
        involving its geom, including foot-against-shin self-contact, and used
        the mere existence of a contact pair rather than whether it carried
        load - so a foot grazing the ground scored the same as a foot bearing
        the robot's weight.
        """
        contacts = np.zeros(4)
        undesired = 0.0
        trunk = False
        foot_ids = self.foot_geom_ids
        force = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.floor_geom_id:
                other = g2
            elif g2 == self.floor_geom_id:
                other = g1
            else:
                continue  # robot-on-robot contact; not a ground contact
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if abs(force[0]) < force_threshold:
                continue
            hit = np.where(foot_ids == other)[0]
            if hit.size:
                contacts[hit[0]] = 1.0
            else:
                undesired += 1.0
                if other in self.trunk_geom_ids:
                    trunk = True
        return contacts, undesired, trunk

    def _feet_height(self):
        """Height of each foot geom above the ground plane, world frame."""
        return self.data.geom_xpos[self.foot_geom_ids][:, 2].copy()

    def _feet_velocity_xy(self):
        """Horizontal velocity of each foot geom, world frame. Shape (4, 2)."""
        out = np.zeros((4, 2))
        res = np.zeros(6)
        for i, gid in enumerate(self.foot_geom_ids):
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, int(gid), res, 0
            )
            out[i] = res[3:5]  # res is [angular(3), linear(3)]
        return out

    # ------------------------------------------------------------------ #
    #  Observation                                                        #
    # ------------------------------------------------------------------ #

    def _get_obs(self):
        joint_pos = self.data.qpos[7:] - self.default_joint_pos
        joint_vel = self.data.qvel[6:]
        phase_rad = 2.0 * np.pi * self.gait_phase

        obs = np.concatenate([
            self._projected_gravity(),
            self._base_ang_vel() * self.OBS_SCALE_ANG_VEL,
            self._base_lin_vel() * self.OBS_SCALE_LIN_VEL,
            joint_pos,
            joint_vel * self.OBS_SCALE_JOINT_VEL,
            self.prev_action,
            self.command.vec * self.OBS_SCALE_CMD,
            [np.sin(phase_rad), np.cos(phase_rad)],
        ])

        if self.obs_noise_scale > 0.0:
            obs = obs + self.np_random.normal(0.0, self.obs_noise_scale, obs.shape)
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Reward                                                             #
    # ------------------------------------------------------------------ #

    def _build_reward_state(self, contacts, undesired, first_contact, air_time):
        desired = gait_mod.desired_contact(
            self.gait_phase, self._gait_offsets, self._gait_duty
        )
        return reward_mod.RewardState(
            lin_vel_b=self._base_lin_vel(),
            ang_vel_b=self._base_ang_vel(),
            proj_gravity=self._projected_gravity(),
            base_height=float(self.data.qpos[2]),
            joint_pos=self.data.qpos[7:].copy(),
            joint_vel=self.data.qvel[6:].copy(),
            joint_vel_prev=self._prev_joint_vel.copy(),
            default_joint_pos=self.default_joint_pos,
            soft_joint_limits=self.soft_joint_limits,
            torques=self._last_torques.copy(),
            action=self._current_action.copy(),
            prev_action=self.prev_action.copy(),
            contact=contacts,
            desired_contact=desired,
            feet_air_time=air_time,
            feet_first_contact=first_contact,
            feet_vel_xy=self._feet_velocity_xy(),
            feet_height=self._feet_height(),
            cmd=self.command.vec,
            cmd_base_height=self.command.base_height,
            undesired_contacts=undesired,
            dt=self.dt,
            lin_vel_sigma=self.lin_vel_sigma,
            ang_vel_sigma=self.ang_vel_sigma,
        )

    # ------------------------------------------------------------------ #
    #  Randomisation                                                      #
    # ------------------------------------------------------------------ #

    def _randomize_domain(self):
        # Always restore the PD scales, even when randomisation is off, so a
        # disabled flag really means "nominal dynamics".
        self._kp_scale = 1.0
        self._kd_scale = 1.0
        if not self.randomize_dynamics:
            return

        # Ground and foot friction. Sampled from the model defaults each reset
        # so the randomisation cannot compound across episodes.
        friction_scale = self.np_random.uniform(0.4, 1.4)
        for gid in list(self.foot_geom_ids) + [self.floor_geom_id]:
            self.model.geom_friction[gid] = (
                self.default_geom_friction[gid] * friction_scale
            )

        # Payload on the trunk: a real robot carries a battery, a lidar, a bag.
        self.model.body_mass[:] = self.default_body_mass
        self.model.body_mass[self.base_body_id] = (
            self.default_body_mass[self.base_body_id]
            + self.np_random.uniform(-1.0, 2.0)
        )

        # Actuator gains: stands in for unmodelled motor and driver dynamics.
        self._kp_scale = float(self.np_random.uniform(0.8, 1.2))
        self._kd_scale = float(self.np_random.uniform(0.8, 1.2))

    def _schedule_push(self):
        lo, hi = self.push_interval_s
        self._push_countdown = float(self.np_random.uniform(lo, hi))
        self._push_remaining = 0.0

    def _update_push(self):
        """B11: a real force, for a real duration, at an unpredictable time."""
        if not self.push_enabled:
            return
        if self._push_remaining > 0.0:
            self._push_remaining -= self.dt
            if self._push_remaining <= 0.0:
                self.data.xfrc_applied[self.base_body_id, :3] = 0.0
                self._schedule_push()
            return

        self._push_countdown -= self.dt
        if self._push_countdown <= 0.0:
            f = self.np_random.uniform(self.push_force[0], self.push_force[1], size=3)
            f[2] *= 0.25  # mostly horizontal shoves; vertical yanks are unphysical
            self.data.xfrc_applied[self.base_body_id, :3] = f
            self._push_remaining = self.push_duration_s

    # ------------------------------------------------------------------ #
    #  Gymnasium interface                                                #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # B1: start from the standing keyframe, not qpos0.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        # B8: randomise the whole initial condition, not just joint angles. A
        # policy trained from a single start state has never seen the states it
        # will actually visit after a stumble, so it cannot recover from one.
        n = self.reset_noise_scale
        self.data.qpos[7:] += self.np_random.uniform(-n, n, size=self.n_joints)
        self.data.qpos[2] += self.np_random.uniform(-0.02, 0.05)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        roll_pitch = self.np_random.uniform(-0.05, 0.05, size=2)
        self.data.qpos[3:7] = self._euler_to_quat(roll_pitch[0], roll_pitch[1], yaw)
        self.data.qvel[0:3] = self.np_random.uniform(-0.3, 0.3, size=3)
        self.data.qvel[3:6] = self.np_random.uniform(-0.3, 0.3, size=3)
        self.data.qvel[6:] = self.np_random.uniform(-0.5, 0.5, size=self.n_joints)
        self.data.xfrc_applied[:] = 0.0

        self._randomize_domain()
        mujoco.mj_forward(self.model, self.data)
        self._refresh_frame_cache()

        self.command = self.command_sampler.sample(self.np_random)
        self._apply_gait(self.command.gait)
        # Random initial phase: otherwise every episode starts on the same foot
        # and the policy can key off the episode timer instead of the clock.
        self.gait_phase = float(self.np_random.uniform(0.0, 1.0))

        self.step_count = 0
        self.prev_action[:] = 0.0
        self._current_action[:] = 0.0
        self._prev_joint_vel[:] = self.data.qvel[6:]
        self._last_torques[:] = 0.0
        self._feet_air_time[:] = 0.0
        self._last_contact[:] = 0.0
        self._time_since_command = 0.0
        self._reward_terms = {}
        self._tracking_scores.clear()
        self._schedule_push()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._current_action = action.copy()

        target_pos = self.default_joint_pos + action * self.action_scale
        kp = self.kp * self._kp_scale
        kd = self.kd * self._kd_scale

        self._update_push()

        # B6: recompute the PD torque every physics substep. Holding one torque
        # for the whole control period turns the D term into an open-loop
        # constant and is the main source of v1's visible judder.
        for _ in range(self.decimation):
            torque = kp * (target_pos - self.data.qpos[7:]) - kd * self.data.qvel[6:]
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)
        self._last_torques = torque
        self._refresh_frame_cache()

        self.step_count += 1
        self._time_since_command += self.dt

        # Advance the gait clock before scoring, so the schedule the reward
        # checks is the one the policy saw in its observation last step.
        self.gait_phase = gait_mod.advance_phase(
            self.gait_phase, self.command.gait_frequency, self.dt
        )

        contacts, undesired, trunk_touching = self._contact_state()
        first_contact = ((contacts > 0.5) & (self._last_contact < 0.5)).astype(float)
        air_time_at_landing = self._feet_air_time * first_contact
        self._feet_air_time = np.where(contacts > 0.5, 0.0, self._feet_air_time + self.dt)
        self._last_contact = contacts

        state = self._build_reward_state(
            contacts, undesired, first_contact, air_time_at_landing
        )
        raw_reward, terms = reward_mod.compute(state, self.reward_weights)
        reward = float(raw_reward) * self.dt
        self._reward_terms = {k: float(v) * self.dt for k, v in terms.items()}

        terminated = self._check_termination(trunk_touching)
        truncated = self.step_count >= self.max_episode_steps

        if self.command_sampler.should_resample(self._time_since_command):
            self.command = self.command_sampler.sample(self.np_random)
            self._apply_gait(self.command.gait)
            self._time_since_command = 0.0

        self._prev_joint_vel[:] = self.data.qvel[6:]
        self.prev_action = action.copy()
        obs = self._get_obs()

        info = self._build_info(state, contacts)

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _build_info(self, state, contacts):
        """Diagnostics. Keys prefixed ``rew/`` are picked up by the logger."""
        lin_err = float(np.linalg.norm(state.cmd[:2] - state.lin_vel_b[:2]))
        ang_err = float(abs(state.cmd[2] - state.ang_vel_b[2]))
        info = {
            "x_position": float(self.data.qpos[0]),
            "base_height": float(self.data.qpos[2]),
            "lin_vel_b": state.lin_vel_b.copy(),
            "ang_vel_b": state.ang_vel_b.copy(),
            "cmd": state.cmd.copy(),
            "gait": self.command.gait,
            "gait_phase": self.gait_phase,
            "contacts": contacts.copy(),
            "tracking_lin_err": lin_err,
            "tracking_ang_err": ang_err,
            # Normalised in [0, 1]; this is the signal the adaptive curriculum
            # closes its loop on.
            "tracking_score": float(
                np.exp(-lin_err ** 2 / self.lin_vel_sigma) * 0.5
                + np.exp(-ang_err ** 2 / self.ang_vel_sigma) * 0.5
            ),
        }
        self._tracking_scores.append(info["tracking_score"])
        for k, v in self._reward_terms.items():
            info["rew/" + k] = v
        return info

    def _check_termination(self, trunk_touching):
        if trunk_touching:
            return True
        if self.data.qpos[2] < self.min_base_height:
            return True
        # Attitude check via projected gravity: upright means the z component is
        # -1. No trig, no gimbal edge cases, no yaw dependence.
        gz = self._projected_gravity()[2]
        return bool(gz > -np.cos(self.max_pitch_roll))

    @staticmethod
    def _euler_to_quat(roll, pitch, yaw):
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ])

    # ------------------------------------------------------------------ #
    #  Rendering (B9)                                                     #
    # ------------------------------------------------------------------ #

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                w, h = self.render_size
                self._renderer = mujoco.Renderer(self.model, height=h, width=w)
                self._camera = mujoco.MjvCamera()
                self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._camera.trackbodyid = self.base_body_id
                self._camera.distance = 2.0
                self._camera.azimuth = 135
                self._camera.elevation = -20
            self._renderer.update_scene(self.data, camera=self._camera)
            return self._renderer.render()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
