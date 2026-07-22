"""Custom Gymnasium environment for the Unitree Go2 quadruped robot.

Implements a multi-term reward function encouraging natural walking gaits,
command velocity tracking, and robust locomotion.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class Go2Env(gym.Env):
    """Unitree Go2 quadruped locomotion environment.

    Observation space (49 dims):
        - Base orientation quaternion (4)
        - Base angular velocity (3)
        - Base linear velocity (3)
        - Joint positions (12)
        - Joint velocities (12)
        - Previous actions (12)
        - Foot contact forces (4) - binary
        - Command velocity (vx, vy, yaw_rate) (3) - target

    Action space (12 dims):
        Position target deltas from default standing pose, normalized to [-1, 1]
        and scaled by action_scale (default 0.3 rad).

    Reward (8 terms):
        + linear velocity tracking (exp penalty on error)
        + angular velocity tracking (yaw rate)
        + alive bonus
        - lateral velocity penalty
        - body orientation penalty (stay upright)
        - action rate penalty (smooth motions)
        - joint torque penalty (energy efficiency)
        + foot contact regularity (alternating gait)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    # Foot geom names in the Go2 MJCF (MuJoCo Menagerie convention)
    FOOT_GEOM_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    def __init__(
        self,
        xml_path="mujoco_menagerie/unitree_go2/scene.xml",
        frame_skip=20,
        cmd_vel=(0.5, 0.0, 0.0),
        action_scale=0.5,
        healthy_z_range=(0.15, 0.6),
        max_pitch_roll=1.0,
        reset_noise_scale=0.05,
        max_episode_steps=1000,
        randomize_dynamics=False,
        render_mode=None,
    ):
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.healthy_z_range = healthy_z_range
        self.max_pitch_roll = max_pitch_roll
        self.reset_noise_scale = reset_noise_scale
        self.max_episode_steps = max_episode_steps
        self.randomize_dynamics = randomize_dynamics
        self.render_mode = render_mode

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep * self.frame_skip

        # Store default standing pose for action offsets
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.default_joint_pos = self.data.qpos[7:].copy()

        # Command velocity target [vx, vy, yaw_rate]
        self.cmd_vel = np.array(cmd_vel, dtype=np.float32)

        # Foot geom IDs for contact detection
        self.foot_geom_ids = []
        for name in self.FOOT_GEOM_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.foot_geom_ids.append(gid)
        # Fallback: if names not found, use last 4 geoms
        if len(self.foot_geom_ids) != 4:
            self.foot_geom_ids = list(range(self.model.ngeom - 4, self.model.ngeom))

        # Store default dynamics for domain randomization
        self.default_friction = self.model.geom_friction.copy()
        self.default_mass = self.model.body_mass.copy()

        # Spaces
        self.n_actuators = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actuators,), dtype=np.float32
        )
        # 4 + 3 + 3 + 12 + 12 + 12 + 4 + 3 = 53
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32
        )

        # Internal state
        self.prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self.step_count = 0

        # Renderer
        self._viewer = None

    def set_cmd_vel(self, cmd_vel):
        """Update the command velocity target. Used by curriculum callback."""
        self.cmd_vel = np.array(cmd_vel, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Observation                                                        #
    # ------------------------------------------------------------------ #

    def _get_obs(self):
        quat = self.data.qpos[3:7].copy()          # 4: base orientation
        ang_vel = self.data.qvel[3:6].copy()        # 3: base angular velocity
        lin_vel = self.data.qvel[0:3].copy()        # 3: base linear velocity
        joint_pos = self.data.qpos[7:].copy()       # 12: joint positions
        joint_vel = self.data.qvel[6:].copy()       # 12: joint velocities
        contacts = self._get_foot_contacts()        # 4: binary foot contacts
        return np.concatenate([
            quat, ang_vel, lin_vel,
            joint_pos, joint_vel,
            self.prev_action,
            contacts,
            self.cmd_vel,
        ]).astype(np.float32)

    def _get_foot_contacts(self):
        """Return binary contact indicators for each foot."""
        contacts = np.zeros(4, dtype=np.float32)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for foot_idx, gid in enumerate(self.foot_geom_ids):
                if c.geom1 == gid or c.geom2 == gid:
                    contacts[foot_idx] = 1.0
        return contacts

    # ------------------------------------------------------------------ #
    #  Reward (8 terms)                                                   #
    # ------------------------------------------------------------------ #

    def _compute_reward(self):
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]

        # 1. Linear velocity tracking (dominant positive signal)
        # Direct reward for forward velocity, clipped to target
        forward_vel = min(base_lin_vel[0], self.cmd_vel[0])
        r_lin_vel = max(0.0, forward_vel) * 3.0

        # 2. Angular velocity tracking (yaw)
        ang_vel_error = (self.cmd_vel[2] - base_ang_vel[2]) ** 2
        r_ang_vel = math.exp(-ang_vel_error / 0.25) * 0.5

        # 3. Alive bonus (small — must not dominate velocity)
        r_alive = 0.1

        # 4. Lateral velocity penalty (don't crab-walk)
        r_lateral = -abs(base_lin_vel[1]) * 0.5

        # 5. Body orientation penalty (stay upright via projected gravity)
        quat = self.data.qpos[3:7]
        projected_gravity = self._quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        r_orient = -np.sum(np.square(projected_gravity[:2])) * 1.0

        # 6. Action rate penalty (smooth motions)
        action_diff = self.prev_action - self._current_action
        r_action_rate = -np.sum(np.square(action_diff)) * 0.01

        # 7. Joint torque penalty (energy efficiency)
        torques = self.data.qfrc_actuator[6:] if len(self.data.qfrc_actuator) > 6 else self.data.ctrl
        r_torque = -np.sum(np.square(torques)) * 0.0001

        # 8. Foot contact regularity (encourage alternating diagonal pairs)
        r_contact = self._gait_reward() * 0.1

        reward = (r_lin_vel + r_ang_vel + r_alive +
                  r_lateral + r_orient + r_action_rate +
                  r_torque + r_contact)

        self._reward_components = {
            "r_lin_vel": r_lin_vel,
            "r_ang_vel": r_ang_vel,
            "r_alive": r_alive,
            "r_lateral": r_lateral,
            "r_orient": r_orient,
            "r_action_rate": r_action_rate,
            "r_torque": r_torque,
            "r_contact": r_contact,
        }

        return reward

    def _gait_reward(self):
        """Reward alternating diagonal foot contacts (trot gait)."""
        contacts = self._get_foot_contacts()
        # Trot: FL+RR and FR+RL should alternate
        # Reward when diagonal pairs match
        diag1 = contacts[0] * contacts[3]  # FL & RR
        diag2 = contacts[1] * contacts[2]  # FR & RL
        # Reward if one diagonal pair is active and the other is not
        return abs(diag1 - diag2)

    @staticmethod
    def _quat_rotate_inverse(quat, vec):
        """Rotate a vector by the inverse of a quaternion (wxyz convention)."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Conjugate quaternion rotation: q* v q
        t = 2.0 * np.cross(np.array([x, y, z]), vec)
        return vec - w * t + np.cross(np.array([x, y, z]), t)

    # ------------------------------------------------------------------ #
    #  Termination                                                        #
    # ------------------------------------------------------------------ #

    def _check_termination(self):
        z = self.data.qpos[2]
        if z < self.healthy_z_range[0] or z > self.healthy_z_range[1]:
            return True

        # Check pitch and roll from quaternion
        quat = self.data.qpos[3:7]
        # Convert quaternion to euler (approximate pitch/roll)
        w, x, y, z_q = quat[0], quat[1], quat[2], quat[3]
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z_q)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z_q * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = math.asin(sinp)

        if abs(roll) > self.max_pitch_roll or abs(pitch) > self.max_pitch_roll:
            return True

        return False

    # ------------------------------------------------------------------ #
    #  Domain randomization                                               #
    # ------------------------------------------------------------------ #

    def _randomize_domain(self):
        """Randomize friction and mass at episode reset for robustness."""
        if not self.randomize_dynamics:
            return

        # Friction: +/- 20%
        for gid in self.foot_geom_ids:
            scale = self.np_random.uniform(0.8, 1.2)
            self.model.geom_friction[gid] = self.default_friction[gid] * scale

        # Mass: +/- 10%
        for bid in range(self.model.nbody):
            scale = self.np_random.uniform(0.9, 1.1)
            self.model.body_mass[bid] = self.default_mass[bid] * scale

    def _apply_external_push(self):
        """Occasionally apply random push for robustness."""
        if self.step_count > 0 and self.step_count % 200 == 0:
            push = self.np_random.uniform(-3.0, 3.0, size=3)
            self.data.qvel[0:3] += push

    # ------------------------------------------------------------------ #
    #  Core Gymnasium interface                                           #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Add small noise to initial joint positions only
        noise = self.np_random.uniform(
            -self.reset_noise_scale, self.reset_noise_scale, size=self.n_actuators
        )
        self.data.qpos[7:] = self.default_joint_pos + noise

        self._randomize_domain()
        mujoco.mj_forward(self.model, self.data)

        self.prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self._current_action = np.zeros(self.n_actuators, dtype=np.float32)
        self._reward_components = {}
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        self._current_action = action.copy()

        # Convert normalized action to joint position targets
        target = self.default_joint_pos + action * self.action_scale
        self.data.ctrl[:] = target

        # Apply occasional external push
        self._apply_external_push()

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "x_position": self.data.qpos[0],
            "x_velocity": self.data.qvel[0],
            "y_velocity": self.data.qvel[1],
            "body_height": self.data.qpos[2],
            **self._reward_components,
        }

        self.prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            img = renderer.render()
            renderer.close()
            return img

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
