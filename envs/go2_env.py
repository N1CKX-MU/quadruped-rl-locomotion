"""Custom Gymnasium environment for the Unitree Go2 quadruped robot."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class Go2Env(gym.Env):
    """Unitree Go2 quadruped locomotion environment.

    Uses the Go2 MJCF model from MuJoCo Menagerie. The robot is rewarded
    for forward velocity, staying upright, and penalized for excessive
    control effort.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        xml_path="mujoco_menagerie/unitree_go2/scene.xml",
        frame_skip=20,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.05,
        healthy_reward=1.0,
        healthy_z_range=(0.15, 0.6),
        reset_noise_scale=0.1,
        max_episode_steps=1000,
        render_mode=None,
    ):
        self.frame_skip = frame_skip
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_z_range = healthy_z_range
        self.reset_noise_scale = reset_noise_scale
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Set simulation timestep info
        self.dt = self.model.opt.timestep * self.frame_skip

        # Action space: torques for 12 joints (4 legs x 3 joints each)
        self.n_actuators = self.model.nu
        action_low = np.full(self.n_actuators, -1.0, dtype=np.float32)
        action_high = np.full(self.n_actuators, 1.0, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high)

        # Observation space: qpos (excluding free joint xy) + qvel + previous action
        # Free joint: 7 (3 pos + 4 quat), we keep z + quat = 5, plus joint positions
        obs_size = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Renderer
        self._renderer = None
        if render_mode == "human":
            self._renderer = mujoco.viewer.launch_passive(self.model, self.data)

        self._step_count = 0
        self._last_action = np.zeros(self.n_actuators, dtype=np.float32)

    def _get_obs(self):
        """Build observation vector."""
        # Body height (z), orientation quaternion
        body_pos_z = self.data.qpos[2:3]
        body_quat = self.data.qpos[3:7]

        # Joint positions and velocities
        joint_pos = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]

        # Body linear and angular velocity
        body_vel = self.data.qvel[0:3]
        body_angvel = self.data.qvel[3:6]

        obs = np.concatenate([
            body_pos_z,      # 1
            body_quat,       # 4
            joint_pos,       # 12
            body_vel,        # 3
            body_angvel,     # 3
            joint_vel,       # 12
            self._last_action,  # 12
        ]).astype(np.float32)

        return obs

    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        return self.healthy_z_range[0] <= z <= self.healthy_z_range[1]

    def step(self, action):
        # Store previous x position for forward reward
        x_before = self.data.qpos[0]

        # Apply action (scaled to actuator range)
        ctrl = action * self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = ctrl

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        x_after = self.data.qpos[0]
        self._step_count += 1
        self._last_action = action.copy()

        # Rewards
        forward_reward = self.forward_reward_weight * (x_after - x_before) / self.dt
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        healthy_reward = self.healthy_reward if self.is_healthy else 0.0

        reward = forward_reward + healthy_reward - ctrl_cost

        # Termination
        terminated = not self.is_healthy
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "forward_reward": forward_reward,
            "ctrl_cost": ctrl_cost,
            "healthy_reward": healthy_reward,
            "x_position": x_after,
            "x_velocity": (x_after - x_before) / self.dt,
        }

        if self.render_mode == "human" and self._renderer is not None:
            self._renderer.sync()

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Add noise to initial state
        noise_scale = self.reset_noise_scale
        self.data.qpos[:] += self.np_random.uniform(
            low=-noise_scale, high=noise_scale, size=self.model.nq
        )
        self.data.qvel[:] += self.np_random.uniform(
            low=-noise_scale, high=noise_scale, size=self.model.nv
        )

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._last_action = np.zeros(self.n_actuators, dtype=np.float32)

        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            img = renderer.render()
            renderer.close()
            return img

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
