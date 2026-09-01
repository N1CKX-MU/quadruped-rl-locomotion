"""Environment-level tests.

The headline one is `test_nominal_pose_is_a_standing_pose`. That is the direct
regression test for bug B1, the defect that capped v1's performance: the
"default pose" was `qpos0`, all joints at zero, which is not a pose the robot
can stand in. Everything else in this file is scaffolding around that.
"""

import os

import numpy as np
import pytest

from envs.commands import CommandRanges
from envs.go2_env import Go2Env

XML = "mujoco_menagerie/unitree_go2/scene.xml"

pytestmark = pytest.mark.skipif(
    not os.path.exists(XML),
    reason="mujoco_menagerie is not checked out; run `make setup`",
)


@pytest.fixture(scope="module")
def env():
    e = Go2Env(xml_path=XML, push_enabled=False, randomize_dynamics=False)
    yield e
    e.close()


# --------------------------------------------------------------------------- #
#  B1: the root cause                                                          #
# --------------------------------------------------------------------------- #


def test_nominal_pose_is_a_standing_pose(env):
    """The nominal pose must be the model's `home` keyframe, not qpos0.

    Go2's standing pose is hip 0, thigh +0.9, calf -1.8 on every leg. v1 used
    qpos0 - all zeros - so with action_scale 0.5 the calf could only ever reach
    +/-0.5 rad, nowhere near the -1.8 it needs. The policy was being asked to
    walk out of a pose it had no authority to leave.
    """
    expected = np.tile([0.0, 0.9, -1.8], 4)
    assert np.allclose(env.default_joint_pos, expected, atol=1e-6)


def test_zero_action_holds_a_stable_stand(env):
    """With action 0 the PD controller targets the nominal pose. If that pose
    is a real standing pose, the robot should simply stand there."""
    env.reset(seed=0)
    # Cancel the reset randomisation for a clean test of the pose itself.
    env.data.qpos[7:] = env.default_joint_pos
    env.data.qpos[2] = env.nominal_base_height
    env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qvel[:] = 0.0

    heights = []
    for _ in range(100):  # 2 s at 50 Hz
        _, _, terminated, _, info = env.step(np.zeros(12))
        heights.append(info["base_height"])
        assert not terminated, "the robot fell over while asked to stand still"

    settled = np.array(heights[50:])
    assert settled.min() > 0.2, settled.min()
    assert settled.std() < 0.02, "standing height is oscillating, PD gains are off"


def test_v1_nominal_pose_would_not_stand():
    """Demonstrates the bug rather than just asserting the fix.

    Drive the same PD controller toward the all-zeros pose v1 used and the robot
    collapses within two seconds. This test is what makes B1 a fact rather than
    an opinion, and it is quoted in docs/14-debugging-log.md.
    """
    env = Go2Env(xml_path=XML, push_enabled=False, randomize_dynamics=False)
    try:
        env.reset(seed=0)
        env.default_joint_pos = np.zeros(12)  # v1's qpos0 pose
        env.data.qpos[2] = 0.30
        env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        env.data.qpos[7:] = 0.0
        env.data.qvel[:] = 0.0

        fell = False
        for _ in range(100):
            _, _, terminated, _, _ = env.step(np.zeros(12))
            if terminated:
                fell = True
                break
        assert fell, "expected the all-zeros pose to collapse"
    finally:
        env.close()


# --------------------------------------------------------------------------- #
#  Spaces, shapes, plumbing                                                    #
# --------------------------------------------------------------------------- #


def test_model_path_resolves_from_any_working_directory():
    """Scripts must run when invoked by absolute path from anywhere.

    The default xml_path is relative, so before this the whole repository was
    only usable with the repo root as cwd; anywhere else you got MuJoCo's
    "ParseXML: Error opening file", which names neither the file nor the reason.
    """
    import os

    from envs.go2_env import resolve_asset_path

    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(cwd)) or os.sep)
        resolved = resolve_asset_path(XML)
        assert os.path.exists(resolved), resolved
        assert os.path.isabs(resolved)
    finally:
        os.chdir(cwd)


def test_missing_model_says_where_it_looked():
    from envs.go2_env import resolve_asset_path

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_asset_path("mujoco_menagerie/does_not_exist/scene.xml")
    message = str(excinfo.value)
    assert "current directory" in message
    assert "repository root" in message
    assert "mujoco_menagerie.git" in message   # tells you how to fix it


def test_reset_never_spawns_a_foot_through_the_floor(env):
    """B21. The keyframe height IS the standing height, so any downward
    perturbation at reset - the height noise, or a joint angle that extends a
    leg - buries a foot in the ground.

    Measured before the fix: 144 of 200 resets penetrated, the deepest by
    6.6 cm. MuJoCo's CPU solver absorbs that silently, so it never raised an
    error; it just began 72% of episodes with a large spurious contact impulse.
    The identical reset under MJX, whose solver runs far fewer iterations,
    diverged to NaN outright.
    """
    worst = 0.0
    for seed in range(100):
        env.reset(seed=seed)
        foot_z = env.data.geom_xpos[env.foot_geom_ids][:, 2] - env.foot_radius
        worst = min(worst, float(foot_z.min()))
    assert worst > -1e-6, "deepest foot penetration at reset: %.5f m" % worst


def test_observation_matches_the_declared_space(env):
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.shape == (50,)
    assert np.all(np.isfinite(obs))


def test_control_rate_is_50hz(env):
    assert env.dt == pytest.approx(0.02)
    assert env.metadata["render_fps"] == 50


def test_random_rollout_stays_finite(env):
    """A thousand random actions must not produce a NaN or an infinite reward."""
    env.reset(seed=2)
    rng = np.random.default_rng(2)
    for _ in range(1000):
        action = rng.uniform(-1, 1, size=12)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        for key, value in info.items():
            if key.startswith("rew/"):
                assert np.isfinite(value), key
        if terminated or truncated:
            env.reset()


def test_reward_terms_match_the_configured_weights(env):
    env.reset(seed=3)
    _, _, _, _, info = env.step(np.zeros(12))
    logged = {k[len("rew/"):] for k in info if k.startswith("rew/")}
    expected = {k for k, w in env.reward_weights.items() if w != 0.0}
    assert logged == expected


def test_action_is_clipped_not_trusted(env):
    """Nothing guarantees a policy respects the box, so the env must clip."""
    env.reset(seed=4)
    env.step(np.full(12, 50.0))
    assert np.all(np.abs(env.prev_action) <= 1.0)


# --------------------------------------------------------------------------- #
#  Commands (B3)                                                               #
# --------------------------------------------------------------------------- #


def test_command_appears_in_the_observation(env):
    env.reset(seed=5)
    env.set_command(lin_vel_x=1.0, lin_vel_y=0.0, ang_vel_yaw=0.0)
    obs_a, _, _, _, _ = env.step(np.zeros(12))
    env.set_command(lin_vel_x=-1.0)
    obs_b, _, _, _, _ = env.step(np.zeros(12))
    # The command block sits at offset 3+3+3+36 = 45.
    assert not np.allclose(obs_a[45:48], obs_b[45:48])


def test_commands_are_resampled_within_an_episode(env):
    """v1 held one command for an entire run, so the policy never had to read
    it. Here it must change mid-episode."""
    env.command_sampler.resample_interval_s = 0.1
    env.reset(seed=6)
    seen = set()
    for _ in range(200):
        env.step(np.zeros(12))
        seen.add(tuple(np.round(env.command.vec, 4)))
        if env.step_count > 150:
            break
    env.command_sampler.resample_interval_s = 5.0
    assert len(seen) > 1


def test_info_reports_the_command_the_reward_used(env):
    """Diagnostics must describe the command the reward was computed against.

    Commands are resampled mid-episode, and the new one goes into the NEXT
    observation. If the info dict were built after the resample it would pair
    the freshly-drawn command with the previous command's reward, which quietly
    corrupts both the curriculum signal and the gait_fraction traces.
    """
    env.command_sampler.resample_interval_s = env.dt   # resample every step
    env.reset(seed=11)
    try:
        for _ in range(20):
            before = env.command.vec.copy()
            _, _, _, _, info = env.step(np.zeros(12))
            assert np.allclose(info["cmd"], before), (info["cmd"], before)
    finally:
        env.command_sampler.resample_interval_s = 5.0


def test_sideways_command_is_expressible(env):
    """Bug B3 made this impossible: vy was untracked and all lateral motion was
    penalised, so a strafe command could never score well."""
    from envs import rewards as R

    env.reset(seed=7)
    env.set_command(lin_vel_x=0.0, lin_vel_y=0.6, ang_vel_yaw=0.0)
    state = env._build_reward_state(np.ones(4), 0.0, np.zeros(4), np.zeros(4))
    state.lin_vel_b = np.array([0.0, 0.6, 0.0])
    assert R.track_lin_vel_xy(state) == pytest.approx(1.0)


def test_set_command_ranges_accepts_the_curriculum_payload(env):
    ranges = CommandRanges(lin_vel_x=(-0.1, 0.2))
    env.set_command_ranges(ranges)
    assert env.command_sampler.ranges.lin_vel_x == (-0.1, 0.2)


# --------------------------------------------------------------------------- #
#  Contacts (B10) and termination                                              #
# --------------------------------------------------------------------------- #


def test_standing_robot_registers_four_foot_contacts(env):
    env.reset(seed=8)
    env.data.qpos[7:] = env.default_joint_pos
    env.data.qpos[2] = env.nominal_base_height
    env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qvel[:] = 0.0
    for _ in range(50):
        _, _, _, _, info = env.step(np.zeros(12))
    assert info["contacts"].sum() == 4.0
    assert info["rew/collision"] == pytest.approx(0.0)


def test_upside_down_terminates(env):
    env.reset(seed=9)
    env.data.qpos[3:7] = Go2Env._euler_to_quat(np.pi, 0.0, 0.0)
    env._refresh_frame_cache()   # the frame conversions are cached per step
    assert env._check_termination(False)


def test_gym_api_compliance(env):
    from stable_baselines3.common.env_checker import check_env

    e = Go2Env(xml_path=XML, push_enabled=False, randomize_dynamics=False,
               max_episode_steps=50)
    try:
        check_env(e, warn=True, skip_render_check=True)
    finally:
        e.close()
