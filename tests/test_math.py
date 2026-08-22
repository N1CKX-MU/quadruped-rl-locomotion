"""Frame-transform tests.

The v1 environment had a *correct* quaternion helper that it simply never
applied to the velocities (bug B2). That is the worst kind of bug: the code
looks right in isolation. These tests pin the helper against MuJoCo's own
implementation so that if anyone "simplifies" it later the failure is loud.
"""

import numpy as np
import mujoco
import pytest

from envs.go2_env import Go2Env


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_quat_rotate_inverse_matches_mujoco():
    """quat_rotate_inverse(q, v) must equal rotating v by the conjugate of q."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = random_quat(rng)
        v = rng.normal(size=3)

        conj = np.array([q[0], -q[1], -q[2], -q[3]])
        expected = np.zeros(3)
        mujoco.mju_rotVecQuat(expected, v, conj)

        got = Go2Env.quat_rotate_inverse(q, v)
        assert np.allclose(got, expected, atol=1e-9), (q, v, got, expected)


def test_identity_quaternion_is_a_no_op():
    v = np.array([1.0, -2.0, 3.0])
    assert np.allclose(Go2Env.quat_rotate_inverse(np.array([1.0, 0, 0, 0]), v), v)


def test_projected_gravity_is_yaw_invariant():
    """The whole reason we feed projected gravity instead of the quaternion.

    Yaw is a commanded quantity in this task, so the attitude signal handed to
    the policy must not encode absolute heading. Projected gravity does not;
    the raw quaternion v1 used does.
    """
    g = np.array([0.0, 0.0, -1.0])
    for yaw in np.linspace(-np.pi, np.pi, 17):
        q = Go2Env._euler_to_quat(0.0, 0.0, yaw)
        assert np.allclose(Go2Env.quat_rotate_inverse(q, g), g, atol=1e-12)


@pytest.mark.parametrize("roll,pitch", [(0.3, 0.0), (0.0, -0.4), (0.2, 0.1)])
def test_projected_gravity_tilts_the_expected_way(roll, pitch):
    """Rolling right pushes gravity's y component; pitching pushes x."""
    g = np.array([0.0, 0.0, -1.0])
    pg = Go2Env.quat_rotate_inverse(Go2Env._euler_to_quat(roll, pitch, 0.0), g)
    assert np.isclose(np.linalg.norm(pg), 1.0)
    # Upright is (0, 0, -1); any tilt moves magnitude into the xy plane.
    assert np.linalg.norm(pg[:2]) > 1e-6
    assert pg[2] < 0.0  # still the right way up for these small angles


def _rx(a):
    return np.array([[1, 0, 0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a), np.cos(a)]])


def _ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)],
                     [0, 1, 0],
                     [-np.sin(a), 0, np.cos(a)]])


def _rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a), np.cos(a), 0],
                     [0, 0, 1]])


def test_euler_to_quat_is_intrinsic_zyx():
    """Pin the convention explicitly: R = Rz(yaw) Ry(pitch) Rx(roll).

    Euler conventions are the classic silent sign bug - every library picks a
    different one, and a wrong choice here would rotate the reset attitude
    noise in a way nothing else would catch. MuJoCo spells this same convention
    'XYZ' (uppercase = intrinsic), which is checked alongside so the test does
    not just restate our own arithmetic.
    """
    rng = np.random.default_rng(1)
    for _ in range(100):
        roll, pitch, yaw = rng.uniform(-1.0, 1.0, size=3)
        q = Go2Env._euler_to_quat(roll, pitch, yaw)
        assert np.isclose(np.linalg.norm(q), 1.0)

        mat_ours = np.zeros(9)
        mujoco.mju_quat2Mat(mat_ours, q)
        expected = _rz(yaw) @ _ry(pitch) @ _rx(roll)
        assert np.allclose(mat_ours.reshape(3, 3), expected, atol=1e-12)

        euler_q = np.zeros(4)
        mujoco.mju_euler2Quat(euler_q, np.array([roll, pitch, yaw]), "XYZ")
        mat_mj = np.zeros(9)
        mujoco.mju_quat2Mat(mat_mj, euler_q)
        assert np.allclose(mat_ours, mat_mj, atol=1e-9)
