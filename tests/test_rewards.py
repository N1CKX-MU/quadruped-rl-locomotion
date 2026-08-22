"""Reward-term tests.

Each term is a pure function, so each can be pinned at a hand-constructed
state. The point is not to check arithmetic - it is to pin the *sign* and the
*bounds*, because a term with the wrong sign trains perfectly happily and just
produces a robot that does the opposite of what you meant.

Convention under test: penalty-named functions return a NON-NEGATIVE cost, and
the config supplies the negative weight.
"""

import numpy as np
import pytest

from envs import rewards as R
from envs.gait import desired_contact, gait_params


def make_state(**overrides):
    """A neutral, upright, motionless state that every test perturbs."""
    n = 12
    base = dict(
        lin_vel_b=np.zeros(3),
        ang_vel_b=np.zeros(3),
        proj_gravity=np.array([0.0, 0.0, -1.0]),
        base_height=0.30,
        joint_pos=np.zeros(n),
        joint_vel=np.zeros(n),
        joint_vel_prev=np.zeros(n),
        default_joint_pos=np.zeros(n),
        soft_joint_limits=np.stack([np.full(n, -1.0), np.full(n, 1.0)], axis=1),
        torques=np.zeros(n),
        action=np.zeros(n),
        prev_action=np.zeros(n),
        contact=np.ones(4),
        desired_contact=np.ones(4),
        feet_air_time=np.zeros(4),
        feet_first_contact=np.zeros(4),
        feet_vel_xy=np.zeros((4, 2)),
        feet_height=np.full(4, 0.01),
        cmd=np.zeros(3),
        cmd_base_height=0.30,
        undesired_contacts=0.0,
        dt=0.02,
    )
    base.update(overrides)
    return R.RewardState(**base)


# --------------------------------------------------------------------------- #
#  Tracking                                                                   #
# --------------------------------------------------------------------------- #


def test_lin_vel_tracking_is_maximal_on_a_perfect_match():
    s = make_state(cmd=np.array([1.0, 0.3, 0.0]),
                   lin_vel_b=np.array([1.0, 0.3, 0.0]))
    assert R.track_lin_vel_xy(s) == pytest.approx(1.0)


def test_lin_vel_tracking_is_bounded_and_decreasing():
    """Bounded in (0, 1]: this is why an exponential kernel is used instead of
    a negative squared error, which is unbounded below and drowns out every
    other term early in training."""
    prev = 1.0
    for err in np.linspace(0.0, 3.0, 20):
        s = make_state(cmd=np.array([err, 0.0, 0.0]))
        r = R.track_lin_vel_xy(s)
        assert 0.0 < r <= 1.0
        assert r <= prev + 1e-12
        prev = r


def test_lin_vel_tracking_covers_the_lateral_axis():
    """v1's bug B3: only vx was tracked, so a sideways command was unsatisfiable."""
    matched = make_state(cmd=np.array([0.0, 0.8, 0.0]),
                         lin_vel_b=np.array([0.0, 0.8, 0.0]))
    ignored = make_state(cmd=np.array([0.0, 0.8, 0.0]),
                         lin_vel_b=np.zeros(3))
    assert R.track_lin_vel_xy(matched) > R.track_lin_vel_xy(ignored)


def test_yaw_tracking_rewards_turning_on_command():
    turning = make_state(cmd=np.array([0.0, 0.0, 1.0]),
                         ang_vel_b=np.array([0.0, 0.0, 1.0]))
    still = make_state(cmd=np.array([0.0, 0.0, 1.0]))
    assert R.track_ang_vel_yaw(turning) == pytest.approx(1.0)
    assert R.track_ang_vel_yaw(still) < 0.1


# --------------------------------------------------------------------------- #
#  Gait                                                                       #
# --------------------------------------------------------------------------- #


def test_gait_phase_is_one_when_contacts_match_the_schedule():
    offsets, duty = gait_params("trot")
    d = desired_contact(0.1, offsets, duty)
    assert R.gait_phase(make_state(contact=d, desired_contact=d)) == pytest.approx(1.0)


def test_gait_phase_is_zero_when_every_foot_is_wrong():
    offsets, duty = gait_params("trot")
    d = desired_contact(0.1, offsets, duty)
    s = make_state(contact=1.0 - d, desired_contact=d)
    assert R.gait_phase(s) == pytest.approx(0.0)


def test_gait_phase_is_bounded():
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = rng.integers(0, 2, size=4).astype(float)
        d = rng.integers(0, 2, size=4).astype(float)
        assert 0.0 <= R.gait_phase(make_state(contact=c, desired_contact=d)) <= 1.0


def test_feet_air_time_pays_for_long_strides_and_charges_for_hops():
    moving = dict(cmd=np.array([1.0, 0.0, 0.0]),
                  feet_first_contact=np.array([1.0, 0.0, 0.0, 0.0]))
    long_stride = make_state(feet_air_time=np.array([0.8, 0, 0, 0]), **moving)
    short_hop = make_state(feet_air_time=np.array([0.1, 0, 0, 0]), **moving)
    assert R.feet_air_time(long_stride) > 0.0
    assert R.feet_air_time(short_hop) < 0.0


def test_feet_air_time_is_off_when_told_to_stand():
    """Otherwise it bribes a stationary robot into fidgeting."""
    s = make_state(cmd=np.zeros(3),
                   feet_air_time=np.array([0.8, 0, 0, 0]),
                   feet_first_contact=np.array([1.0, 0, 0, 0]))
    assert R.feet_air_time(s) == pytest.approx(0.0)


def test_stand_still_is_the_mirror_image():
    displaced = np.full(12, 0.3)
    standing = make_state(cmd=np.zeros(3), joint_pos=displaced)
    walking = make_state(cmd=np.array([1.0, 0, 0]), joint_pos=displaced)
    assert R.stand_still(standing) > 0.0
    assert R.stand_still(walking) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
#  Regularisation: all of these are costs, so all must be non-negative         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fn,perturbation",
    [
        (R.lin_vel_z, dict(lin_vel_b=np.array([0.0, 0.0, 0.7]))),
        (R.ang_vel_xy, dict(ang_vel_b=np.array([0.5, -0.5, 0.0]))),
        (R.orientation, dict(proj_gravity=np.array([0.3, 0.2, -0.93]))),
        (R.base_height, dict(base_height=0.20)),
        (R.torques, dict(torques=np.full(12, 5.0))),
        (R.joint_acceleration, dict(joint_vel=np.full(12, 1.0))),
        (R.action_rate, dict(action=np.full(12, 0.4))),
        (R.joint_limits, dict(joint_pos=np.full(12, 1.5))),
        (R.collision, dict(undesired_contacts=2.0)),
        (R.feet_slip, dict(feet_vel_xy=np.full((4, 2), 0.5))),
    ],
)
def test_penalties_are_non_negative_and_zero_at_rest(fn, perturbation):
    assert fn(make_state()) == pytest.approx(0.0, abs=1e-12)
    assert fn(make_state(**perturbation)) > 0.0


def test_joint_limits_ignores_the_interior_of_the_range():
    inside = make_state(joint_pos=np.full(12, 0.9))     # limits are +/- 1.0
    outside = make_state(joint_pos=np.full(12, 1.2))
    assert R.joint_limits(inside) == pytest.approx(0.0)
    assert R.joint_limits(outside) == pytest.approx(12 * 0.2)


def test_feet_clearance_is_zero_when_every_foot_should_be_planted():
    """A stand command schedules all four feet in stance, so the term vanishes."""
    s = make_state(desired_contact=np.ones(4), feet_height=np.zeros(4))
    assert R.feet_clearance(s) == pytest.approx(0.0)


def test_feet_clearance_pays_smoothly_for_lifting_a_swing_foot():
    """The whole point of the term: a gradient BEFORE contact breaks.

    gait_phase is a step function of a binary contact flag, so it is flat right
    up to the discontinuity. This one must be strictly decreasing in foot height
    over the approach to the target.
    """
    desired = np.array([1.0, 0.0, 0.0, 1.0])          # two feet swinging
    costs = []
    for h in (0.0, 0.005, 0.01, 0.02, 0.04, 0.08):
        s = make_state(desired_contact=desired, feet_height=np.full(4, h))
        costs.append(R.feet_clearance(s))
    assert all(b < a for a, b in zip(costs, costs[1:])), costs
    assert costs[-1] == pytest.approx(0.0)            # exactly at the target


def test_feet_clearance_only_looks_at_swing_feet():
    desired = np.array([1.0, 1.0, 0.0, 0.0])
    s = make_state(desired_contact=desired, feet_height=np.zeros(4))
    # Only the two swing feet contribute: 2 * (0 - 0.08)^2
    assert R.feet_clearance(s) == pytest.approx(2 * 0.08 ** 2)


def test_feet_slip_ignores_feet_in_the_air():
    """A swinging foot is supposed to move; only a loaded one is skating."""
    airborne = make_state(contact=np.zeros(4), feet_vel_xy=np.full((4, 2), 1.0))
    planted = make_state(contact=np.ones(4), feet_vel_xy=np.full((4, 2), 1.0))
    assert R.feet_slip(airborne) == pytest.approx(0.0)
    assert R.feet_slip(planted) > 0.0


def test_orientation_is_blind_to_yaw():
    upright = make_state(proj_gravity=np.array([0.0, 0.0, -1.0]))
    assert R.orientation(upright) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
#  Aggregation                                                                #
# --------------------------------------------------------------------------- #


def test_resolve_weights_rejects_typos():
    """A silently ignored reward key costs a whole training run."""
    with pytest.raises(KeyError):
        R.resolve_weights({"track_lin_velocity_xy": 1.0})


def test_resolve_weights_overrides_defaults():
    w = R.resolve_weights({"torques": -1.0})
    assert w["torques"] == -1.0
    assert w["track_lin_vel_xy"] == R.DEFAULT_WEIGHTS["track_lin_vel_xy"]


def test_compute_skips_zero_weights_and_returns_weighted_terms():
    weights = dict.fromkeys(R.REWARD_TERMS, 0.0)
    weights["track_lin_vel_xy"] = 2.0
    total, terms = R.compute(make_state(), weights)
    assert set(terms) == {"track_lin_vel_xy"}
    assert total == pytest.approx(2.0)  # perfect tracking of a zero command


def test_every_registered_term_has_a_default_weight():
    assert set(R.REWARD_TERMS) == set(R.DEFAULT_WEIGHTS)


def test_default_weight_signs_match_the_convention():
    carrots = {"track_lin_vel_xy", "track_ang_vel_yaw", "gait_phase",
               "feet_air_time", "alive"}
    for name, w in R.DEFAULT_WEIGHTS.items():
        if name in carrots:
            assert w > 0.0, name
        else:
            assert w < 0.0, name
