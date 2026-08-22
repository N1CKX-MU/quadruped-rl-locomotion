"""Gait clock and schedule tests.

These exist because v1's gait reward was maximised by a robot standing
perfectly still (bug B4). The replacement only works if the *schedule* really
is periodic and really does alternate, so that is what gets pinned here.
"""

import numpy as np
import pytest

from envs.gait import (
    FOOT_ORDER,
    GAITS,
    advance_phase,
    clock_signal,
    desired_contact,
    gait_params,
)

FL, FR, RL, RR = range(4)


def test_foot_order_is_the_documented_one():
    assert FOOT_ORDER == ("FL", "FR", "RL", "RR")


def test_phase_wraps_and_completes_one_cycle_per_period():
    dt, freq = 0.02, 2.0          # 50 Hz control, 2 strides per second
    steps_per_cycle = int(round(1.0 / (freq * dt)))
    phase = 0.0
    for _ in range(steps_per_cycle):
        phase = advance_phase(phase, freq, dt)
        assert 0.0 <= phase < 1.0
    assert phase == pytest.approx(0.0, abs=1e-9)


def test_higher_frequency_means_more_cycles():
    """Step frequency is a command, so it must actually change stride rate.

    After exactly one second both clocks land back near zero (1 and 3 whole
    cycles), so compare the number of wraps rather than the final phase.
    """
    dt = 0.02
    crossings_slow = crossings_fast = 0
    p_s = p_f = 0.0
    for _ in range(50):
        n_s = advance_phase(p_s, 1.0, dt)
        n_f = advance_phase(p_f, 3.0, dt)
        crossings_slow += n_s < p_s
        crossings_fast += n_f < p_f
        p_s, p_f = n_s, n_f
    assert crossings_fast > crossings_slow


def test_trot_puts_diagonal_pairs_in_phase():
    """FL+RR against FR+RL. This is the definition of a trot."""
    offsets, duty = gait_params("trot")
    for phase in np.linspace(0.0, 1.0, 101, endpoint=False):
        d = desired_contact(phase, offsets, duty)
        assert d[FL] == d[RR], phase
        assert d[FR] == d[RL], phase
        # With duty 0.5, exactly one diagonal pair is in stance at any moment.
        assert d[FL] != d[FR], phase


def test_pace_and_bound_pair_the_other_ways():
    for name, pair_a, pair_b in [
        ("pace", (FL, RL), (FR, RR)),     # lateral pairs
        ("bound", (FL, FR), (RL, RR)),    # front pair, rear pair
    ]:
        offsets, duty = gait_params(name)
        for phase in np.linspace(0.0, 1.0, 51, endpoint=False):
            d = desired_contact(phase, offsets, duty)
            assert d[pair_a[0]] == d[pair_a[1]], (name, phase)
            assert d[pair_b[0]] == d[pair_b[1]], (name, phase)


def test_walk_keeps_three_feet_down():
    """A four-beat walk has duty 0.75, so three feet are in stance at a time."""
    offsets, duty = gait_params("walk")
    for phase in np.linspace(0.0, 1.0, 101, endpoint=False):
        assert desired_contact(phase, offsets, duty).sum() == 3.0


def test_stand_never_lifts_a_foot():
    offsets, duty = gait_params("stand")
    for phase in np.linspace(0.0, 1.0, 51, endpoint=False):
        assert desired_contact(phase, offsets, duty).sum() == 4.0


def test_duty_factor_is_the_stance_fraction():
    """Averaged over a full cycle, each foot is in stance exactly `duty` of it."""
    for name, spec in GAITS.items():
        offsets, duty = gait_params(name)
        samples = np.array([
            desired_contact(p, offsets, duty)
            for p in np.linspace(0.0, 1.0, 1000, endpoint=False)
        ])
        assert np.allclose(samples.mean(axis=0), duty, atol=2e-3), name


def test_v1_gait_reward_was_maximised_by_standing_still():
    """Regression guard, and a demonstration for docs/10-reward-engineering.md.

    v1 scored gait quality as ``abs(FL*RR - FR*RL)`` on instantaneous contacts.
    A robot frozen with one diagonal pair loaded scores 1.0 - the maximum -
    forever, without ever moving. The v2 schedule cannot be gamed that way
    because the target alternates in time.
    """
    def v1_gait_reward(contacts):
        return abs(contacts[FL] * contacts[RR] - contacts[FR] * contacts[RL])

    frozen = np.array([1.0, 0.0, 0.0, 1.0])   # standing on FL+RR, never moving
    assert v1_gait_reward(frozen) == 1.0

    offsets, duty = gait_params("trot")
    scores = []
    phase = 0.0
    for _ in range(50):
        d = desired_contact(phase, offsets, duty)
        scores.append(float(np.mean(frozen * d + (1 - frozen) * (1 - d))))
        phase = advance_phase(phase, 2.0, 0.02)
    # The frozen robot matches the alternating schedule only half the time.
    assert np.mean(scores) == pytest.approx(0.5, abs=0.05)


def test_clock_signal_shape_and_range():
    offsets, _ = gait_params("trot")
    sig = clock_signal(0.3, offsets)
    assert sig.shape == (8,)
    assert np.all(np.abs(sig) <= 1.0 + 1e-12)
