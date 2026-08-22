"""Gait definitions and the periodic phase clock.

Why this file exists
--------------------
In v1 the gait was supposed to *emerge* from a reward that looked at
instantaneous foot contacts (``abs(diag1 - diag2)``). It could not work: that
expression has no memory, so a robot standing perfectly still on one diagonal
pair scores the maximum value forever. See docs/10-reward-engineering.md.

v2 replaces emergence with a *reference clock*. A single scalar phase
``phi in [0, 1)`` advances every control step. Each foot is assigned a phase
offset ``theta_i``; a foot is *supposed* to be on the ground whenever its local
phase ``(phi + theta_i) mod 1`` falls inside the duty window. The reward then
simply asks "does the actual contact state match the schedule?".

Two things fall out of this for free:

1. The gait becomes *controllable*. Swapping the offset vector turns a trot
   into a pace or a bound with no retraining, because the clock is part of the
   observation and the policy learns to condition on it.
2. Step frequency becomes a command. ``phi`` advances by ``f * dt`` per step, so
   raising ``f`` makes the policy take faster steps.

Foot ordering is (FL, FR, RL, RR) everywhere in this repository. That ordering
is fixed by ``Go2Env.FOOT_GEOM_NAMES`` and must not be permuted here.
"""

from __future__ import annotations

import numpy as np

FOOT_ORDER = ("FL", "FR", "RL", "RR")

# Phase offsets per foot, in fractions of a full cycle, plus the duty factor
# (fraction of the cycle each foot spends in stance).
#
# duty = 0.5  -> half the cycle in stance, half in swing
# duty = 0.75 -> a slow "walk"; three feet down most of the time
# duty = 1.0  -> never lift a foot (standing)
GAITS: dict[str, dict] = {
    # Diagonal pairs move together. The default and by far the most stable
    # gait for a robot of the Go2's proportions.
    "trot":  {"offsets": (0.00, 0.50, 0.50, 0.00), "duty": 0.50},
    # Lateral pairs move together. Efficient, but rolls the body noticeably.
    "pace":  {"offsets": (0.00, 0.50, 0.00, 0.50), "duty": 0.50},
    # Front pair then rear pair. Pitches the body; the "bunny hop".
    "bound": {"offsets": (0.00, 0.00, 0.50, 0.50), "duty": 0.50},
    # Four-beat walk: one foot swings at a time, three always in stance.
    "walk":  {"offsets": (0.00, 0.50, 0.25, 0.75), "duty": 0.75},
    # All four leave the ground together. Included mostly as a demo of how far
    # the phase reward can push the gait away from the trot basin.
    "pronk": {"offsets": (0.00, 0.00, 0.00, 0.00), "duty": 0.50},
    # Degenerate schedule used when the commanded velocity is ~zero: every
    # foot is supposed to stay planted, so the gait reward stops asking the
    # robot to march on the spot.
    "stand": {"offsets": (0.00, 0.00, 0.00, 0.00), "duty": 1.00},
}

GAIT_NAMES = tuple(GAITS.keys())


def gait_params(name: str) -> tuple[np.ndarray, float]:
    """Return ``(offsets, duty)`` for a named gait."""
    if name not in GAITS:
        raise KeyError(f"unknown gait {name!r}; known gaits: {GAIT_NAMES}")
    g = GAITS[name]
    return np.asarray(g["offsets"], dtype=np.float64), float(g["duty"])


def advance_phase(phase: float, frequency: float, dt: float) -> float:
    """Advance the global gait clock by one control step.

    ``phase`` is kept in [0, 1) rather than [0, 2*pi) so that phase offsets read
    as plain fractions of a stride. The conversion to radians happens only when
    the clock is embedded in the observation as (sin, cos).
    """
    return float((phase + frequency * dt) % 1.0)


def desired_contact(phase, offsets, duty):
    """Boolean-valued (as floats) stance schedule for each foot.

    ``1.0`` means "this foot should be on the ground right now", ``0.0`` means
    "this foot should be swinging". Written with array ops only so that the
    same function works under numpy and under jax.numpy for the MJX backend.
    """
    xp = _backend(offsets)
    local_phase = xp.mod(phase + offsets, 1.0)
    return xp.where(local_phase < duty, 1.0, 0.0)


def clock_signal(phase, offsets):
    """Per-foot (sin, cos) clock, shape (2 * n_feet,).

    Not used by the default observation (which carries only the *global* clock)
    but handy for experiments and for plotting in scripts/gait_analysis.py.
    """
    xp = _backend(offsets)
    local = 2.0 * xp.pi * xp.mod(phase + offsets, 1.0)
    return xp.concatenate([xp.sin(local), xp.cos(local)])


def _backend(x):
    """Return numpy or jax.numpy depending on the array type handed in.

    Every function in this module and in envs/rewards.py is written against
    this so that the MJX training path (Phase 5) can reuse the exact same
    reward and gait maths as the CPU environment. Sharing the code is the only
    practical way to guarantee the two backends optimise the same objective.
    """
    if type(x).__module__.startswith("jaxlib") or type(x).__module__.startswith("jax"):
        import jax.numpy as jnp
        return jnp
    return np
