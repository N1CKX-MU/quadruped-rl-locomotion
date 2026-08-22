"""Velocity command sampling and the adaptive command curriculum.

Why this file exists
--------------------
v1 had a single ``cmd_vel`` tuple, fixed for an entire training run and only
ever ramped along x by the curriculum callback. A policy that is only ever
shown one command has no reason to *condition* on the command, so it collapses
into an open-loop forward gait. That is the whole explanation for "it only does
a forward gait": the environment never asked for anything else.

v2 samples a fresh command at every episode reset and again every few seconds
*within* an episode. The command enters the observation, so the policy must
read it to score well. Three details matter:

* **Stand-still commands.** If zero velocity is just another point in a
  continuous range it is sampled with probability ~0, and the policy learns to
  march on the spot when told to stop. So zero is sampled as an explicit
  discrete mode with its own probability.
* **Mid-episode resampling.** Without it the policy can infer the command from
  its own early motion and stop attending to the observation. Changing the
  command mid-episode also trains the transitions, which is what you feel when
  driving the robot with scripts/play.py.
* **Curriculum over the *ranges*, not over a scalar.** See CommandCurriculum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.gait import GAIT_NAMES


@dataclass
class CommandRanges:
    """Sampling ranges for the command vector.

    Backwards walking is deliberately given a smaller range than forwards: it
    is harder, less useful, and letting it dominate the sampling slows down the
    forward gait everyone actually looks at.
    """

    lin_vel_x: tuple = (-1.0, 1.5)
    lin_vel_y: tuple = (-0.7, 0.7)
    ang_vel_yaw: tuple = (-1.5, 1.5)
    gait_frequency: tuple = (1.5, 3.0)
    base_height: tuple = (0.28, 0.34)

    def lerp(self, other, t):
        """Interpolate between two range sets. Used by the curriculum."""
        t = float(np.clip(t, 0.0, 1.0))

        def _l(a, b):
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

        return CommandRanges(
            lin_vel_x=_l(self.lin_vel_x, other.lin_vel_x),
            lin_vel_y=_l(self.lin_vel_y, other.lin_vel_y),
            ang_vel_yaw=_l(self.ang_vel_yaw, other.ang_vel_yaw),
            gait_frequency=_l(self.gait_frequency, other.gait_frequency),
            base_height=_l(self.base_height, other.base_height),
        )

    def as_dict(self):
        return {
            "lin_vel_x": tuple(self.lin_vel_x),
            "lin_vel_y": tuple(self.lin_vel_y),
            "ang_vel_yaw": tuple(self.ang_vel_yaw),
            "gait_frequency": tuple(self.gait_frequency),
            "base_height": tuple(self.base_height),
        }


@dataclass
class Command:
    """One sampled command. Linear and angular velocities are in the base frame."""

    lin_vel_x: float = 0.0
    lin_vel_y: float = 0.0
    ang_vel_yaw: float = 0.0
    gait: str = "trot"
    gait_frequency: float = 2.0
    base_height: float = 0.30

    @property
    def vec(self):
        """The three tracked quantities, in the order used by the observation."""
        return np.array(
            [self.lin_vel_x, self.lin_vel_y, self.ang_vel_yaw], dtype=np.float64
        )

    @property
    def is_standing(self):
        return float(np.linalg.norm(self.vec)) < 0.1


class CommandSampler:
    """Draws commands, and decides when a running episode needs a new one."""

    def __init__(
        self,
        ranges=None,
        stand_probability=0.10,
        resample_interval_s=5.0,
        gaits=("trot",),
        gait_probabilities=None,
    ):
        self.ranges = ranges or CommandRanges()
        self.stand_probability = float(stand_probability)
        self.resample_interval_s = float(resample_interval_s)
        for g in gaits:
            if g not in GAIT_NAMES:
                raise KeyError("unknown gait %r; known: %s" % (g, (GAIT_NAMES,)))
        self.gaits = tuple(gaits)
        self.gait_probabilities = gait_probabilities

    def sample(self, rng):
        r = self.ranges
        gait = str(rng.choice(self.gaits, p=self.gait_probabilities))
        cmd = Command(
            gait=gait,
            gait_frequency=float(rng.uniform(*r.gait_frequency)),
            base_height=float(rng.uniform(*r.base_height)),
        )

        # Explicit stand-still mode. Without this, "stop" is measure-zero in a
        # continuous range and the policy never learns to actually stop.
        if rng.random() < self.stand_probability:
            cmd.gait = "stand"
            return cmd

        cmd.lin_vel_x = float(rng.uniform(*r.lin_vel_x))
        cmd.lin_vel_y = float(rng.uniform(*r.lin_vel_y))
        cmd.ang_vel_yaw = float(rng.uniform(*r.ang_vel_yaw))

        # A command whose magnitude lands near zero by chance is snapped to a
        # true stand, so the gait schedule and the stand_still reward agree
        # instead of fighting each other in the ambiguous band.
        if float(np.linalg.norm(cmd.vec)) < 0.15:
            cmd.lin_vel_x = 0.0
            cmd.lin_vel_y = 0.0
            cmd.ang_vel_yaw = 0.0
            cmd.gait = "stand"
        return cmd

    def should_resample(self, elapsed_s):
        if self.resample_interval_s <= 0.0:
            return False
        return elapsed_s >= self.resample_interval_s


class CommandCurriculum:
    """Widens the command ranges as the policy proves it can track them.

    v1's curriculum ramped a single forward-speed scalar on a fixed timestep
    schedule. That has two problems: it is open loop (it widens whether or not
    the policy is coping), and a scalar cannot describe a three-dimensional
    command space.

    This version is closed loop. After every rollout the trainer reports the
    mean *normalised* tracking score, i.e. how close the achieved velocity was
    to the commanded one, in [0, 1]. Cross the threshold and the ranges widen a
    notch toward ``final``; drop well below it and they narrow again. The level
    is the single number worth watching in TensorBoard on a long run.
    """

    def __init__(
        self,
        initial,
        final,
        threshold=0.85,
        step=0.05,
        decay_threshold=0.65,
    ):
        self.initial = initial
        self.final = final
        self.threshold = float(threshold)
        self.decay_threshold = float(decay_threshold)
        self.step = float(step)
        self.level = 0.0

    def update(self, tracking_score):
        """Return the new curriculum level given the latest tracking score."""
        if tracking_score >= self.threshold:
            self.level = min(1.0, self.level + self.step)
        elif tracking_score < self.decay_threshold:
            self.level = max(0.0, self.level - self.step)
        return self.level

    @property
    def ranges(self):
        return self.initial.lerp(self.final, self.level)


def ranges_from_config(cfg):
    """Build a CommandRanges from a YAML mapping, falling back to defaults."""
    if not cfg:
        return CommandRanges()
    d = CommandRanges()
    return CommandRanges(
        lin_vel_x=tuple(cfg.get("lin_vel_x", d.lin_vel_x)),
        lin_vel_y=tuple(cfg.get("lin_vel_y", d.lin_vel_y)),
        ang_vel_yaw=tuple(cfg.get("ang_vel_yaw", d.ang_vel_yaw)),
        gait_frequency=tuple(cfg.get("gait_frequency", d.gait_frequency)),
        base_height=tuple(cfg.get("base_height", d.base_height)),
    )
