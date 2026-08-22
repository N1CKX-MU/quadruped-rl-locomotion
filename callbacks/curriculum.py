"""Adaptive command curriculum.

What was wrong with v1
----------------------
Two separate problems, one of them a silent performance bug.

1. *Open loop.* The old callback ramped a single forward-speed scalar linearly
   in wall-clock timesteps, regardless of whether the policy could actually
   track it. If the policy fell behind, the curriculum kept widening anyway and
   training diverged; if the policy raced ahead, the curriculum held it back.

2. *One IPC round trip per environment per step.* ``_on_step`` looped over
   ``num_envs`` and issued a separate ``env_method`` call for each, with
   ``indices=[i]``. Under ``SubprocVecEnv`` every one of those is a pickle,
   a pipe write and a blocking pipe read. At 8 envs that is 8 blocking round
   trips on **every environment step**, to send a value that had not changed.

This version fixes both. It closes the loop on the measured tracking score,
and it touches the environments at most once per rollout - roughly once every
2048 steps rather than once every step.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from envs.commands import CommandCurriculum, CommandRanges


class CommandCurriculumCallback(BaseCallback):
    """Widen the command sampling ranges as tracking performance allows.

    The environments report a ``tracking_score`` in [0, 1] in their info dict
    (1.0 = the achieved body-frame velocity matches the command exactly). We
    average it over a rollout and hand it to :class:`CommandCurriculum`, which
    nudges a scalar ``level`` up or down. The level interpolates the sampling
    ranges from ``initial`` to ``final``.

    Watch ``curriculum/level`` in TensorBoard. It should climb steadily and then
    sit at 1.0. A level that oscillates means the threshold is too close to
    what the policy can actually achieve; a level pinned at 0 means something
    upstream is broken, not that the curriculum is too strict.
    """

    def __init__(
        self,
        initial_ranges: CommandRanges,
        final_ranges: CommandRanges,
        threshold: float = 0.85,
        decay_threshold: float = 0.65,
        step: float = 0.05,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.curriculum = CommandCurriculum(
            initial=initial_ranges,
            final=final_ranges,
            threshold=threshold,
            decay_threshold=decay_threshold,
            step=step,
        )
        self._scores: list[float] = []
        self._pushed_level = None

    def _on_training_start(self) -> None:
        # Make sure the workers start from the narrow ranges rather than
        # whatever their constructor defaulted to.
        self._push_ranges(force=True)

    def _on_step(self) -> bool:
        # In-process bookkeeping only. No IPC here - that was v1's mistake.
        for info in self.locals.get("infos", ()):
            score = info.get("tracking_score")
            if score is not None:
                self._scores.append(score)
        return True

    def _on_rollout_end(self) -> None:
        if not self._scores:
            return
        score = float(np.mean(self._scores))
        self._scores.clear()
        level = self.curriculum.update(score)

        self.logger.record("curriculum/tracking_score", score)
        self.logger.record("curriculum/level", level)
        r = self.curriculum.ranges
        self.logger.record("curriculum/max_vel_x", r.lin_vel_x[1])
        self.logger.record("curriculum/max_vel_y", r.lin_vel_y[1])
        self.logger.record("curriculum/max_yaw_rate", r.ang_vel_yaw[1])

        self._push_ranges()

    def _push_ranges(self, force: bool = False) -> None:
        level = self.curriculum.level
        if not force and self._pushed_level == level:
            return  # nothing changed; do not pay for the IPC
        # One broadcast call for every worker, not one call per worker.
        self.training_env.env_method("set_command_ranges", self.curriculum.ranges)
        self._pushed_level = level


# Backwards-compatible alias so older scripts and the v1 docs still import.
CurriculumCallback = CommandCurriculumCallback
