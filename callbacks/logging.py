"""Per-term reward logging.

Why bother
----------
A locomotion reward is a sum of a dozen competing terms. When training goes
wrong the aggregate reward tells you *that* it went wrong, never *which term
won*. Almost every failure mode in this project shows up as one term quietly
dominating:

* The robot stands still  -> ``alive`` and the penalty terms outweigh
  ``track_lin_vel_xy``.
* The robot shuffles      -> ``gait_phase`` is high but ``feet_air_time`` is
  near zero: contacts toggle on cue without real strides.
* The robot skates        -> ``feet_slip`` is large and negative.
* The robot judders       -> ``joint_acceleration`` and ``action_rate``
  dominate the penalties.

Reading those four traces is faster than any amount of watching the viewer.
The environment already computes each weighted term and puts it in the info
dict under ``rew/<name>``; this callback just averages and records them.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardTermLoggerCallback(BaseCallback):
    """Average every ``rew/*`` info key over a rollout and send it to TensorBoard.

    Also logs the tracking errors and the gait mix, which together answer
    "is it actually following commands, and in which gait" without needing to
    render anything.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)
        self._gait_counts = defaultdict(int)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            for key, value in info.items():
                if key.startswith("rew/") or key in (
                    "tracking_lin_err",
                    "tracking_ang_err",
                    "base_height",
                ):
                    self._sums[key] += float(value)
                    self._counts[key] += 1
            gait = info.get("gait")
            if gait is not None:
                self._gait_counts[gait] += 1
        return True

    def _on_rollout_end(self) -> None:
        for key, total in self._sums.items():
            n = max(self._counts[key], 1)
            # rew/* keys keep their prefix so they group in the TensorBoard UI;
            # the rest go under diagnostics/.
            name = key if key.startswith("rew/") else "diagnostics/" + key
            self.logger.record(name, total / n)

        total_gait = sum(self._gait_counts.values())
        if total_gait:
            for gait, count in self._gait_counts.items():
                self.logger.record("gait_fraction/" + gait, count / total_gait)

        self._sums.clear()
        self._counts.clear()
        self._gait_counts.clear()


class EpisodeStatsCallback(BaseCallback):
    """Log the mean episode length and return over a window.

    ``Monitor`` already puts these in ``ep_info_buffer`` and SB3 logs them, but
    only for the training envs and only under ``rollout/``. Having the same
    numbers recorded here keeps the from-scratch PPO trainer (which has no SB3
    logger) comparable against the SB3 runs on identical axes.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        if not buf:
            return
        self.logger.record("episode/mean_length", float(np.mean([e["l"] for e in buf])))
        self.logger.record("episode/mean_return", float(np.mean([e["r"] for e in buf])))
