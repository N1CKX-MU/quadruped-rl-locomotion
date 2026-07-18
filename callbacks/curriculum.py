"""Curriculum learning callback for progressive velocity targets."""

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """Gradually increase target velocity as training progresses.

    Starts at start_vel and linearly ramps to max_vel over warmup_steps.
    """

    def __init__(self, max_vel=1.2, start_vel=0.3, warmup_steps=500_000, verbose=0):
        super().__init__(verbose)
        self.max_vel = max_vel
        self.start_vel = start_vel
        self.warmup_steps = warmup_steps

    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self.warmup_steps)
        current_vel = self.start_vel + (self.max_vel - self.start_vel) * progress

        # Update command velocity in all environments
        for env_idx in range(self.training_env.num_envs):
            self.training_env.env_method(
                "set_cmd_vel",
                (current_vel, 0.0, 0.0),
                indices=[env_idx],
            )

        self.logger.record("curriculum/target_vel", current_vel)
        self.logger.record("curriculum/progress", progress)
        return True
