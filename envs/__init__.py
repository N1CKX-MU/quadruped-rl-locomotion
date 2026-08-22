import gymnasium as gym

from envs.go2_env import Go2Env
from envs.commands import Command, CommandRanges, CommandCurriculum, CommandSampler
from envs.gait import GAITS, GAIT_NAMES, FOOT_ORDER

gym.register(
    id="Go2Walk-v0",
    entry_point="envs.go2_env:Go2Env",
    max_episode_steps=1000,
)

__all__ = [
    "Go2Env",
    "Command",
    "CommandRanges",
    "CommandCurriculum",
    "CommandSampler",
    "GAITS",
    "GAIT_NAMES",
    "FOOT_ORDER",
]
