import gymnasium as gym
from envs.go2_env import Go2Env

gym.register(
    id="Go2Walk-v0",
    entry_point="envs.go2_env:Go2Env",
    max_episode_steps=1000,
)

__all__ = ["Go2Env"]
