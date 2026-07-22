# Run 5 Evaluation Results — 5M Steps PPO with PD Controller

## Config
- total_timesteps: 5,000,000
- n_envs: 8
- curriculum: 0.3 -> 0.8 m/s over 1M steps
- PD controller: kp=40, kd=1
- action_scale: 0.5 rad
- Training time: 1h 50m

## Results (50 episodes, cmd_vel=0.8 m/s)

| Metric              | Run 4 (2M) | Run 5 (5M) |
|---------------------|------------|------------|
| Mean reward         | 69.14      | 523.23     |
| Mean episode length | 107.7      | 276.9      |
| Survival rate       | 0.0%       | 0.0%       |
| Mean forward speed  | 0.250 m/s  | 0.737 m/s  |
| Best episode speed  | 0.463 m/s  | 0.874 m/s  |

## Analysis

Major improvement. The robot now walks at 0.737 m/s (target: 0.8 m/s).
Several episodes exceeded the target velocity. The slower curriculum
(1M warmup instead of 500k) and 2.5x more training time paid off.

The gait is functional but episodes still end around 200-400 steps
(~8-16 seconds). The robot falls eventually, likely due to accumulated
drift in orientation. Longer episodes (800+ steps) do occur,
suggesting the policy is close to stable.
