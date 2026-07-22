# Run 4 Evaluation Results — 2M Steps PPO with PD Controller

## Config
- total_timesteps: 2,000,000
- n_envs: 8
- curriculum: 0.3 -> 1.2 m/s over 500k steps
- PD controller: kp=40, kd=1
- action_scale: 0.5 rad
- Training time: 44 minutes

## Results (50 episodes, cmd_vel=0.8 m/s)

| Metric              | Run 1 (torque) | Run 4 (PD) |
|---------------------|----------------|------------|
| Mean reward         | 337.25         | 69.14      |
| Mean episode length | 390.7 steps    | 107.7      |
| Survival rate       | 2.0%           | 0.0%       |
| Mean forward speed  | -0.005 m/s     | 0.250 m/s  |
| Best episode speed  | 0.137 m/s      | 0.463 m/s  |

## Diagnosis

The robot is now **walking forward** at 0.25 m/s average. This is a
breakthrough — Runs 1-3 all produced stationary balancing policies.

The PD controller was the key fix: torque actuators need a PD layer to
convert position targets into appropriate torques. This is standard
practice in all major quadruped RL frameworks (IsaacGym, legged_gym).

### Remaining issue: falls after ~108 steps

The gait is unstable — the robot walks but topples. Likely causes:
- 2M steps is not enough to learn a stable gait with curriculum
- Curriculum ramps too fast (0.3 -> 1.2 m/s in 500k steps)
- May need more training time (5M+ steps)

### Plan for Run 5
- Increase to 5M timesteps
- Slow curriculum: 0.3 -> 0.8 m/s over 1M steps (less aggressive)
