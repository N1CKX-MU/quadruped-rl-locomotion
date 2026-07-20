# Run 1 Evaluation Results — 2M Steps PPO

## Config
- total_timesteps: 2,000,000
- n_envs: 8
- curriculum: 0.3 -> 1.2 m/s over 500k steps
- Training time: 47 minutes

## Results (50 episodes, cmd_vel=0.8 m/s)

| Metric              | Value       |
|---------------------|-------------|
| Mean reward         | 337.25      |
| Std reward          | 177.13      |
| Mean episode length | 390.7 steps |
| Survival rate       | 2.0%        |
| Mean forward speed  | -0.005 m/s  |
| Std forward speed   | 0.040 m/s   |

## Diagnosis

The robot learned to **balance in place** but did NOT learn to walk forward.
Mean forward speed is essentially 0 m/s despite a cmd_vel target of 0.8 m/s.

### Root cause: alive bonus dominates velocity tracking

The reward function has a structural imbalance:

1. **Alive bonus = 0.5 per step, unconditionally** — just for not falling.
   Over 400 steps that's +200 reward for doing nothing.

2. **Velocity tracking = exp(-error^2 / 0.25)** — when standing still with
   cmd_vel=0.8, error^2 = 0.64, so reward = exp(-0.64/0.25) = exp(-2.56) ≈ 0.077.
   That's tiny compared to the alive bonus.

3. The agent discovered: "standing still = guaranteed 0.5/step" vs
   "trying to walk = risk falling for a marginal velocity reward of ~0.08/step".
   Rational choice: stand still.

### Fix plan for Run 2

- Reduce alive bonus: 0.5 -> 0.2
- Increase velocity tracking weight: 1.0 -> 2.0
- Widen the exp denominator: 0.25 -> 0.5 (less harsh penalty for partial speed)
- Increase action_scale: 0.3 -> 0.5 (allow larger joint movements)
