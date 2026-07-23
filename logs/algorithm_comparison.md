# Algorithm Comparison — PPO vs SAC vs TD3

## Config
- timesteps: 1,000,000 each (single env, DummyVecEnv)
- Same environment, reward function, PD controller
- cmd_vel: [0.5, 0.0, 0.0] (no curriculum for fair comparison)
- Eval: 20 episodes at cmd_vel=0.8 m/s

## Results

| Metric              | PPO      | SAC      | TD3      |
|---------------------|----------|----------|----------|
| Mean reward         | 182.78   | 232.45   | 259.39   |
| Mean episode length | 264.2    | 243.7    | 440.2    |
| Survival rate       | 0.0%     | 0.0%     | 10.0%    |
| Mean forward speed  | 0.140    | 0.027    | 0.012    |

## Analysis

At 1M steps with a single environment:

- **PPO** walks the fastest (0.14 m/s) but falls sooner. PPO is on-policy
  and benefits most from parallel envs (8x in the main training run).
  With 8 envs and 5M steps it achieved 0.737 m/s.

- **SAC** achieves higher reward than PPO but barely moves forward. It
  learned a stable balancing policy with slight movement. SAC is
  off-policy and more sample-efficient in theory, but locomotion tasks
  typically favor on-policy methods with large batch rollouts.

- **TD3** has the longest episodes (440 steps) and 10% survival rate,
  but near-zero forward speed. It learned the most stable standing
  policy. TD3's conservative updates (delayed policy, clipped double-Q)
  make it cautious — good for stability, bad for exploration.

## Conclusion

PPO is the best algorithm for this task when given sufficient parallel
environments. Its ability to collect large on-policy rollouts (2048
steps x 8 envs = 16k per update) gives it the exploration needed to
discover walking gaits. SAC and TD3 converge to local optima (standing)
because their replay buffers are dominated by early falling experiences.
