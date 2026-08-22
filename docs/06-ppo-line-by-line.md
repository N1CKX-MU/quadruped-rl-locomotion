# 6. PPO line by line

This chapter walks `ppo_from_scratch/ppo.py` from top to bottom, mapping each
block to the equations of chapters 4 and 5. Have the file open beside this.

The file is about 450 lines. Roughly 120 of those are the algorithm; the rest is
the machinery that every real implementation needs and no paper mentions. The
gap between those two numbers is the point of this chapter.

## 6.1 `RunningMeanStd` — observation normalisation

```python
class RunningMeanStd:
    def update(self, x): ...
    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)
```

Not in any PPO paper. Absolutely required in practice.

The observation mixes quantities on wildly different scales: joint angles are
$O(1)$ rad, joint velocities reach 20 rad/s, the gait clock lives in $[-1,1]$.
The first linear layer applies one weight matrix to all of them. Without
normalisation, the velocity block dominates the layer's output, and early
training is spent learning to down-weight it — capacity and gradient steps spent
on a problem you could have removed with three lines of arithmetic.

The update rule is Chan's parallel variance algorithm, which merges two sets of
(count, mean, variance) statistics without revisiting the data:

$$
\begin{aligned}
\Delta &= \mu_B - \mu_A \\
\mu &= \mu_A + \Delta \frac{n_B}{n_A + n_B} \\
M_2 &= \sigma_A^2 n_A + \sigma_B^2 n_B + \Delta^2 \frac{n_A n_B}{n_A + n_B} \\
\sigma^2 &= M_2 / (n_A + n_B)
\end{aligned}
$$

The naive alternative — accumulating $\sum x$ and $\sum x^2$ — loses catastrophic
precision when the mean is large relative to the variance, which is exactly the
regime for joint angles clustered around $-1.8$ rad.

Two operational points, both of which have bitten this repository:

- **The statistics are part of the model.** A policy trained on normalised
  observations is meaningless when fed raw ones. `save()` writes them alongside
  the weights, deliberately. Bug B12 (chapter 14) is what happens when they get
  separated.
- **Freeze at evaluation.** `_normalize(..., update=False)` for the terminal
  observation, so that evaluation does not shift the statistics.

## 6.2 `ActorCritic` — the networks

```python
self.actor  = orthogonal_init(mlp([obs_dim, *hidden, act_dim]))
nn.init.orthogonal_(self.actor[-2].weight, 0.01)
self.critic = orthogonal_init(mlp([obs_dim, *hidden, 1]))
self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
```

**Separate trunks.** The two networks share no parameters. Sharing is common in
Atari work, where both heads need the same visual features, and a bad idea here:
the critic must model the return of a stochastic, externally-pushed robot — it
needs features encoding "how likely am I to fall" — while the actor needs a
smooth control law. Forcing one representation to serve both makes the value
loss and the policy loss fight over the shared layers, mediated only by
$c_v$.

**ELU, not ReLU or tanh.** ELU is smooth at zero and does not have ReLU's dead
units. For a controller producing continuous joint targets, the smoothness
matters: a ReLU network's output is piecewise linear with kinks, and those kinks
show up as small discontinuities in the commanded joint trajectory. This is
mostly aesthetic in simulation and mostly not on hardware.

**Orthogonal initialisation.** An orthogonal matrix preserves the norm of its
input, so activations neither blow up nor vanish through a deep stack at
initialisation. The gain $\sqrt 2$ compensates for the expected halving of
variance through a ReLU-family nonlinearity.

**Gain 0.01 on the actor's output layer.** This one is genuinely important and
often skipped. It makes the initial policy output near-zero, i.e. *the nominal
standing pose*. Combined with `log_std_init = -1.0` ($\sigma \approx 0.37$), the
robot begins by standing and jittering slightly, rather than by slamming all
twelve joints to their limits. An agent that falls in the first three control
steps of every episode collects no information about walking.

**`log_std` is a free parameter**, not a network output. Chapter 1, §1.3
explains why: a state-dependent std can shrink its own exploration precisely in
the difficult states.

### `evaluate`

```python
def evaluate(self, obs, actions):
    dist = self.distribution(obs)
    return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), self.value(obs)
```

This re-scores **stored** actions under the **current** policy. It is the heart
of the whole thing: the actions came from $\pi_{\theta_\text{old}}$, and Eq 5.2
needs $\log \pi_\theta(a_t \mid s_t)$ for those same actions to build the ratio.

`.sum(-1)` because the policy is a diagonal Gaussian: independent components, so
the joint density is a product and its log is a sum. Forgetting this sum is a
classic bug — the ratio then reflects one joint instead of the whole action.

## 6.3 `RolloutBuffer` — storage and GAE

Shapes are `(n_steps, n_envs, ...)`. Everything is discarded after the update;
that is what on-policy means.

### The `dones` convention

```python
self.dones[i] = self._last_done
```

Note the buffer stores the done flag of the *previous* step alongside the
current observation. So `dones[t]` answers "is `obs[t]` the first state of a
fresh episode?", and therefore `dones[t+1]` answers "did the episode end at step
`t`?". This is the CleanRL convention and it makes the GAE loop read cleanly:

```python
next_non_terminal = 1.0 - self.dones[t + 1]
next_value        = self.values[t + 1]
```

It is also an easy off-by-one to get wrong, and getting it wrong produces a
model that trains, converges, and is subtly worse — with no error anywhere.

### `compute_returns`

```python
rewards = self.rewards.copy()
rewards += gamma * self.truncated_values          # Eq 4.9

for t in reversed(range(self.n_steps)):
    delta    = rewards[t] + gamma * next_value * next_non_terminal - self.values[t]   # Eq 4.2
    last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae              # Eq 4.6
    advantages[t] = last_gae

returns = advantages + self.values                # Eq 4.8
```

Six lines for the whole of chapter 4. Three things to notice:

1. **Backwards.** $\hat A_t$ depends on $\hat A_{t+1}$, seeded by the value of
   the state the rollout stopped at.
2. **The truncation bootstrap is folded into the reward** before the loop. This
   is the cleanest place to put it: it makes the loop itself unaware of the
   distinction, which is what you want, since the distinction is about the
   *reward*, not about the recursion.
3. **`returns` is the $\lambda$-return, not the Monte-Carlo return.** Fitting the
   critic to this rather than to $G_t$ is what keeps the critic's targets
   low-variance.

## 6.4 `collect_rollout`

```python
clipped = np.clip(action_np, -1.0, 1.0)
next_obs, rewards, dones, infos = self.env.step(clipped)
self.buffer.add(self._last_obs, action_np, log_prob.cpu().numpy(), ...)
```

**Store the unclipped action, send the clipped one.** The Gaussian is unbounded;
the action space is $[-1,1]^{12}$. If you stored the *clipped* action, then in
the update `evaluate()` would compute $\log \pi_\theta(\text{clipped})$, which
does not match the stored $\log \pi_{\theta_\text{old}}(\text{unclipped})$, and
the ratio in Eq 5.2 would be systematically wrong for every saturated dimension.

Since `log_std_init = -1.0` and the policy operates near zero, saturation is
rare early on — which is precisely why this bug is hard to notice. It only bites
once the policy is confident, at which point training mysteriously destabilises.

### Truncation handling

```python
for i, info in enumerate(infos):
    if dones[i] and info.get("TimeLimit.truncated", False):
        terminal_obs = self._normalize(np.asarray(info["terminal_observation"])[None],
                                       update=False)
        truncated_values[i] = float(self.policy.value(...).item())
```

SB3's vectorised wrappers auto-reset on `done` and stash the true final
observation in `info["terminal_observation"]`. Without that, `next_obs` is
already the *reset* observation and $V(s_T)$ would be evaluated at the wrong
state. `update=False` so a handful of terminal observations do not skew the
running statistics.

## 6.5 `update` — the objective

```python
ratio = torch.exp(log_prob - mb["log_probs"])                        # Eq 5.2

adv = mb["advantages"]
adv = (adv - adv.mean()) / (adv.std() + 1e-8)                        # Eq 4.10

surrogate_1  = ratio * adv
surrogate_2  = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
policy_loss  = -torch.min(surrogate_1, surrogate_2).mean()           # Eq 5.6
```

Four lines. That is PPO.

The `-` is because PyTorch minimises: we want to maximise $L^{\text{CLIP}}$.

```python
value_clipped = mb["values"] + torch.clamp(value - mb["values"], -clip_vf, clip_vf)
value_loss = torch.max((value - mb["returns"]) ** 2,
                       (value_clipped - mb["returns"]) ** 2).mean()  # Eq 5.8
```

`max` here versus `min` above: for the policy we take the pessimistic
(smaller) *objective*, for the value we take the pessimistic (larger) *loss*.
Both are the conservative choice; the sign flips because one is being maximised
and the other minimised.

```python
loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss   # Eq 5.7
```

with `entropy_loss = -entropy.mean()`, so adding it with a positive coefficient
*maximises* entropy.

### Gradient clipping

```python
nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
```

Distinct from the PPO clip and doing a different job. PPO's clip bounds how far
the *policy distribution* moves. Nothing in Eq 5.7 bounds the *gradient
magnitude* — a minibatch where the advantage standardisation divides by a tiny
std can produce an enormous gradient and undo an entire rollout's progress.
Clipping by global norm is the cheap insurance.

### Approximate KL and early stop

```python
log_ratio = log_prob - mb["log_probs"]
approx_kl = ((ratio - 1) - log_ratio).mean().item()                   # Eq 5.10
if approx_kl > 1.5 * self.cfg.target_kl:
    early_stopped = True
    break
```

Note this is inside `torch.no_grad()` — it is a diagnostic, not part of the
objective.

The `1.5` multiplier is SB3's convention and is arbitrary but sensible: it lets
a single noisy minibatch through without abandoning the update, while stopping
a genuine drift.

### Explained variance

```python
out["explained_variance"] = 1 - np.var(y_true - y_pred) / np.var(y_true)
```

$$
\mathrm{EV} = 1 - \frac{\operatorname{Var}[G - \hat V]}{\operatorname{Var}[G]}
\tag{6.1}
$$

1.0 means the critic explains the return perfectly. 0.0 means it is no better
than predicting the mean. **Negative** means it is worse than predicting the
mean, which is a real and common state early in training.

This is the most diagnostic single number in a PPO run. Chapter 13 explains why.

## 6.6 `learn` — the loop

```python
for update in range(1, n_updates + 1):
    if self.cfg.anneal_lr:
        frac = 1.0 - (update - 1) / n_updates
        for group in self.optimizer.param_groups:
            group["lr"] = frac * self.cfg.learning_rate

    advantages, returns = self.collect_rollout()
    stats = self.update(advantages, returns)
```

Linear learning-rate annealing to zero. Late in training the advantages carry
less signal (the policy is near a local optimum, so $A \approx 0$ plus noise)
and a constant learning rate keeps injecting that noise into the weights. The
schedule is not a nicety; on this task it visibly reduces the late-training
wobble in episode return.

## 6.7 What is *not* here

Honest accounting of what a production implementation adds:

| Feature | Why it is omitted |
|---|---|
| Recurrent policies | The observation is Markov enough; adds substantial complexity |
| Reward normalisation | SB3's `VecNormalize` does it; here the reward is already $O(0.5)$ by design (chapter 10) |
| Multiple optimisers | One Adam over all parameters is simpler and works |
| Distributed rollout workers | `SubprocVecEnv` is borrowed from SB3; process management is not the algorithm |
| Learning-rate schedules beyond linear | Diminishing returns |
| KL-penalty PPO variant | The clipped variant is what everyone uses |

## 6.8 Verifying it

The test that matters is parity with SB3:

```bash
python scripts/train.py --n-envs 8 --timesteps 500000 --run-name sb3_ref --seed 0
python ppo_from_scratch/train_scratch.py --n-envs 8 --timesteps 500000 --run-name scratch_ref --seed 0
tensorboard --logdir logs/tensorboard
```

`rollout/ep_rew_mean` should track between the two within seed noise. On a short
smoke run (24k steps, 4 envs), the two implementations produce episode returns
of 36–39 at the same point in training, which is the expected agreement.

They will not be bit-identical, and chasing that is not worth it: SB3 differs in
its initialisation details, its handling of the final partial minibatch, and its
observation-normalisation warm-up. What matters is that the *curves* agree. If
they do not, the difference is a specific implementation detail, and finding it
teaches more about PPO than re-reading the paper.

---

**Previous:** [5. From TRPO to PPO](05-trpo-to-ppo.md) ·
**Next:** [7. Why PPO and not SAC or TD3](07-why-not-sac-td3.md)
