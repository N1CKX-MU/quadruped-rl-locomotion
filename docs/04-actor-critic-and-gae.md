# 4. Actor-critic and generalised advantage estimation

Chapter 3 ended with

$$
\nabla_\theta J(\theta) = \mathbb{E}\!\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, A^\pi(s_t, a_t) \right]
$$

and one unresolved problem: we do not know $A^\pi$. This chapter is about
estimating it, and about the fact that every estimator sits somewhere on a
bias–variance spectrum that you get to choose a point on.

## 4.1 The actor-critic split

Two networks, two jobs.

- The **actor** is $\pi_\theta(a \mid s)$. It chooses actions.
- The **critic** is $\hat V_\psi(s) \approx V^\pi(s)$. It scores states.

The critic never chooses anything. Its only purpose is to supply the baseline
that turns high-variance returns into low-variance advantages. It is trained by
regression:

$$
L_V(\psi) = \mathbb{E}\!\left[ \big( \hat V_\psi(s_t) - \hat G_t \big)^2 \right]
\tag{4.1}
$$

where $\hat G_t$ is some target for the return. Which target you use is exactly
the question this chapter answers.

In `ppo_from_scratch/ppo.py` the two are `ActorCritic.actor` and
`ActorCritic.critic` — separate MLPs, no shared trunk. The reasons are in the
class docstring and in chapter 6.

## 4.2 The TD error

The single most important quantity:

$$
\delta_t = r_t + \gamma \hat V(s_{t+1}) - \hat V(s_t)
\tag{4.2}
$$

*What actually happened, minus what I predicted would happen.*

If the critic were exact, the Bellman equation (2.6) says
$\mathbb{E}[\delta_t] = 0$: prediction and reality agree on average. A non-zero
$\delta_t$ is surprise, and surprise is information.

The key fact, and the reason $\delta$ appears everywhere: **if $\hat V = V^\pi$,
then $\delta_t$ is an unbiased estimate of $A^\pi(s_t, a_t)$.**

Proof. By definition $Q^\pi(s_t,a_t) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1})]$.
So

$$
\mathbb{E}[\delta_t] = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1})] - V^\pi(s_t)
= Q^\pi(s_t,a_t) - V^\pi(s_t) = A^\pi(s_t,a_t)
$$

That is the whole justification for one-step actor-critic.

## 4.3 $n$-step returns: the spectrum

$\delta_t$ uses one real reward and then trusts the critic. The Monte-Carlo
return uses all real rewards and never trusts the critic. Between them lies a
family:

$$
\begin{aligned}
\hat A_t^{(1)} &= r_t + \gamma \hat V(s_{t+1}) - \hat V(s_t) = \delta_t \\
\hat A_t^{(2)} &= r_t + \gamma r_{t+1} + \gamma^2 \hat V(s_{t+2}) - \hat V(s_t) = \delta_t + \gamma \delta_{t+1} \\
\hat A_t^{(3)} &= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} \\
&\;\;\vdots \\
\hat A_t^{(n)} &= \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}
\end{aligned}
\tag{4.3}
$$

The telescoping in 4.3 is worth verifying once by hand. For $n = 2$:

$$
\delta_t + \gamma\delta_{t+1}
= \big[r_t + \gamma \hat V_{t+1} - \hat V_t\big] + \gamma\big[r_{t+1} + \gamma \hat V_{t+2} - \hat V_{t+1}\big]
= r_t + \gamma r_{t+1} + \gamma^2 \hat V_{t+2} - \hat V_t
$$

The intermediate $\hat V_{t+1}$ cancels. Every $n$-step advantage is a
discounted sum of consecutive TD errors.

The trade-off along this family:

| | small $n$ | large $n$ |
|---|---|---|
| relies on | the critic | actual rewards |
| **bias** | high — inherits every error in $\hat V$ | low — real returns, no model |
| **variance** | low — one random reward | high — sum of $n$ random rewards |

Neither end is right. $n=1$ with a critic that is 20% wrong gives you a gradient
that is systematically 20% wrong. $n=\infty$ gives you an unbiased gradient so
noisy that you cannot take a useful step.

## 4.4 Generalised advantage estimation

GAE (Schulman et al., 2015) refuses to pick an $n$ and takes an
exponentially-weighted average of all of them:

$$
\hat A_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat A_t^{(n)}
\tag{4.4}
$$

The $(1-\lambda)$ normalises the weights, since $\sum_{n\ge1}\lambda^{n-1} = 1/(1-\lambda)$.

### Deriving the closed form

Substitute 4.3 into 4.4 and swap the order of summation:

$$
\begin{aligned}
\hat A_t
&= (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \sum_{l=0}^{n-1} \gamma^l \delta_{t+l} \\
&= (1-\lambda) \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} \sum_{n=l+1}^{\infty} \lambda^{n-1}
&& \text{(each } \delta_{t+l} \text{ appears in every } n > l\text{)} \\
&= (1-\lambda) \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} \cdot \frac{\lambda^{l}}{1-\lambda}
&& \textstyle \left(\sum_{n=l+1}^{\infty}\lambda^{n-1} = \lambda^l/(1-\lambda)\right) \\
&= \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}
\end{aligned}
$$

$$
\boxed{\;\hat A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}\;}
\tag{4.5}
$$

A single exponentially-discounted sum of TD errors. All the machinery collapses
to something you can compute in one backwards pass:

$$
\hat A_t = \delta_t + \gamma\lambda \, \hat A_{t+1}
\tag{4.6}
$$

which, with the episode-boundary mask, is exactly the loop in
`RolloutBuffer.compute_returns`:

```python
for t in reversed(range(self.n_steps)):
    next_non_terminal = 1.0 - self.dones[t + 1]
    next_value = self.values[t + 1]
    delta = rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
    last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    advantages[t] = last_gae
```

The `next_non_terminal` factor is what makes 4.6 respect episode boundaries: at
a real termination the future contributes nothing, so both the bootstrap and
the trace are zeroed.

### The full form with boundaries

$$
\hat A_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \left( \prod_{j=1}^{l} (1 - d_{t+j}) \right) \delta_{t+l}
\tag{4.7}
$$

where $d$ is the termination indicator. Equation 4.7 is the one implemented.

## 4.5 What $\lambda$ does

Two special cases recover the endpoints:

- $\lambda = 0$: $\hat A_t = \delta_t$. Pure one-step TD. Minimum variance,
  maximum reliance on the critic.
- $\lambda = 1$: $\hat A_t = \sum_l \gamma^l \delta_{t+l} = G_t - \hat V(s_t)$
  (the sum telescopes completely). Monte-Carlo. Unbiased, maximum variance.

So $\lambda$ is a **bias–variance dial**, and it is genuinely continuous: 0.95
means "mostly trust real rewards over the next $\sim 1/(1-\gamma\lambda) \approx 17$
steps, then hand over to the critic".

Note that $\gamma$ and $\lambda$ both discount, but they mean different things
and should not be conflated:

- $\gamma$ defines **the problem**: how much you care about the future. Changing
  it changes what "optimal" means.
- $\lambda$ defines **the estimator**: how much you trust your critic. Changing
  it changes only how you estimate the same objective.

This repository uses $\gamma = 0.99$, $\lambda = 0.95$, which is the standard
setting for continuous control and a good default. If you find yourself tuning
$\lambda$, the real question is usually whether your critic is any good — check
`explained_variance` first.

## 4.6 Value targets

The critic needs regression targets. From 4.6, the natural choice is

$$
\hat G_t = \hat A_t + \hat V(s_t)
\tag{4.8}
$$

which is the **$\lambda$-return**. Two things to notice.

First, this is a *bootstrapped* target: it is built partly from the critic's own
current predictions. Fitting a network to targets derived from itself is a
moving-target problem, and it is why value learning can oscillate in a way
supervised regression does not.

Second, using 4.8 rather than the Monte-Carlo return $G_t$ is what keeps the
critic's targets low-variance. It is the same bias–variance argument as for the
advantages, applied to the critic's own training data.

In code:

```python
returns = advantages + self.values
```

one line, immediately after the GAE loop.

## 4.7 Truncation, again

Equation 4.7 masks at $d_{t+j} = 1$. But as chapter 1 argued, "the episode
ended" and "the return stopped" are different statements.

At a **termination** (the robot fell), $V(s_T) = 0$ genuinely, and masking is
correct.

At a **truncation** (the 20-second limit), the return did not stop. Masking
tells the critic the future is worthless, which is false and which biases every
state near the time limit — and, through the Bellman recursion, states well
before it.

The correct treatment adds the bootstrap into the reward at the boundary:

$$
\tilde r_t = r_t + \gamma \hat V(s_{t}^{\text{final}}) \quad \text{if truncated at } t
\tag{4.9}
$$

and then masks the advantage recursion as usual. In
`ppo_from_scratch/ppo.py` this is the `truncated_values` array and the line

```python
rewards += gamma * self.truncated_values
```

Stable-Baselines3 does the same thing internally, keyed off the
`TimeLimit.truncated` info flag its vectorised wrappers insert. It was worth
verifying: this is a case where the original repository was *not* buggy, and
the check is recorded in chapter 14 alongside the bugs that were real.

## 4.8 Advantage normalisation

One more practical step, applied per minibatch just before the loss:

$$
\tilde A_i = \frac{\hat A_i - \operatorname{mean}(\hat A)}{\operatorname{std}(\hat A) + \epsilon}
\tag{4.10}
$$

This is *not* in the theory, and it introduces a small bias: the normalised
quantity is no longer an unbiased estimate of $A^\pi$. It is used anyway,
universally, because it makes the effective step size independent of the reward
scale.

That property matters a lot in this project specifically. Chapter 10 changes
reward weights constantly during tuning; without 4.10, every weight change
would rescale the gradient and require re-tuning the learning rate. With it,
the two are decoupled.

The mean-subtraction also guarantees the failure mode noted at the end of
chapter 3 — all advantages sharing a sign — cannot occur within a minibatch.

## 4.9 Putting it together

One PPO iteration, in full:

1. Run $\pi_\theta$ for $N$ steps in each of $E$ environments. Store
   $(s_t, a_t, r_t, \log\pi_\theta(a_t|s_t), \hat V(s_t), d_t)$.
2. Compute $\delta_t$ for every stored step (Eq 4.2), with the truncation
   correction (Eq 4.9).
3. Backwards pass for $\hat A_t$ (Eq 4.6).
4. Value targets $\hat G_t = \hat A_t + \hat V(s_t)$ (Eq 4.8).
5. Normalise advantages per minibatch (Eq 4.10).
6. Several epochs of minibatch gradient steps on the PPO objective — chapter 5.
7. Discard everything and go to 1.

Step 7 is what "on-policy" costs you: an expensive rollout is used for a handful
of gradient steps and then thrown away. Chapter 7 discusses whether that is
worth it.

---

**Previous:** [3. Policy gradients from first principles](03-policy-gradients.md) ·
**Next:** [5. From TRPO to PPO](05-trpo-to-ppo.md)
