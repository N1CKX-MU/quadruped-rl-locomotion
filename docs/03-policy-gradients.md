# 3. Policy gradients from first principles

This chapter derives $\nabla_\theta J(\theta)$ in full. It is the mathematical
core of the book: PPO is a modification of the result derived here, and if you
follow this chapter the rest is detail.

## 3.1 The obstacle

We want to maximise

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[ R(\tau) ],
\qquad R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t
$$

and to do gradient ascent we need $\nabla_\theta J$.

The difficulty is immediate. Write the expectation as an integral:

$$
J(\theta) = \int p_\theta(\tau) R(\tau) \mathrm{d}\tau
$$

so

$$
\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) R(\tau) \mathrm{d}\tau
\tag{3.1}
$$

$\theta$ appears in the **distribution we are sampling from**, not in the thing
we are averaging. Equation 3.1 is not an expectation over $p_\theta$ any more —
it is an integral against $\nabla_\theta p_\theta$, which is not a probability
distribution and which we cannot sample from. We cannot estimate it by running
the policy.

## 3.2 The log-derivative trick

The way out is one line of calculus. For any positive $p$,

$$
\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}
\quad\Longrightarrow\quad
\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)
\tag{3.2}
$$

Substituting into 3.1:

$$
\nabla_\theta J(\theta)
= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) \mathrm{d}\tau
= \mathbb{E}_{\tau \sim p_\theta}\left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right]
\tag{3.3}
$$

This is now an expectation over trajectories we can actually generate. Run the
policy, collect trajectories, average $\nabla_\theta \log p_\theta(\tau) R(\tau)$.

The trick has a name — the **score function estimator**, or REINFORCE — and it
is worth appreciating how much it buys for how little. It converts a gradient
of an expectation into an expectation of a gradient, which is the difference
between "impossible" and "a for loop".

## 3.3 The dynamics cancel

Equation 3.3 still looks unusable: $\log p_\theta(\tau)$ contains the transition
kernel $P$, which we do not know. Expand it, using equation 1.3:

$$
\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T-1} \Big[ \log \pi_\theta(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t) \Big]
$$

Now differentiate with respect to $\theta$. The initial-state term $\log p(s_0)$
does not depend on $\theta$. Neither does any $\log P$ term — the physics does
not care what our network weights are. Both vanish:

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
\tag{3.4}
$$

**This is why model-free RL works.** We never need $P$. The gradient depends on
the environment only through the rewards that the sampled trajectories happened
to receive. Substituting 3.4 into 3.3:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau}\left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) R(\tau) \right]
\tag{3.5}
$$

This is **REINFORCE** (Williams, 1992). It is a correct, unbiased estimator of
the policy gradient, and it is nearly useless in practice. The next two sections
explain why, and fix it.

## 3.4 Problem one: the future cannot change the past

In 3.5, every action in the trajectory is weighted by the **total** return
$R(\tau)$, including reward collected *before* that action was taken. An action
at $t = 300$ is credited with reward earned at $t = 5$.

That contributes nothing but noise, and we can prove it. Consider the
contribution of a single early reward $r_{t'}$ to the gradient term for a later
action $a_t$ with $t > t'$:

$$
\mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot r_{t'} \right]
$$

Condition on everything up to and including $s_t$. Then $r_{t'}$ is a fixed
number, and

$$
\mathbb{E}_{a_t \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
= \int \pi_\theta(a \mid s_t) \frac{\nabla_\theta \pi_\theta(a \mid s_t)}{\pi_\theta(a \mid s_t)} \mathrm{d}a
= \nabla_\theta \int \pi_\theta(a \mid s_t) \mathrm{d}a
= \nabla_\theta 1 = 0
\tag{3.6}
$$

The expected score is zero. So the cross-term has expectation zero: it is
unbiased noise, pure variance with no signal. Dropping it gives the
**reward-to-go** form:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right],
\qquad G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k
\tag{3.7}
$$

Equation 3.6 is worth remembering on its own. It is used again immediately, and
again in chapter 5.

## 3.5 Problem two: baselines

Even with reward-to-go, variance is brutal. $G_t$ for this robot is a sum of a
hundred-odd noisy terms; two runs of the *same* policy from the *same* state
can differ substantially because of contact chaos.

The fix uses 3.6 again. Subtract any function $b(s_t)$ that does not depend on
$a_t$:

$$
\mathbb{E}_{a_t}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) b(s_t) \right]
= b(s_t) \mathbb{E}_{a_t}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = 0
$$

So for **any** state-dependent baseline $b$:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \big( G_t - b(s_t) \big) \right]
\tag{3.8}
$$

The estimator stays unbiased for every choice of $b$, but its **variance
changes**. Choosing $b$ well is free variance reduction.

### The optimal baseline

Minimise the variance of the estimator with respect to $b$. Writing
$g = \nabla_\theta \log \pi_\theta(a \mid s)$ and treating one component at a time,

$$
\operatorname{Var}\left[ g (G - b) \right] = \mathbb{E}\left[ g^2 (G-b)^2 \right] - \left( \mathbb{E}[g(G-b)] \right)^2
$$

The second term does not depend on $b$ (it is the unbiased gradient). Setting
$\partial / \partial b$ of the first term to zero:

$$
-2\mathbb{E}\left[ g^2 (G - b) \right] = 0
\quad\Longrightarrow\quad
b^\star = \frac{\mathbb{E}\left[ g^2 G \right]}{\mathbb{E}\left[ g^2 \right]}
\tag{3.9}
$$

— a score-weighted average return. In practice nobody uses 3.9: it needs
per-parameter baselines and second moments of the score. The near-optimal,
practical choice is the unweighted average return from that state, which is
exactly

$$
b(s) = V^\pi(s)
$$

And with that choice, $G_t - V^\pi(s_t)$ is an unbiased sample of
$Q^\pi(s_t,a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$. The advantage function from
chapter 2 appears, not as a heuristic, but as the natural consequence of
optimal variance reduction:

$$
\boxed{\quad 
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) A^\pi(s_t, a_t) \right]
\quad }
\tag{3.10}
$$

This is the **policy gradient theorem** in the form every modern algorithm
uses.

## 3.6 Reading the gradient

Equation 3.10 has an interpretation worth internalising, because it explains
every training pathology you will meet.

$\nabla_\theta \log \pi_\theta(a \mid s)$ points in the direction of parameter
space that makes $a$ **more likely** in $s$. Multiplying by $A$:

- $A > 0$ (better than the policy's average): step in that direction. Make this
  action more likely.
- $A < 0$: step against it. Make this action less likely.
- $A \approx 0$: no update.

So the policy gradient is: *do more of what worked, less of what did not,
where "worked" means "beat my own average".*

Two corollaries:

**If the critic is biased, the policy learns the bias.** $A$ is estimated as
$G_t - \hat V(s_t)$. If $\hat V$ systematically underestimates value in some
region, every action in that region gets a positive advantage and is reinforced
regardless of merit. This is why `explained_variance` is the number to watch.

**If all advantages have the same sign, nothing useful happens.** The policy
pushes uniformly toward every action it took, which mostly just inflates or
deflates the entropy. In practice advantages are standardised per minibatch
(see `normalize_advantage` in `ppo_from_scratch/ppo.py`) partly to guarantee
this cannot happen.

## 3.7 The loss you actually write in PyTorch

Autodiff frameworks minimise, and they differentiate a scalar. You do not code
equation 3.10 directly — you code a surrogate scalar whose gradient equals it:

$$
L(\theta) = - \frac{1}{N} \sum_{i} \log \pi_\theta(a_i \mid s_i) \hat A_i
\tag{3.11}
$$

with $\hat A_i$ treated as a **constant** (detached from the graph). Then
$\nabla_\theta L$ is exactly minus 3.10, and gradient descent on $L$ is gradient
ascent on $J$.

$L$ is not a loss in any meaningful sense. Its value is not a measure of
performance; it can be any sign and it does not decrease as training improves.
Watching `policy_loss` go down tells you nothing. This is the single most
common misreading of an RL training log.

## 3.8 Why REINFORCE alone is not enough

Everything so far is correct and still insufficient. Three problems remain, and
they define the next two chapters.

**1. $A^\pi$ is unknown.** Equation 3.10 needs the true advantage. Using the
Monte-Carlo estimate $G_t - \hat V(s_t)$ is unbiased but high-variance; using a
one-step TD estimate is low-variance but biased by the critic's error. Chapter 4
(GAE) is the principled interpolation.

**2. The data is stale after one step.** Equation 3.10 is an expectation under
$\pi_\theta$ — the *current* policy. As soon as you take one gradient step, your
collected trajectories were generated by a different policy and the estimator is
no longer valid. Strict on-policy learning therefore permits one update per
rollout, which is a catastrophic waste of expensive simulation. Chapter 5
(importance sampling and PPO) is how you get many updates out of one rollout
without the estimator falling apart.

**3. Step size is unbounded and consequences are asymmetric.** In supervised
learning a too-large step gives a bad prediction and the next batch corrects it.
In RL a too-large step changes the policy, which changes the *data distribution*
you collect next. A policy that becomes bad enough collects only data about
falling over, and there is no gradient back to competence. This is why RL needs
trust regions and supervised learning does not — again chapter 5.

---

**Previous:** [2. Value functions and policies](02-value-functions-and-policies.md) ·
**Next:** [4. Actor-critic and GAE](04-actor-critic-and-gae.md)
