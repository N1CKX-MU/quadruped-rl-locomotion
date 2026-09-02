# 5. From TRPO to PPO

Chapter 4 gave us an advantage estimate. Chapter 3 gave us a gradient that uses
it. What remains is the question that actually separates working RL from
non-working RL: **how far should one update move the policy?**

## 5.1 Why step size is a first-class problem

In supervised learning, an over-large step produces a bad model, the next batch
produces a corrective gradient, and you recover. The data distribution is fixed;
it does not care how bad your model is.

In RL the policy *generates the data*. Take too large a step, and:

1. The new policy is worse.
2. It collects worse data — data about a worse region of state space.
3. That data is used to compute the next update.

There is no fixed distribution to fall back on. A policy that degrades enough
collects only trajectories of the robot falling over in the first half-second,
and there is no gradient in that data pointing back toward walking. The failure
is not slow convergence — it is a one-way collapse.

Hence: constrain the size of the update, measured in a way that reflects how
much the *behaviour* changed, not how much the *parameters* changed.

That distinction is the crux. Parameter distance is meaningless — the same
Euclidean step in $\theta$ can leave a policy almost unchanged or destroy it,
depending on where you are. What we care about is distance between the
distributions $\pi_{\theta_\text{old}}$ and $\pi_\theta$.

## 5.2 Off-policy evaluation via importance sampling

Before the trust region, one more tool. We want to estimate a quantity under
$\pi_\theta$ using samples drawn from $\pi_{\theta_\text{old}}$. Importance
sampling does exactly that:

$$
\mathbb{E}_{x \sim q}[f(x)] = \int q(x) f(x) \mathrm{d}x
= \int p(x) \frac{q(x)}{p(x)} f(x) \mathrm{d}x
= \mathbb{E}_{x \sim p}\left[ \frac{q(x)}{p(x)} f(x) \right]
\tag{5.1}
$$

valid wherever $p > 0$ where $q > 0$. Applied to the policy gradient, define the
**importance ratio**

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}
\tag{5.2}
$$

and the policy gradient (3.10) becomes computable from old data:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}}\left[ \sum_t \rho_t(\theta) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat A_t \right]
\tag{5.3}
$$

This is what buys us multiple epochs per rollout. The catch: importance sampling
is only well-behaved when $p$ and $q$ are close. As $\pi_\theta$ drifts from
$\pi_{\theta_\text{old}}$, the ratios spread out, and the variance of the
estimator grows without bound — a single sample with $\rho = 40$ dominates the
entire minibatch.

So we have arrived at the same place from two directions. Both the data-collapse
argument and the importance-sampling-variance argument say: *keep the new policy
close to the old one*.

### A note on computing the ratio

Never compute 5.2 as a ratio of densities. For a 12-dimensional diagonal
Gaussian, $\pi(a|s)$ is a product of twelve densities and underflows float32
routinely. Compute it in log space:

```python
ratio = torch.exp(log_prob - mb["log_probs"])
```

which is numerically stable because the log-probabilities are $O(10)$ rather
than $O(10^{-20})$.

## 5.3 The surrogate objective

Define the **conservative policy iteration** objective:

$$
L^{\text{CPI}}(\theta) = \mathbb{E}_t\left[ \rho_t(\theta) \hat A_t \right]
\tag{5.4}
$$

Note $\nabla_\theta L^{\text{CPI}}\big|_{\theta = \theta_\text{old}}$ equals the
policy gradient 3.10, because $\rho_t(\theta_\text{old}) = 1$ and
$\nabla \rho_t = \rho_t \nabla \log \pi_\theta$. So 5.4 is a *local* stand-in
for $J$: correct to first order at the old policy, and computable from old data.

Maximising 5.4 without constraint is disastrous, and it is worth seeing exactly
why. If $\hat A_t > 0$ for some sample, the objective increases without bound as
$\pi_\theta(a_t \mid s_t) \to 1$. The optimiser will happily drive the policy to
put all its mass on whichever handful of actions happened to have positive
estimated advantage in this rollout — advantage estimates that are noisy, and
that were only valid near $\theta_\text{old}$ anyway.

## 5.4 TRPO: an explicit trust region

Trust Region Policy Optimization (Schulman et al., 2015) constrains the update
by KL divergence:

$$
\begin{aligned}
\max_\theta \quad & \mathbb{E}_t\left[ \rho_t(\theta) \hat A_t \right] \\
\text{s.t.} \quad & \mathbb{E}_t\left[ D_{\mathrm{KL}}\big( \pi_{\theta_\text{old}}(\cdot \mid s_t) \Vert \pi_\theta(\cdot \mid s_t) \big) \right] \le \delta
\end{aligned}
\tag{5.5}
$$

KL divergence is the right measure because it is about distributions, not
parameters. It is invariant to how you happen to parameterise the policy.

TRPO's theoretical backing is real: there is a monotonic improvement guarantee
of the form

$$
J(\pi_\text{new}) \ge L^{\text{CPI}}(\pi_\text{new}) - C \cdot \max_s D_{\mathrm{KL}}
$$

so staying inside the trust region guarantees you do not make things worse.

The problem is the machinery. Solving 5.5 requires a second-order method: build
the Fisher information matrix (the local quadratic approximation to KL),
compute a natural gradient step by conjugate gradient with
Hessian-vector products, then line-search the step size to enforce the
constraint exactly. It works, it is expensive, and it is fiddly enough that
small implementation differences change the results.

## 5.5 PPO: the same idea, by clipping

Proximal Policy Optimization (Schulman et al., 2017) asks: can we get the effect
of the trust region using only first-order optimisation?

The insight is that we do not need to *forbid* large steps. We only need to
remove the *incentive* to take them.

$$
\boxed{\quad 
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[ \min\Big( \rho_t(\theta) \hat A_t, \quad \quad \operatorname{clip}\big(\rho_t(\theta), 1-\varepsilon, 1+\varepsilon\big) \hat A_t \Big) \right]
\quad }
\tag{5.6}
$$

with $\varepsilon = 0.2$ typically. Read it case by case — this is the part
worth being able to reconstruct from memory.

**Case $\hat A_t > 0$** (the action was better than average; we want it more
likely, i.e. $\rho$ to increase):

- $\rho < 1 + \varepsilon$: the clip is inactive, both branches equal
  $\rho \hat A$, gradient flows normally.
- $\rho > 1 + \varepsilon$: the clipped branch is
  $(1+\varepsilon)\hat A$, a constant. The min picks it (it is smaller). The
  gradient with respect to $\theta$ is **zero**.

So once the action is already $1+\varepsilon$ times more likely than it was, no
further reward for making it more likely still. Not a penalty — an absence of
incentive.

**Case $\hat A_t < 0$** (we want $\rho$ to decrease):

- $\rho > 1 - \varepsilon$: unclipped, gradient flows.
- $\rho < 1 - \varepsilon$: the clipped branch is $(1-\varepsilon)\hat A$, which
  since $\hat A < 0$ is *larger* than $\rho\hat A$. The min picks the
  **unclipped** branch, so the gradient keeps flowing.

That asymmetry is deliberate and is the point of the `min` rather than a plain
clip. If an action turned out to be much worse than expected, the policy is
allowed to keep pushing it down without limit. Being pessimistic is always safe;
being optimistic is what needs restraining.

### The pessimism framing

Another way to see 5.6: $L^{\text{CLIP}}$ is a **lower bound** on
$L^{\text{CPI}}$. The `min` always selects the more conservative of the two
estimates. PPO maximises a pessimistic bound on the improvement, which is why it
tends to be stable even when the advantage estimates are poor.

## 5.6 The full objective

In practice three terms are optimised jointly:

$$
L(\theta, \psi) = \underbrace{-L^{\text{CLIP}}(\theta)}_{\text{policy}}
+ c_v \underbrace{L^{V}(\psi)}_{\text{value}}
- c_e \underbrace{\mathbb{E}_t\big[ \mathcal{H}[\pi_\theta(\cdot \mid s_t)] \big]}_{\text{entropy}}
\tag{5.7}
$$

(negated, because frameworks minimise.)

### The value loss, and clipping it too

$$
L^{V}(\psi) = \mathbb{E}_t \Big[ \max\Big( (\hat V_\psi(s_t) - \hat G_t)^2, \quad (\hat V^{\text{clip}}_\psi(s_t) - \hat G_t)^2 \Big) \Big]
\tag{5.8}
$$

$$
\hat V^{\text{clip}}_\psi(s_t) = \hat V_{\psi_\text{old}}(s_t) + \operatorname{clip}\big( \hat V_\psi(s_t) - \hat V_{\psi_\text{old}}(s_t), -\varepsilon_v, +\varepsilon_v \big)
$$

Same motivation as the policy clip. The critic is fit to targets built from its
own old predictions (Eq 4.8), so a large critic update makes the very targets it
was fit to stale, and the remaining epochs of this update are optimising against
a value function that no longer exists. The `max` again picks the pessimistic
branch.

$c_v = 0.5$ here. This coefficient trades off actor and critic learning; too
large and the shared optimiser budget goes to the critic, too small and the
critic lags and the advantages are noise.

### The entropy bonus

$$
\mathcal{H}[\pi] = -\int \pi(a \mid s) \log \pi(a \mid s) \mathrm{d}a
$$

For a diagonal Gaussian this has a closed form:

$$
\mathcal{H} = \sum_{i=1}^{n} \left( \log \sigma_i + \tfrac{1}{2}\log(2\pi e) \right)
\tag{5.9}
$$

so maximising entropy simply pushes $\log\sigma$ up. It counteracts premature
collapse: without it, the policy's variance shrinks monotonically (lower
variance means fewer bad actions means higher immediate return) and exploration
dies before a good gait is found.

The coefficient matters more than it looks. v1 used $c_e = 0.01$; v2 uses
0.002. At $c_e = 0.01$, with reward magnitudes around $0.5$ per step, the
entropy bonus is a large fraction of the objective, and the policy is paid to
stay noisy — which for a legged robot means it never settles into the clean,
low-variance limit cycle a trot requires. The symptom is a policy that keeps
walking but never stops looking twitchy, with `std` in the logs refusing to come
down.

## 5.7 Approximate KL and early stopping

Even with clipping, it is worth *measuring* how far the policy moved. The naive
estimator

$$
\widehat{\mathrm{KL}} = \mathbb{E}_t[-\log \rho_t]
$$

is unbiased but can go negative on a finite sample, which is embarrassing for a
divergence. Schulman's low-variance alternative:

$$
\widehat{\mathrm{KL}} = \mathbb{E}_t\left[ (\rho_t - 1) - \log \rho_t \right]
\tag{5.10}
$$

This is non-negative for every sample, since $x - 1 - \log x \ge 0$ for all
$x>0$ with equality at $x=1$. It is also lower variance. It is what appears as
`approx_kl` in both SB3's logs and ours.

PPO implementations typically use it to **early-stop** an update: if the
measured KL exceeds some multiple of a target (here, $1.5 \times 0.02$), abandon
the remaining epochs. The reasoning is the importance-sampling one from 5.2 —
past that point the collected data is simply too off-policy to learn from,
clipping or not.

```python
if approx_kl > 1.5 * self.cfg.target_kl:
    early_stopped = True
    break
```

## 5.8 Why multiple epochs work at all

PPO takes $K$ epochs of minibatch SGD on one rollout. Chapter 3 said the policy
gradient is only valid for the policy that collected the data. So why is this
not simply wrong?

Because 5.6 is not the policy gradient — it is a surrogate objective that
*remains meaningful* for policies near $\theta_\text{old}$, via the importance
ratio, and that *stops rewarding movement* once the policy has drifted past
$1 \pm \varepsilon$. The first epoch is nearly on-policy; by the fifth, many
samples are clipped and contributing no gradient at all, which is exactly the
intended behaviour. The algorithm degrades gracefully into doing nothing rather
than into doing something wrong.

This is why `clip_fraction` is a useful diagnostic and why it should be
non-zero. See chapter 13.

## 5.9 The hyperparameters, and this repository's values

| Symbol | Name | v2 value | v1 value | Comment |
|---|---|---|---|---|
| $\varepsilon$ | `clip_range` | 0.2 | 0.2 | The standard value; rarely worth changing |
| $\varepsilon_v$ | `clip_range_vf` | 0.2 | none | Value clipping, added in v2 |
| $c_v$ | `vf_coef` | 0.5 | 0.5 | |
| $c_e$ | `ent_coef` | 0.002 | 0.01 | v1's was high enough to prevent the policy settling |
| $K$ | `n_epochs` | 5 | 10 | |
| $N$ | `n_steps` | 512 | 2048 | per environment |
| $E$ | `n_envs` | 16 | 8 | |
| | rollout | 8192 | 16384 | $N \times E$ |
| $B$ | `batch_size` | 2048 | 64 | |
| | grad steps/rollout | 20 | **2560** | $K \cdot N E / B$ |
| | `target_kl` | 0.02 | none | |
| | `max_grad_norm` | 1.0 | 0.5 | |

The row worth staring at is **grad steps per rollout**. v1 did 2560 gradient
steps on a single rollout of data. No amount of clipping keeps a policy near
$\theta_\text{old}$ across 2560 updates; the clip fraction saturates, most
samples contribute nothing, and the compute is spent for nothing. It is
simultaneously a stability risk and a large waste of time. See chapter 14, B14.

## 5.10 The complete algorithm

```
initialise theta, psi
for iteration = 1, 2, ...:
    # --- collect ---
    for t = 1..N in each of E environments:
        a_t ~ pi_theta(. | s_t)
        step the environment, store (s, a, r, logp, V, done)

    # --- estimate (chapter 4) ---
    delta_t  = r_t + gamma V(s_{t+1})(1 - d_{t+1}) - V(s_t)
    A_t      = delta_t + gamma lambda (1 - d_{t+1}) A_{t+1}      backwards
    G_t      = A_t + V(s_t)

    # --- optimise (this chapter) ---
    for epoch = 1..K:
        for each minibatch:
            rho    = exp(logpi_theta(a|s) - logp_old)
            A      = (A - mean A) / std A
            L_clip = -min(rho A, clip(rho, 1-eps, 1+eps) A)
            L_v    = max((V - G)^2, (V_clip - G)^2)
            L      = L_clip + c_v L_v - c_e H
            step Adam on L, with gradient norm clipped
            if approx_kl > 1.5 * target: stop
```

That is the entire algorithm. `ppo_from_scratch/ppo.py` is this, plus
observation normalisation, plus bookkeeping. Chapter 6 walks through it.

---

**Previous:** [4. Actor-critic and GAE](04-actor-critic-and-gae.md) ·
**Next:** [6. PPO line by line](06-ppo-line-by-line.md)
