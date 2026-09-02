# 1. What reinforcement learning actually is

## 1.1 The problem it solves

Supervised learning needs, for every input, the right answer. If you want a
network that maps images to labels, you need labelled images.

Now consider the problem in this repository. The input is the robot's sensor
state — joint angles, body attitude, the commanded velocity. The output is
twelve joint position targets, fifty times a second. What is the "right answer"
for the front-left knee at $t = 3.42$ s?

Nobody knows. There is no dataset of correct knee angles. There is not even a
unique correct answer: many different joint trajectories produce a perfectly
good trot, and which one is best depends on the terrain, the payload, and the
robot's own momentum a fraction of a second earlier.

What we *can* say is whether the outcome was good. Did it move at the commanded
velocity? Did it stay upright? Did it use a reasonable amount of torque? That
is a much weaker signal than a label — it evaluates the result of a whole
sequence of decisions, and it does not say which decision was responsible — but
it is the only signal available, and reinforcement learning is the set of
techniques for learning from exactly that.

The core difficulty, and the thing that makes RL genuinely harder than
supervised learning, is **credit assignment**. The robot falls at $t = 4.0$ s.
The mistake was probably a badly-timed foot placement at $t = 3.6$ s. Nothing
in the reward signal says so. Everything in this book is, one way or another,
machinery for solving that problem.

## 1.2 Markov decision processes

The formalisation is a **Markov decision process** (MDP), a five-tuple:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

- $\mathcal{S}$ is the **state space**: everything the world can be.
- $\mathcal{A}$ is the **action space**: everything the agent can do.
- $P(s' \mid s, a)$ is the **transition kernel**: the probability of landing in
  $s'$ after taking $a$ in $s$.
- $R(s, a, s')$ is the **reward function**: a scalar emitted on each transition.
- $\gamma \in [0, 1)$ is the **discount factor**.

The word *Markov* is doing real work. It asserts that

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t)
\tag{1.1}
$$

— the future depends on the past **only through the current state**. If that
holds, a policy that looks only at $s_t$ loses nothing, and we can search over
functions of one state instead of functions of entire histories.

This is not a technicality you can wave through. It is a design constraint on
what you put in the observation, and it is why chapter 9 spends so long
justifying each of the fifty numbers. A concrete example from this repository:
the gait clock $\phi$ is part of the observation. If it were not, two moments
with identical joint angles but different points in the stride cycle would look
identical to the policy while having genuinely different correct actions — and
the process would stop being Markov in the observed variables.

### An important caveat: this is a POMDP

Strictly, a robot does not observe the state. It observes sensors. The true
state includes ground friction, the exact mass distribution, actuator
temperature — none of which appear in $o_t$. That makes this a *partially
observable* MDP, and the honest position is that we are approximating a POMDP
with an MDP over observations.

Two things make the approximation work in practice:

1. **Domain randomisation.** If the hidden variables are randomised across
   episodes, the policy is forced to learn a control law that works for all of
   them rather than exploiting one particular value. See chapter 12.
2. **Enough observable state.** The Go2's proprioception — joint angles,
   velocities, and body attitude — determines the immediate dynamics well
   enough that the residual uncertainty behaves like noise rather than like a
   hidden mode.

## 1.3 Policies

A **policy** is a rule for choosing actions. Deterministic:

$$
a_t = \mu_\theta(s_t)
$$

Stochastic — a distribution:

$$
a_t \sim \pi_\theta(\cdot \mid s_t)
$$

Everything in this book uses stochastic policies, specifically diagonal
Gaussians:

$$
\pi_\theta(a \mid s) = \mathcal{N}\left(a \quad ;\quad \mu_\theta(s), \operatorname{diag}(\sigma^2)\right)
\tag{1.2}
$$

where the mean $\mu_\theta(s)$ is a neural network and $\log \sigma$ is a free,
learnable parameter vector independent of $s$.

Why stochastic? Three reasons, in increasing order of importance:

1. **Exploration.** A deterministic policy in a continuous action space visits a
   measure-zero slice of the possibilities. Randomness is how the agent
   discovers that a slightly different foot placement works better.
2. **The gradient exists.** The policy gradient theorem (chapter 3) is a
   statement about $\nabla_\theta \log \pi_\theta(a \mid s)$. A deterministic
   policy has no such quantity — it needs a different derivation entirely (the
   deterministic policy gradient, which is what DDPG and TD3 use).
3. **Smoothness.** Optimising a distribution smooths the objective. The
   expected return of a stochastic policy is differentiable in $\theta$ even
   when the environment's dynamics are full of discontinuous contact events —
   and a legged robot is nothing but discontinuous contact events.

Why *diagonal* Gaussian? A full covariance would let the policy correlate joint
noise, which is arguably more physical. It also costs $O(n^2)$ parameters and
makes the entropy and log-probability computations substantially more
expensive, for a benefit nobody has convincingly demonstrated on this class of
task.

Why is $\sigma$ **state-independent**? A state-dependent $\sigma_\theta(s)$ is
strictly more expressive. It is also strictly more dangerous: it lets the
policy collapse its own exploration in exactly the states where exploration
matters — the difficult ones — because reducing variance there raises expected
return in the short run. The state-independent version anneals globally as
training proceeds, which is the behaviour you want.

## 1.4 Trajectories and return

Running a policy generates a **trajectory**:

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)
$$

with probability

$$
p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)
\tag{1.3}
$$

Equation 1.3 matters more than it looks. Notice that $\theta$ appears **only**
in the $\pi_\theta$ factors, not in $p(s_0)$ or $P$. That single observation is
what makes model-free RL possible: when we differentiate $\log p_\theta(\tau)$,
every term involving the unknown environment dynamics disappears. Chapter 3
turns that into an algorithm.

The **return** is the discounted sum of future rewards:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
\tag{1.4}
$$

and the objective is to maximise its expectation:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\left[ G_0 \right]
= \mathbb{E}_{\tau \sim p_\theta}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
\tag{1.5}
$$

That is the whole problem statement. Every algorithm in this book is a way of
estimating $\nabla_\theta J(\theta)$ from samples.

## 1.5 Why discount

$\gamma$ has three distinct justifications, and it is worth separating them
because they suggest different values.

**Mathematical.** With $|r| \le r_\max$ and $\gamma < 1$, equation 1.4 converges:
$|G_t| \le r_\max / (1 - \gamma)$. Without discounting, the return of a
non-terminating task is $\infty$ and the objective is meaningless.

**Statistical.** The estimate of $G_t$ from a single trajectory has variance
that grows with the number of terms contributing to it. Discounting truncates
the effective horizon, and hence the variance. This is a bias–variance trade:
a smaller $\gamma$ gives a lower-variance, more biased estimate of the true
infinite-horizon objective.

**Practical.** $\gamma$ sets the **effective horizon** — how far ahead the agent
plans. The weight on a reward $k$ steps away is $\gamma^k$, so the horizon is
roughly $1/(1 - \gamma)$ steps.

For this repository: $\gamma = 0.99$ at 50 Hz gives

$$
\frac{1}{1 - 0.99} = 100 \text{ steps} = 2.0 \text{ seconds}
$$

Two seconds is about four strides at 2 Hz. That is the right order of magnitude
for locomotion: the consequences of a foot placement play out over the next
stride or two, not over the next thirty seconds. If you raised $\gamma$ to
0.999 (20 s horizon) you would not gain foresight, you would gain variance —
the critic would be trying to predict twenty seconds of a chaotic contact
process from one state.

Note the interaction with control rate, which is easy to get wrong. v1 ran at
25 Hz with $\gamma = 0.99$, giving a 4-second horizon; v2 runs at 50 Hz with the
same $\gamma$, giving 2 seconds. Doubling the control rate **halved** the
effective planning horizon. If you change `decimation`, you have implicitly
changed $\gamma$'s meaning.

## 1.6 Episodes, termination, and truncation

Real training runs episodes: reset, act until something ends it, reset. Two
different things can end an episode, and conflating them is a genuine bug that
appears in a lot of code.

**Termination** — the MDP genuinely reached an absorbing state. The robot fell
over. There is no future reward, so $G_t$ really does stop. The correct value
target uses $V(s_{T}) = 0$.

**Truncation** — we stopped the episode for our own convenience, usually a time
limit. The robot was walking perfectly well at second 20 and would have carried
on. The return did *not* stop; we simply stopped watching.

If you treat truncation as termination, you tell the critic that every state 20
seconds into an episode has zero future value. That bias propagates backwards
through the Bellman recursion and corrupts value estimates everywhere, and it
looks like nothing more specific than "training is a bit worse than it should
be".

The fix is to bootstrap: on truncation, add $\gamma V(s_T)$ to the final reward
and treat the boundary as a break in the advantage recursion only. In this
repository:

- Gymnasium's API distinguishes them: `step()` returns `terminated` and
  `truncated` separately. `Go2Env.step` sets `terminated` from
  `_check_termination` and `truncated` from the step counter.
- Stable-Baselines3's vectorised wrappers convert this into a
  `TimeLimit.truncated` key in the info dict and handle the bootstrap for you.
- `ppo_from_scratch/ppo.py` does it explicitly, in `RolloutBuffer.compute_returns`
  and in the `truncated_values` array — go and read that code, because it is
  the clearest place in the repository to see the distinction made concrete.

## 1.7 This robot's MDP, written out

To make all of the above stop being abstract, here is the actual MDP.

**State** (as observed; 50 dimensions — see chapter 9):

$$
o_t = \big[\quad 
\underbrace{g_b}_{3} \quad 
\underbrace{\omega_b}_{3} \quad 
\underbrace{v_b}_{3} \quad 
\underbrace{q - q_\text{nom}}_{12} \quad 
\underbrace{\dot q}_{12} \quad 
\underbrace{a_{t-1}}_{12} \quad 
\underbrace{c}_{3} \quad 
\underbrace{\sin 2\pi\phi, \cos 2\pi\phi}_{2}
\quad \big]
$$

where $g_b$ is gravity in the base frame, $\omega_b$ and $v_b$ are base-frame
angular and linear velocity, $q$ the joint angles, $c$ the commanded velocity
$(v_x^*, v_y^*, \omega_z^*)$, and $\phi$ the gait phase.

**Action** ($\mathcal{A} = [-1, 1]^{12}$): joint position offsets. The applied
target is

$$
q^*_t = q_\text{nom} + \alpha a_t, \qquad \alpha = 0.40 \text{ rad}
$$

and a PD controller converts that to torque at each physics substep:

$$
\tau = k_p (q^* - q) - k_d \dot q, \qquad k_p = 55, \quad k_d = 1.4
\tag{1.6}
$$

**Transitions**: MuJoCo, 2 ms physics step, 10 substeps per control step.
Stochastic across episodes because of randomised initial conditions, randomised
friction and mass, and randomly-timed external pushes.

**Reward**: seventeen weighted terms, summed and scaled by $\Delta t$. Chapter 10.

**Discount**: $\gamma = 0.99$, i.e. a 2-second horizon.

**Termination**: trunk contacts the ground, or base height $< 0.12$ m, or tilt
exceeds 0.8 rad.

**Truncation**: 1000 control steps, i.e. 20 seconds.

## 1.8 What makes this problem hard

Worth stating plainly, because it explains why so much of the rest of this book
is about reward design rather than about the optimiser.

**The action space is continuous and 12-dimensional.** Enumerating actions is
out. Everything must be gradient-based.

**Rewards are dense but misleading.** We can score every step, which is much
better than a sparse "did you reach the goal" signal. But every dense reward is
a *proxy* for what you want, and the optimiser will find the gap between the
proxy and your intent. Chapter 10 has a worked example from this very
repository where the gait reward was maximised by a robot standing perfectly
still.

**The dynamics are contact-rich and stiff.** Feet make and break contact
constantly. The dynamics are effectively discontinuous, the value function has
sharp structure, and small action changes can flip a contact event.

**Failures are terminal and early.** An untrained policy falls in under a
second. Almost all early data is about falling over, which is exactly the
regime where the reward signal is least informative about what walking looks
like.

**The behaviour must be periodic.** Locomotion is a limit cycle. Nothing in the
MDP formulation prefers periodic solutions, and a policy optimising a
velocity-tracking reward will happily find aperiodic, twitching gaits that
track velocity acceptably and look wrong. Chapter 11 is about putting the
periodicity in explicitly rather than hoping for it.

---

**Next:** [2. Value functions and policies](02-value-functions-and-policies.md)
— what we need to estimate before we can compute a gradient.
