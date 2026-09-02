# 2. Value functions and policies

Chapter 1 ended with the objective:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

To improve it we need to answer a question the reward signal does not answer
directly: *given where I am, how well am I doing?* Value functions are that
answer, and the advantage function — the difference between "how well am I
doing" and "how well would I do if I took this particular action" — is the
quantity every algorithm in this book ultimately estimates.

## 2.1 The state-value function

$$
V^\pi(s) = \mathbb{E}_{\tau \sim \pi}\left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \quad \middle|\quad s_t = s \right]
\tag{2.1}
$$

In words: start in $s$, follow $\pi$ forever, average the discounted return over
all the randomness in the policy and the dynamics.

$V^\pi$ depends on $\pi$. This is not a technicality — it is the source of most
confusion about value-based methods. Standing upright with legs folded is a
great state if your policy knows how to push off from it, and a terrible one if
your policy will fall over from it. The value of a state is a property of the
state *and the behaviour*, jointly.

For the Go2, in a policy that walks well, $V^\pi$ is roughly "expected reward
per step $\times$ 100" — around $0.5 \times 100 = 50$ for a policy earning 0.5
per step, less near states from which a fall is likely.

## 2.2 The action-value function

$$
Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi}\left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \quad \middle|\quad s_t = s, a_t = a \right]
\tag{2.2}
$$

Same thing, except the first action is forced to be $a$ and only afterwards do
we follow $\pi$. The two are related by averaging over the first action:

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}\left[ Q^\pi(s, a) \right]
\tag{2.3}
$$

## 2.3 The advantage function

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
\tag{2.4}
$$

*How much better than average is this action, in this state, under this policy?*

Equation 2.3 gives the property that makes the advantage useful:

$$
\mathbb{E}_{a \sim \pi(\cdot \mid s)}\left[ A^\pi(s, a) \right] = 0
\tag{2.5}
$$

The advantage is zero-mean under the policy's own action distribution. Half the
actions the policy takes are better than its average; half are worse.

This is the single most important object in policy-gradient RL, for one reason:
it is **centred and scale-free with respect to the state**. Consider two states,
one where every action leads to a return around 50 and one where every action
leads to a return around 5. The raw returns differ by an order of magnitude and
tell you almost nothing about which action to take. The advantages in both
states are centred on zero and tell you precisely that.

Concretely for the Go2: suppose the robot is mid-stride with the front-left
foot about to touch down. $V^\pi(s) \approx 48$. An action that plants the foot
cleanly gives $Q \approx 48.4$; an action that plants it half a stride early
gives $Q \approx 46.1$. The advantages are $+0.4$ and $-1.9$. Those are the
numbers that should drive the gradient, and the fact that both raw $Q$ values
are "about 48" is exactly the information we want to discard.

## 2.4 The Bellman equations

Value functions satisfy recursions, which is what lets us estimate them without
simulating to infinity.

Split off the first term of the sum in equation 2.1:

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi,\quad s' \sim P}\left[ r(s, a, s') + \gamma V^\pi(s') \right]
\tag{2.6}
$$

and similarly

$$
Q^\pi(s, a) = \mathbb{E}_{s' \sim P}\left[ r(s, a, s') + \gamma \mathbb{E}_{a' \sim \pi}\left[ Q^\pi(s', a') \right] \right]
\tag{2.7}
$$

These are the **Bellman expectation equations**. Each says the same structural
thing: the value of now equals the immediate reward plus the discounted value
of next.

### Derivation of 2.6

Starting from 2.1 and writing $G_t = r_t + \gamma G_{t+1}$:

$$
\begin{aligned}
V^\pi(s)
&= \mathbb{E}\left[ G_t \mid s_t = s \right] \\
&= \mathbb{E}\left[ r_t + \gamma G_{t+1} \mid s_t = s \right] \\
&= \mathbb{E}\left[ r_t \mid s_t = s \right] + \gamma \mathbb{E}\left[ G_{t+1} \mid s_t = s \right] \\
&= \mathbb{E}\left[ r_t \mid s_t = s \right] + \gamma \mathbb{E}_{s'}\left[ \mathbb{E}\left[ G_{t+1} \mid s_{t+1} = s' \right] \right] \\
&= \mathbb{E}\left[ r_t + \gamma V^\pi(s') \right]
\end{aligned}
$$

The fourth line is the tower property of conditional expectation combined with
the Markov assumption (Eq 1.1): once we condition on $s_{t+1}$, the earlier
history is irrelevant to the future return. Every use of the Bellman equation
is, underneath, a use of the Markov property.

### The optimality equations

For completeness, the optimal value functions satisfy

$$
V^*(s) = \max_a \quad \mathbb{E}_{s'}\left[ r + \gamma V^*(s') \right], \qquad
Q^*(s, a) = \mathbb{E}_{s'}\left[ r + \gamma \max_{a'} Q^*(s', a') \right]
\tag{2.8}
$$

These are the basis of Q-learning and of DQN. Note the $\max_{a'}$: in a
continuous 12-dimensional action space that maximisation is itself a nontrivial
optimisation problem, solved at every single update. This is why value-based
methods need extra machinery in continuous control (DDPG/TD3 train a separate
actor network purely to approximate the argmax; SAC replaces the hard max with
a soft one). Policy-gradient methods sidestep it entirely — see chapter 7.

## 2.5 Why we estimate instead of solving

For a finite MDP with known $P$ and $R$, equation 2.6 is a linear system in
$|\mathcal{S}|$ unknowns and you can just solve it.

For this robot:

- $\mathcal{S}$ is continuous and roughly 50-dimensional. There is no table.
- $P$ is MuJoCo. We can *sample* from it but we have no closed form and no
  ability to enumerate successors.
- Even sampling is expensive: one control step costs ten physics steps of a
  contact-rich rigid-body solve.

So we do the only thing available: **approximate $V^\pi$ with a neural network
and fit it to sampled returns**. In this repository that network is the critic,
a 512-256-128 MLP defined in `ppo_from_scratch/ppo.py` (`ActorCritic.critic`),
trained by regression on GAE-derived targets (chapter 4).

Two consequences follow, and both show up in practice:

1. **The critic is always wrong.** It is a function approximator fitted to noisy
   targets, chasing a moving target as $\pi$ changes. `explained_variance` in
   the training logs measures how wrong; see chapter 13.
2. **The critic's errors propagate into the policy gradient**, because the
   gradient is weighted by advantages estimated *from* the critic. A bad critic
   does not merely slow learning, it points the policy in wrong directions.
   This is the main argument for GAE's $\lambda$ parameter, which lets you trade
   off reliance on the critic against variance.

## 2.6 Asymmetric actor–critic, and why it is not used here

Worth knowing about, because it is the standard next step and the v1 README
recommends it.

Nothing requires the actor and the critic to see the same input. The actor's
input is constrained by deployment: on a real robot it can only use what the
sensors provide. The critic exists only during training and is thrown away
afterwards, so it can be given **privileged information** — ground friction,
the exact payload mass, the terrain height map, the applied push force.

This helps because it changes the critic's problem from partially observed to
fully observed. A critic that cannot see the friction coefficient has to
average over it, and that averaging appears as irreducible variance in its
predictions, which becomes noise in every advantage estimate.

It is not implemented here for a specific reason: sim-to-real is out of scope
for this version, so the actor is *already* allowed privileged information —
base linear velocity $v_b$ is in the observation, and a real Go2 cannot measure
it directly. With no asymmetry between what the actor may see and what the
critic may see, there is nothing for an asymmetric architecture to exploit.
The moment you drop $v_b$ from the actor's observation for sim-to-real
(chapter 16), asymmetric actor–critic becomes the first thing to add back.

## 2.7 Summary

| Object | Definition | What it is for |
|--------|------------|----------------|
| $V^\pi(s)$ | expected return from $s$ under $\pi$ | baseline; reduces gradient variance |
| $Q^\pi(s,a)$ | expected return from $s$ after $a$, then $\pi$ | evaluating a specific action |
| $A^\pi(s,a)$ | $Q^\pi - V^\pi$ | the actual learning signal |
| Bellman eqs | value now = reward + discounted value next | lets us bootstrap instead of rolling out forever |

The next chapter derives the gradient of $J(\theta)$ and shows that the
advantage appears in it naturally, not as a heuristic.

---

**Previous:** [1. What reinforcement learning actually is](01-what-is-rl.md) ·
**Next:** [3. Policy gradients from first principles](03-policy-gradients.md)
