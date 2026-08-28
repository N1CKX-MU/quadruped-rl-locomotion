# Quadruped RL, from scratch

This is the long-form companion to the code in this repository. It is written
to be read in order, and it assumes nothing beyond first-year calculus, linear
algebra, and enough Python to read a `for` loop.

The goal is not to summarise reinforcement learning. It is to derive the
specific algorithm this repository uses, from the definition of a Markov
decision process to the exact line of PyTorch that implements the clipped
surrogate objective, and then to explain every engineering decision made in
turning that algorithm into a robot that walks.

Where the maths and the code correspond, the correspondence is made explicit:
equations are numbered, and `ppo_from_scratch/ppo.py` carries comments like
`(Eq 5.4)` pointing back here.

**Looking for the current state rather than the explanation?**
[`WORKLOG.md`](WORKLOG.md) is the running record: what works today, what is
measured, the decision log, open items, and the traps specific to this machine.
Start there if you are returning to the project; start here if you want to
understand it.

## How to read this

If you want to **understand the algorithm**, read 1 through 6 in order. That is
the mathematical spine: MDP, value functions, policy gradients, GAE, PPO, and
then the implementation line by line.

If you want to **understand the robot**, read 8 through 11. Those chapters are
about the specific problem — a Go2 in MuJoCo — and are largely independent of
which RL algorithm you use.

If you want to **understand what went wrong and how it was found**, read 14. It
is the debugging log for the v1 → v2 rewrite, and it is the most practically
useful chapter in the book. It is also the honest one: every bug listed there
was really in the code, and several of them were subtle enough to survive five
documented training runs.

If you are **about to start a training run**, read 13 first. It tells you what
the TensorBoard traces mean and which failure each one indicates, which is the
difference between debugging in an afternoon and debugging in a week.

## Chapters

### Part I — The algorithm

| # | Chapter | What it covers |
|---|---------|----------------|
| 1 | [What reinforcement learning actually is](01-what-is-rl.md) | MDPs, trajectories, return, discounting; this robot's MDP written out in full |
| 2 | [Value functions and policies](02-value-functions-and-policies.md) | $V^\pi$, $Q^\pi$, the advantage, the Bellman equations, and why we estimate rather than solve them |
| 3 | [Policy gradients from first principles](03-policy-gradients.md) | The log-derivative trick, the policy gradient theorem derived in full, baselines and variance |
| 4 | [Actor-critic and GAE](04-actor-critic-and-gae.md) | TD errors, $n$-step returns, the $\lambda$-return, the GAE derivation, and the bias–variance dial |
| 5 | [From TRPO to PPO](05-trpo-to-ppo.md) | Importance sampling, the surrogate objective, trust regions, and why clipping approximates a KL constraint |
| 6 | [PPO line by line](06-ppo-line-by-line.md) | Every block of `ppo_from_scratch/ppo.py` mapped onto the equations in chapters 4 and 5 |
| 7 | [Why PPO and not SAC or TD3](07-why-not-sac-td3.md) | On-policy vs off-policy, and an honest re-reading of this repo's own algorithm comparison |

### Part II — The robot

| # | Chapter | What it covers |
|---|---------|----------------|
| 8 | [The Go2, MuJoCo, and PD control](08-the-robot.md) | Kinematics, the MJCF model, torque actuators, the PD law, and why the `home` keyframe matters |
| 9 | [Observations and actions](09-observations-and-actions.md) | Every one of the 50 observation dimensions justified; why projected gravity, why the body frame, why a clock |
| 10 | [Reward engineering](10-reward-engineering.md) | All sixteen terms with their equations, exponential kernels vs quadratic costs, and a worked reward-hacking exploit |
| 11 | [Gaits, phase, and periodic control](11-gaits-and-phase.md) | Gait taxonomy, duty factor, phase offsets, and the maths that makes gait a commandable input |
| 12 | [Curriculum and domain randomisation](12-curriculum-and-domain-randomization.md) | Why a fixed schedule fails, what a closed-loop curriculum measures, what to randomise and how much |

### Part III — Practice

| # | Chapter | What it covers |
|---|---------|----------------|
| 13 | [Reading a training run](13-training-diagnostics.md) | Explained variance, approximate KL, clip fraction, entropy; what each failure mode looks like |
| 14 | [The debugging log](14-debugging-log.md) | Sixteen real bugs in v1: the reasoning that found each one, the evidence, and the fix |
| 15 | [Results](15-results.md) | v1 vs v2, measured on the same axes |
| 16 | [What sim-to-real would take](16-sim-to-real.md) | The gaps that remain, honestly enumerated. Not implemented — explained |
| 17 | [Scaling with MJX](17-mjx-and-scaling.md) | Why 16 CPU environments is the wrong shape for this problem, and what to do about it |

## Notation

Used consistently throughout.

| Symbol | Meaning |
|--------|---------|
| $s_t$ | state at time $t$ |
| $a_t$ | action at time $t$ |
| $r_t$ | reward received on the transition out of $s_t$ |
| $\pi_\theta(a \mid s)$ | policy: a distribution over actions, parameterised by $\theta$ |
| $\gamma$ | discount factor, in $[0, 1)$ |
| $\lambda$ | GAE trace-decay parameter, in $[0, 1]$ |
| $G_t$ | return from time $t$ onwards |
| $V^\pi(s)$ | state-value function under $\pi$ |
| $Q^\pi(s, a)$ | action-value function under $\pi$ |
| $A^\pi(s, a)$ | advantage, $Q^\pi - V^\pi$ |
| $\delta_t$ | TD error at time $t$ |
| $\hat{A}_t$ | estimated advantage (GAE) |
| $\tau$ | a trajectory $(s_0, a_0, r_0, s_1, \dots)$ |
| $\rho_t(\theta)$ | importance ratio $\pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ |

Vectors are column vectors. $\|x\|$ is the Euclidean norm. $\mathbb{E}_{\tau \sim \pi}[\cdot]$
means the expectation over trajectories generated by following $\pi$.

## A note on the equations

Markdown renderers vary in their LaTeX support. On GitHub, `$...$` and
`$$...$$` render natively. In a plain text editor they will show as source,
which is still readable — the notation is standard.
