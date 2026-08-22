"""PPO, written out in full.

This file exists to be read. It trains the same Go2Env that scripts/train.py
trains with Stable-Baselines3, and it should reach a comparable learning curve.
If it does, you know that nothing in SB3 is magic - it is this, plus
engineering.

Every block is cross-referenced to a numbered equation in
docs/05-trpo-to-ppo.md and docs/04-actor-critic-and-gae.md. The references look
like (Eq 5.4). Read the chapter with this file open beside it.

Deliberately included, because they are where real implementations differ from
the paper and where most reimplementations quietly go wrong:

  * observation normalisation with running statistics, frozen at evaluation
  * value-target bootstrapping across *truncation* as distinct from termination
  * advantage normalisation per minibatch
  * value-function clipping
  * approximate-KL early stopping
  * a state-independent, learnable log-std for the Gaussian policy

Deliberately excluded, to keep the file readable: recurrent policies, multiple
optimisers, learning-rate schedules beyond linear decay, and any distributed
machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# =========================================================================== #
#  Observation normalisation                                                  #
# =========================================================================== #


class RunningMeanStd:
    """Welford-style running mean and variance, updated batch at a time.

    Why this is not optional: the raw observation mixes joint angles (~1 rad),
    joint velocities (~20 rad/s) and a gait clock (in [-1, 1]). A single dense
    layer applies one weight matrix to all of them, so without normalisation the
    velocity block dominates the first layer's activations and the network
    spends its early capacity undoing the scale difference.

    This is the same object SB3 hides inside VecNormalize - and the same object
    whose loss on resume is bug B12 in this repository's history.
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total

        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)


# =========================================================================== #
#  Networks                                                                   #
# =========================================================================== #


def mlp(sizes, activation=nn.ELU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


def orthogonal_init(module, gain=np.sqrt(2)):
    """Orthogonal weight initialisation.

    Standard in every serious PPO implementation and worth understanding rather
    than copying: an orthogonal matrix preserves the norm of its input, so
    activations neither explode nor vanish as they pass through a deep stack at
    initialisation. The final policy layer gets a much smaller gain (0.01) so
    the initial policy is nearly deterministic at the mean - a policy that
    starts by flailing at full scale destroys the robot before it collects a
    single informative transition.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            nn.init.constant_(m.bias, 0.0)
    return module


class ActorCritic(nn.Module):
    """Gaussian policy and a separate value function.

    The two trunks are separate, not shared. Sharing saves parameters but forces
    one representation to serve two objectives: the critic must model the return
    of a stochastic, externally-pushed robot (high variance, needs to encode
    "how likely am I to fall"), while the actor wants a smooth control law. On
    this task the separate-trunk version is both simpler to reason about and
    empirically better.

    The log-std is a free parameter vector, not a function of the state. A
    state-dependent std is more expressive, but it lets the policy reduce its
    own entropy in exactly the states where exploration matters, and it makes
    the entropy bonus much harder to tune.
    """

    def __init__(self, obs_dim, act_dim, hidden=(512, 256, 128), log_std_init=-1.0):
        super().__init__()
        self.actor = orthogonal_init(mlp([obs_dim, *hidden, act_dim]))
        # Small gain on the output layer: start near the nominal pose.
        nn.init.orthogonal_(self.actor[-2].weight, 0.01)
        self.critic = orthogonal_init(mlp([obs_dim, *hidden, 1]))
        nn.init.orthogonal_(self.critic[-2].weight, 1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def distribution(self, obs):
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs, deterministic=False):
        """Sample an action, and return its log-probability and the value.

        The log-probability is summed over action dimensions because the policy
        is a diagonal Gaussian: the joint density of independent components is
        the product of their densities, so its log is the sum of their logs.
        """
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)

    def evaluate(self, obs, actions):
        """Re-score stored actions under the *current* policy.

        This is the heart of an off-by-one-rollout algorithm: the actions were
        drawn from pi_old, and PPO needs log pi_theta(a|s) for the same actions
        to form the importance ratio (Eq 5.3).
        """
        dist = self.distribution(obs)
        return (
            dist.log_prob(actions).sum(-1),
            dist.entropy().sum(-1),
            self.value(obs),
        )


# =========================================================================== #
#  Rollout storage and GAE                                                    #
# =========================================================================== #


@dataclass
class RolloutBuffer:
    """Fixed-size storage for one on-policy rollout, plus the GAE computation.

    Shapes are (n_steps, n_envs, ...). PPO is on-policy: everything in here is
    thrown away after the update, which is exactly why it needs so many more
    environment steps than an off-policy method - and exactly why it tolerates
    a badly-shaped reward far better.
    """

    n_steps: int
    n_envs: int
    obs_dim: int
    act_dim: int
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        s, e = self.n_steps, self.n_envs
        self.obs = np.zeros((s, e, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((s, e, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros((s, e), dtype=np.float32)
        self.rewards = np.zeros((s, e), dtype=np.float32)
        self.values = np.zeros((s, e), dtype=np.float32)
        # `dones` marks a real episode END (the return genuinely stops).
        # `truncated_values` carries V(s') for steps that ended only because the
        # time limit expired - see `compute_returns`.
        self.dones = np.zeros((s, e), dtype=np.float32)
        self.truncated_values = np.zeros((s, e), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done, truncated_value):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.truncated_values[i] = truncated_value
        self.ptr += 1

    def compute_returns(self, last_value, last_done, gamma, gae_lambda):
        """Generalised Advantage Estimation (Eq 4.7).

            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)     (4.5)
            A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}    (4.7)

        Two subtleties that separate a working implementation from a plausible
        one:

        1. **Truncation is not termination.** When an episode is cut off by the
           time limit the return did not actually stop - the robot was still
           walking. Treating it as terminal teaches the value function that
           every state 20 seconds in is worthless, which biases the critic
           everywhere. So for a truncated step we add gamma * V(s_final) back
           into the reward and then treat the step as a boundary for the
           *advantage recursion* only. SB3 does this via the
           `TimeLimit.truncated` info key; here it is explicit.

        2. **The recursion runs backwards.** A_t depends on A_{t+1}, so the loop
           goes from the end of the rollout to the start, seeded by the value of
           the state the rollout stopped at.
        """
        advantages = np.zeros_like(self.rewards)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)

        rewards = self.rewards.copy()
        # Add the bootstrap for time-limit truncations (subtlety 1).
        rewards += gamma * self.truncated_values

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        # The value target is the advantage plus the current estimate. This is
        # the lambda-return; regressing the critic on it rather than on the raw
        # Monte-Carlo return is what keeps the critic's variance manageable.
        returns = advantages + self.values
        return advantages, returns

    def flat(self, advantages, returns):
        """Flatten (n_steps, n_envs) into one batch of transitions."""
        n = self.n_steps * self.n_envs
        to = lambda x, shape: torch.as_tensor(  # noqa: E731
            x.reshape(shape), device=self.device
        )
        return dict(
            obs=to(self.obs, (n, self.obs_dim)),
            actions=to(self.actions, (n, self.act_dim)),
            log_probs=to(self.log_probs, (n,)),
            advantages=to(advantages, (n,)),
            returns=to(returns, (n,)),
            values=to(self.values, (n,)),
        )

    def reset(self):
        self.ptr = 0


# =========================================================================== #
#  The algorithm                                                              #
# =========================================================================== #


@dataclass
class PPOConfig:
    n_steps: int = 512
    batch_size: int = 2048
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = 0.2
    ent_coef: float = 0.002
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 3.0e-4
    anneal_lr: bool = True
    target_kl: float | None = 0.02
    normalize_advantage: bool = True
    hidden: tuple = (512, 256, 128)
    log_std_init: float = -1.0
    device: str = "auto"


class PPO:
    """Proximal Policy Optimization.

    The whole algorithm is: collect a rollout under the current policy, estimate
    advantages with GAE, then take a handful of gradient steps on a surrogate
    objective that is deliberately pessimistic about moving far from the policy
    that collected the data. Everything else is bookkeeping.
    """

    def __init__(self, env, config: PPOConfig | None = None, seed: int = 0):
        self.cfg = config or PPOConfig()
        self.env = env
        self.n_envs = env.num_envs

        device = self.cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.obs_dim, self.act_dim = obs_dim, act_dim

        self.policy = ActorCritic(
            obs_dim, act_dim, self.cfg.hidden, self.cfg.log_std_init
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.cfg.learning_rate, eps=1e-5
        )
        self.obs_rms = RunningMeanStd(obs_dim)

        self.buffer = RolloutBuffer(
            self.cfg.n_steps, self.n_envs, obs_dim, act_dim, self.device
        )
        self.num_timesteps = 0
        self._last_obs = None
        self._last_done = np.zeros(self.n_envs, dtype=np.float32)
        self._ep_returns = np.zeros(self.n_envs)
        self._ep_lengths = np.zeros(self.n_envs, dtype=int)
        self._completed = []

    # ---------------------------------------------------------------- #
    #  Rollout collection                                              #
    # ---------------------------------------------------------------- #

    def _normalize(self, obs, update=True):
        if update:
            self.obs_rms.update(obs.astype(np.float64))
        return self.obs_rms.normalize(obs).astype(np.float32)

    def collect_rollout(self):
        """Run the current policy for n_steps in every environment."""
        self.buffer.reset()
        if self._last_obs is None:
            raw = self.env.reset()
            self._last_obs = self._normalize(raw)

        for _ in range(self.cfg.n_steps):
            obs_t = torch.as_tensor(self._last_obs, device=self.device)
            with torch.no_grad():
                action, log_prob, value = self.policy.act(obs_t)
            action_np = action.cpu().numpy()

            # The environment's action space is [-1, 1]; the Gaussian is not
            # bounded, so we clip for the environment but store the UNCLIPPED
            # action. Storing the clipped one would make the stored log-prob
            # inconsistent with the action, quietly corrupting the ratio.
            clipped = np.clip(action_np, -1.0, 1.0)
            next_obs, rewards, dones, infos = self.env.step(clipped)

            # Time-limit bootstrapping (see RolloutBuffer.compute_returns).
            truncated_values = np.zeros(self.n_envs, dtype=np.float32)
            for i, info in enumerate(infos):
                if dones[i] and info.get("TimeLimit.truncated", False):
                    terminal_obs = self._normalize(
                        np.asarray(info["terminal_observation"],
                                   dtype=np.float32)[None], update=False
                    )
                    with torch.no_grad():
                        tv = self.policy.value(
                            torch.as_tensor(terminal_obs, device=self.device)
                        )
                    truncated_values[i] = float(tv.item())

            self.buffer.add(
                self._last_obs,
                action_np,
                log_prob.cpu().numpy(),
                rewards,
                value.cpu().numpy(),
                self._last_done,
                truncated_values,
            )

            self._ep_returns += rewards
            self._ep_lengths += 1
            for i, d in enumerate(dones):
                if d:
                    self._completed.append(
                        (float(self._ep_returns[i]), int(self._ep_lengths[i]))
                    )
                    self._ep_returns[i] = 0.0
                    self._ep_lengths[i] = 0

            self._last_obs = self._normalize(next_obs)
            self._last_done = dones.astype(np.float32)
            self.num_timesteps += self.n_envs

        with torch.no_grad():
            last_value = self.policy.value(
                torch.as_tensor(self._last_obs, device=self.device)
            ).cpu().numpy()
        return self.buffer.compute_returns(
            last_value, self._last_done, self.cfg.gamma, self.cfg.gae_lambda
        )

    # ---------------------------------------------------------------- #
    #  The update                                                      #
    # ---------------------------------------------------------------- #

    def update(self, advantages, returns):
        """Several epochs of minibatch SGD on the clipped surrogate objective."""
        data = self.buffer.flat(advantages, returns)
        n = data["obs"].shape[0]
        indices = np.arange(n)

        stats = dict(policy_loss=[], value_loss=[], entropy=[],
                     approx_kl=[], clip_fraction=[])
        early_stopped = False

        for epoch in range(self.cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.cfg.batch_size):
                idx = indices[start:start + self.cfg.batch_size]
                mb = {k: v[idx] for k, v in data.items()}

                log_prob, entropy, value = self.policy.evaluate(
                    mb["obs"], mb["actions"]
                )

                # --- the importance ratio (Eq 5.3) ------------------------- #
                # r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s)
                # computed in log space, because the ratio of two products of
                # twelve Gaussian densities underflows in float32 otherwise.
                ratio = torch.exp(log_prob - mb["log_probs"])

                adv = mb["advantages"]
                if self.cfg.normalize_advantage:
                    # Per-minibatch standardisation. This makes the effective
                    # step size independent of the reward scale, which is why
                    # you can change reward weights without retuning the
                    # learning rate. It also introduces a small bias - the
                    # normalised advantage is no longer an unbiased estimate of
                    # the true advantage - which in practice is worth it.
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # --- the clipped surrogate (Eq 5.4) ------------------------ #
                # L = E[ min( r * A,  clip(r, 1-eps, 1+eps) * A ) ]
                # The min makes the objective a PESSIMISTIC bound: when the
                # ratio moves in a direction that would improve the surrogate
                # too much, the clipped branch wins and the gradient vanishes.
                # There is no penalty for moving too far - there is simply no
                # longer any incentive to.
                surrogate_1 = ratio * adv
                surrogate_2 = torch.clamp(
                    ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range
                ) * adv
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # --- the value loss (Eq 5.6) ------------------------------- #
                if self.cfg.clip_range_vf is None:
                    value_loss = ((value - mb["returns"]) ** 2).mean()
                else:
                    # Clipping the value update too. Same motivation as the
                    # policy clip: the critic is being fit to targets computed
                    # from its own old predictions, so letting it move far in
                    # one update makes the next rollout's advantages stale.
                    value_clipped = mb["values"] + torch.clamp(
                        value - mb["values"],
                        -self.cfg.clip_range_vf,
                        self.cfg.clip_range_vf,
                    )
                    value_loss = torch.max(
                        (value - mb["returns"]) ** 2,
                        (value_clipped - mb["returns"]) ** 2,
                    ).mean()

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping by global norm. PPO's trust region is on the
                # policy *distribution*; nothing in the objective bounds the
                # gradient magnitude, and one bad minibatch can undo a rollout.
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    # Schulman's low-variance approximate KL:
                    #     KL ~= E[ (r - 1) - log r ]
                    # It is non-negative by construction, unlike the naive
                    # E[-log r], which is why it is the one worth logging.
                    log_ratio = log_prob - mb["log_probs"]
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = (
                        (ratio - 1).abs() > self.cfg.clip_range
                    ).float().mean().item()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.mean().item())
                stats["approx_kl"].append(approx_kl)
                stats["clip_fraction"].append(clip_frac)

                if self.cfg.target_kl is not None and approx_kl > 1.5 * self.cfg.target_kl:
                    # Stop this update early. The data was collected under
                    # pi_old; once the policy has moved this far the importance
                    # weights are too large for the remaining epochs to be
                    # trustworthy, clipping or no clipping.
                    early_stopped = True
                    break
            if early_stopped:
                break

        out = {k: float(np.mean(v)) for k, v in stats.items() if v}
        out["early_stopped"] = float(early_stopped)
        out["std"] = float(self.policy.log_std.exp().mean().item())

        # Explained variance of the critic: 1 - Var(returns - values) /
        # Var(returns). Near 1 means the critic explains the return; near 0 (or
        # negative) means it is no better than predicting the mean, and your
        # advantages are mostly noise. It is the single most diagnostic number
        # in a PPO run.
        y_true = data["returns"].cpu().numpy()
        y_pred = data["values"].cpu().numpy()
        var_y = float(np.var(y_true))
        out["explained_variance"] = (
            float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
        )
        return out

    # ---------------------------------------------------------------- #
    #  Training loop                                                   #
    # ---------------------------------------------------------------- #

    def learn(self, total_timesteps, log_every=1, callback=None):
        n_updates = int(total_timesteps // (self.cfg.n_steps * self.n_envs))
        for update in range(1, n_updates + 1):
            if self.cfg.anneal_lr:
                frac = 1.0 - (update - 1) / n_updates
                for group in self.optimizer.param_groups:
                    group["lr"] = frac * self.cfg.learning_rate

            advantages, returns = self.collect_rollout()
            stats = self.update(advantages, returns)

            if self._completed:
                recent = self._completed[-100:]
                stats["ep_rew_mean"] = float(np.mean([r for r, _ in recent]))
                stats["ep_len_mean"] = float(np.mean([l for _, l in recent]))
            stats["timesteps"] = self.num_timesteps
            stats["update"] = update
            stats["lr"] = self.optimizer.param_groups[0]["lr"]

            if callback is not None:
                callback(self, stats)
            if log_every and update % log_every == 0:
                self._print(stats)
        return self

    @staticmethod
    def _print(s):
        print(
            "upd %4d  steps %9d  rew %8.2f  len %6.1f  kl %.4f  clip %.3f  "
            "ev %+.3f  std %.3f"
            % (
                s["update"], s["timesteps"], s.get("ep_rew_mean", float("nan")),
                s.get("ep_len_mean", float("nan")), s["approx_kl"],
                s["clip_fraction"], s["explained_variance"], s["std"],
            )
        )

    # ---------------------------------------------------------------- #
    #  Persistence                                                     #
    # ---------------------------------------------------------------- #

    def save(self, path):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                # The normalisation statistics travel WITH the weights. Saving
                # them separately is how you end up with bug B12.
                "obs_rms": (self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count),
                "config": self.cfg,
                "num_timesteps": self.num_timesteps,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        mean, var, count = ckpt["obs_rms"]
        self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count = mean, var, count
        self.num_timesteps = ckpt["num_timesteps"]
        return self
