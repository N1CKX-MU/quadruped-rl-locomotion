"""Run a brax-trained (GPU) policy in the CPU environment, with NumPy only.

The MJX path trains with brax's PPO and saves its parameters as a pickle of JAX
arrays and brax dataclasses. Unpickling that needs brax and JAX, which the CPU
stack deliberately does not depend on - and evaluation has to happen on the CPU
environment, because MJX's simplified collision model means its own verdict on a
policy does not count (docs/17).

So the GPU side exports the policy to a plain ``.npz``
(``python -m mjx.export_policy``, and ``mjx/train_mjx.py`` does it automatically
after training), and this module runs it. The forward pass is short enough to
write out, and writing it out is the point: it is exactly what brax's
``make_inference_fn(..., deterministic=True)`` computes, and
``tests/test_brax_policy.py`` checks that against brax itself.

    x      = (obs - mean) / std                   running_statistics.normalize
    h      = silu(x W0 + b0) ... silu(h W2 + b2)  brax.training.networks.MLP
    logits = h W3 + b3                            (2 * action_size,)
    loc    = logits[:action_size]                 NormalTanhDistribution
    action = tanh(loc)                            .mode() -> TanhBijector

The other half of ``logits`` parameterises the exploration noise (its softplus
is the standard deviation). A deterministic evaluation ignores it, exactly as
brax does.
"""

from __future__ import annotations

import numpy as np

FORMAT_VERSION = 1


def _silu(x):
    # x * sigmoid(x), written to avoid overflow in exp for large negative x.
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


_ACTIVATIONS = {"silu": _silu, "swish": _silu, "relu": lambda x: np.maximum(x, 0.0),
                "tanh": np.tanh}


class BraxPolicy:
    """Deterministic brax PPO policy. ``policy(obs) -> action`` in [-1, 1]."""

    def __init__(self, obs_mean, obs_std, weights, biases, action_size,
                 activation="silu", meta=None):
        self.obs_mean = np.asarray(obs_mean, dtype=np.float64)
        self.obs_std = np.asarray(obs_std, dtype=np.float64)
        self.weights = [np.asarray(w, dtype=np.float64) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float64) for b in biases]
        self.action_size = int(action_size)
        if activation not in _ACTIVATIONS:
            raise ValueError("unsupported activation %r" % activation)
        self.activation = activation
        self._act = _ACTIVATIONS[activation]
        self.meta = dict(meta or {})

        if self.weights[0].shape[0] != self.obs_mean.shape[0]:
            raise ValueError("first layer expects %d inputs, normaliser has %d"
                             % (self.weights[0].shape[0], self.obs_mean.shape[0]))
        if self.weights[-1].shape[1] != 2 * self.action_size:
            raise ValueError("last layer has %d outputs, expected 2 x %d "
                             "(location and scale of a tanh-normal)"
                             % (self.weights[-1].shape[1], self.action_size))

    @property
    def obs_size(self):
        return self.obs_mean.shape[0]

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            version = int(z["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError("%s: policy format %d, this loader reads %d"
                                 % (path, version, FORMAT_VERSION))
            n = int(z["num_layers"])
            meta = {k[len("meta_"):]: z[k].item() for k in z.files
                    if k.startswith("meta_")}
            return cls(
                obs_mean=z["obs_mean"],
                obs_std=z["obs_std"],
                weights=[z["w%d" % i] for i in range(n)],
                biases=[z["b%d" % i] for i in range(n)],
                action_size=int(z["action_size"]),
                activation=str(z["activation"]),
                meta=meta,
            )

    def save(self, path):
        arrays = dict(
            format_version=np.array(FORMAT_VERSION),
            num_layers=np.array(len(self.weights)),
            action_size=np.array(self.action_size),
            activation=np.array(self.activation),
            obs_mean=self.obs_mean.astype(np.float32),
            obs_std=self.obs_std.astype(np.float32),
        )
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays["w%d" % i] = w.astype(np.float32)
            arrays["b%d" % i] = b.astype(np.float32)
        for k, v in self.meta.items():
            arrays["meta_" + k] = np.array(v)
        np.savez(path, **arrays)

    def logits(self, obs):
        h = (np.asarray(obs, dtype=np.float64) - self.obs_mean) / self.obs_std
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i != last:
                h = self._act(h)
        return h

    def __call__(self, obs):
        """Accepts one observation (obs_size,) or a batch (n, obs_size)."""
        loc = self.logits(obs)[..., :self.action_size]
        return np.tanh(loc).astype(np.float32)
