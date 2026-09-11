"""Export a brax PPO checkpoint to the NumPy format the CPU scripts read.

    python -m mjx.export_policy models/go2_mjx_params.pkl
    # -> models/go2_mjx_params.npz

    python scripts/evaluate.py --model models/go2_mjx_params.npz --grid
    python scripts/play.py     --model models/go2_mjx_params.npz

Needs brax (to unpickle the checkpoint), so run it where you trained.
``mjx/train_mjx.py`` calls ``export`` itself after saving, so this is only
needed for checkpoints written before that existed. The forward pass the result
is evaluated with is documented in ``envs/brax_policy.py``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.brax_policy import BraxPolicy  # noqa: E402

# Must match the network_factory in mjx/train_mjx.py. brax's make_ppo_networks
# defaults to silu, and train_mjx does not override it.
ACTIVATION = "silu"


def policy_from_params(params, action_size, meta=None):
    """Build a BraxPolicy from brax's (normalizer, policy, value) params tuple."""
    normalizer, policy = params[0], params[1]
    layers = policy["params"]

    def index(name):
        m = re.fullmatch(r"hidden_(\d+)", name)
        if m is None:
            raise ValueError("unexpected layer %r in policy params" % name)
        return int(m.group(1))

    names = sorted(layers, key=index)
    return BraxPolicy(
        obs_mean=np.asarray(normalizer.mean),
        obs_std=np.asarray(normalizer.std),
        weights=[np.asarray(layers[n]["kernel"]) for n in names],
        biases=[np.asarray(layers[n]["bias"]) for n in names],
        action_size=action_size,
        activation=ACTIVATION,
        meta=meta,
    )


def export(params, out_path, action_size, meta=None):
    policy = policy_from_params(params, action_size, meta)
    policy.save(out_path)
    return policy


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoint", help="brax params pickle from mjx/train_mjx.py")
    p.add_argument("--out", default=None, help="default: same name, .npz")
    p.add_argument("--action-size", type=int, default=12)
    args = p.parse_args()

    with open(args.checkpoint, "rb") as f:
        params = pickle.load(f)
    out = args.out or os.path.splitext(args.checkpoint)[0] + ".npz"
    policy = export(params, out, args.action_size,
                    meta={"source": os.path.basename(args.checkpoint)})
    print("exported %s -> %s  (%d -> %s -> %d, %s)" % (
        args.checkpoint, out, policy.obs_size,
        " -> ".join(str(w.shape[1]) for w in policy.weights[:-1]),
        policy.action_size, policy.activation))


if __name__ == "__main__":
    main()
