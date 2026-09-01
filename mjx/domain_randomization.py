"""Per-environment friction and mass randomisation for the MJX backend.

Two kinds of domain randomisation exist in this project, and they are
implemented in different places for a reason worth understanding.

**State-level** — actuator gains, push timing, observation noise. These vary per
environment simply by living in the environment's state dict, which ``vmap``
already maps over. `mjx/mjx_env.py` handles them.

**Model-level** — ground friction, link masses. These live in ``mjx.Model``,
which is *shared* across the batch: one model, N states. Randomising them per
environment means building a model whose relevant fields carry a leading batch
dimension, and telling ``vmap`` via ``in_axes`` which fields are batched and
which are not. That is what this file does, in the shape brax's PPO expects
from its ``randomization_fn`` argument.

Getting this wrong is quiet rather than loud: forget the ``in_axes`` and every
environment silently shares environment zero's friction, so the randomisation
appears to be on while doing nothing at all.

Ranges mirror `envs/go2_env.py::_randomize_domain` so the two backends pose the
same problem — see `tests/test_mjx.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def domain_randomize(model, rng, friction_range=(0.4, 1.4),
                     added_mass_range=(-1.0, 2.0), base_body_name_id=None):
    """Return ``(batched_model, in_axes)`` for brax's ``randomization_fn``.

    Args:
        model: an ``mjx.Model``.
        rng: a batch of PRNG keys, one per environment. Its leading dimension
            sets the batch size.
        friction_range: multiplicative range on sliding friction.
        added_mass_range: kilograms added to the trunk. A real robot carries a
            battery, sensors, sometimes a payload.
        base_body_name_id: body id of the trunk. Required for mass
            randomisation; if None, only friction is randomised.
    """

    @jax.vmap
    def sample(key):
        k_friction, k_mass = jax.random.split(key)

        scale = jax.random.uniform(
            k_friction, minval=friction_range[0], maxval=friction_range[1])
        # Column 0 is sliding friction; torsional and rolling are left alone,
        # they matter far less here and randomising them adds variance for
        # nothing.
        friction = model.geom_friction.at[:, 0].set(
            model.geom_friction[:, 0] * scale)

        mass = model.body_mass
        if base_body_name_id is not None:
            added = jax.random.uniform(
                k_mass, minval=added_mass_range[0], maxval=added_mass_range[1])
            mass = mass.at[base_body_name_id].add(added)

        return friction, mass

    friction, mass = sample(rng)

    # Every field defaults to unbatched (None); only the two we generated get a
    # leading batch axis. This pairing is the whole point of the function.
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace({"geom_friction": 0, "body_mass": 0})
    model = model.tree_replace({"geom_friction": friction, "body_mass": mass})
    return model, in_axes


def make_randomization_fn(env, **kwargs):
    """Bind the trunk body id, leaving a brax-compatible callable.

    brax supplies the PRNG keys itself:

        v_randomization_fn = functools.partial(randomization_fn, rng=keys)

    so the returned function must take ``sys`` positionally and ``rng`` as a
    keyword. Pre-binding rng here would be silently overridden.
    """
    import functools

    return functools.partial(
        domain_randomize, base_body_name_id=int(env.base_body_id), **kwargs)
