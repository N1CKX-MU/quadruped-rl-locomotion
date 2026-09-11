"""The NumPy re-implementation of a brax policy must BE the brax policy.

envs/brax_policy.py re-derives brax's deterministic inference by hand so the
CPU scripts can run GPU-trained policies without brax or JAX. That is only safe
if something checks the re-derivation against brax itself - otherwise a wrong
activation, a missed normaliser, or loc/scale taken in the wrong order produces
a policy that loads cleanly and walks badly, and the CPU evaluation silently
blames the training.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from envs.brax_policy import BraxPolicy


def _random_policy(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [50, 64, 32, 24]
    policy = BraxPolicy(
        obs_mean=rng.normal(size=50),
        obs_std=rng.uniform(0.5, 2.0, size=50),
        weights=[rng.normal(scale=0.3, size=(a, b)) for a, b in zip(sizes, sizes[1:])],
        biases=[rng.normal(scale=0.1, size=b) for b in sizes[1:]],
        action_size=12,
        meta={"source": "unit-test", "timesteps": 123},
    )
    path = tmp_path / "p.npz"
    policy.save(path)
    return policy, path


def test_save_load_roundtrip_preserves_the_policy(tmp_path):
    policy, path = _random_policy(tmp_path)
    loaded = BraxPolicy.load(path)
    obs = np.random.default_rng(1).normal(size=(16, 50))
    # float32 on disk, so compare at float32 precision.
    assert np.allclose(policy(obs), loaded(obs), atol=1e-5)
    assert loaded.meta == {"source": "unit-test", "timesteps": 123}


def test_actions_are_bounded_and_batch_agrees_with_single(tmp_path):
    policy, _ = _random_policy(tmp_path)
    obs = np.random.default_rng(2).normal(scale=50.0, size=(8, 50))
    batch = policy(obs)
    assert batch.shape == (8, 12)
    assert np.all(np.abs(batch) <= 1.0)
    assert np.allclose(batch[3], policy(obs[3]))


def test_loading_a_policy_does_not_import_jax(tmp_path):
    """The whole reason this module exists: the CPU stack stays brax/JAX-free."""
    _, path = _random_policy(tmp_path)
    code = ("import sys; from envs.brax_policy import BraxPolicy; "
            "BraxPolicy.load(%r)(__import__('numpy').zeros(50)); "
            "bad = [m for m in ('jax', 'brax', 'flax') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)" % str(path))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], cwd=root,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_numpy_policy_matches_brax_inference(tmp_path):
    """Export real brax PPO networks and compare against brax's own policy."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("brax")
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from mjx.export_policy import export
    from mjx.train_mjx import POLICY_HIDDEN, VALUE_HIDDEN

    obs_size, act_size = 50, 12
    nets = ppo_networks.make_ppo_networks(
        obs_size, act_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=POLICY_HIDDEN,
        value_hidden_layer_sizes=VALUE_HIDDEN,
    )
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    policy_params = nets.policy_network.init(k1)
    value_params = nets.value_network.init(k2)
    # Non-trivial normaliser, so a loader that skipped it would fail loudly.
    norm = running_statistics.init_state(jax.ShapeDtypeStruct((obs_size,), jnp.float32))
    norm = norm.replace(
        mean=jax.random.normal(k3, (obs_size,)),
        std=jax.random.uniform(k4, (obs_size,), minval=0.3, maxval=3.0),
    )
    params = (norm, policy_params, value_params)

    infer = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    obs = np.random.default_rng(3).normal(scale=2.0, size=(64, obs_size)).astype(np.float32)
    expected, _ = infer(jnp.asarray(obs), jax.random.PRNGKey(1))

    path = tmp_path / "exported.npz"
    export(params, str(path), act_size)
    got = BraxPolicy.load(path)(obs)

    assert got.shape == (64, act_size)
    assert np.allclose(got, np.asarray(expected), atol=1e-5), (
        np.abs(got - np.asarray(expected)).max())
    # And not trivially: the actions actually vary across observations.
    assert np.asarray(expected).std() > 0.05
