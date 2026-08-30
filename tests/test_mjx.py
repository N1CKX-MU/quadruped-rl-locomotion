"""MJX backend tests.

The claim these defend is the one that makes the MJX path worth having at all:
that the GPU environment optimises the *same objective* as the CPU one. Two
independent implementations of a sixteen-term reward will drift apart, and the
drift surfaces months later as an unexplained performance gap between the two
backends. Sharing `envs/rewards.py` and `envs/gait.py` across both is only
meaningful if something checks that the sharing actually works.

Skipped entirely when jax is not installed - it is an optional dependency.
"""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from envs import rewards as R  # noqa: E402

XML = "mujoco_menagerie/unitree_go2/scene_mjx.xml"


def make_state(xp):
    """The same hand-constructed state, built with numpy or with jax.numpy."""
    n = 12
    return R.RewardState(
        lin_vel_b=xp.array([0.5, 0.1, -0.05]),
        ang_vel_b=xp.array([0.02, -0.03, 0.4]),
        proj_gravity=xp.array([0.05, -0.02, -0.998]),
        base_height=xp.array(0.29),
        joint_pos=xp.linspace(-0.4, 0.4, n),
        joint_vel=xp.linspace(-2.0, 2.0, n),
        joint_vel_prev=xp.linspace(-1.5, 1.5, n),
        default_joint_pos=xp.zeros(n),
        soft_joint_limits=xp.stack([xp.full((n,), -1.0), xp.full((n,), 1.0)], axis=1),
        torques=xp.linspace(-5.0, 5.0, n),
        action=xp.linspace(-0.5, 0.5, n),
        prev_action=xp.linspace(-0.4, 0.4, n),
        contact=xp.array([1.0, 0.0, 0.0, 1.0]),
        desired_contact=xp.array([1.0, 0.0, 0.0, 1.0]),
        feet_air_time=xp.array([0.0, 0.3, 0.3, 0.0]),
        feet_first_contact=xp.array([1.0, 0.0, 0.0, 0.0]),
        feet_vel_xy=xp.full((4, 2), 0.1),
        feet_height=xp.array([0.01, 0.06, 0.06, 0.01]),
        cmd=xp.array([0.6, 0.0, 0.4]),
        cmd_base_height=xp.array(0.30),
        undesired_contacts=xp.array(1.0),
        dt=0.02,
    )


def test_reward_agrees_between_numpy_and_jax():
    """The headline test: the same reward, to floating-point tolerance."""
    weights = R.resolve_weights(None)
    total_np, terms_np = R.compute(make_state(np), weights)
    total_jx, terms_jx = R.compute(make_state(jnp), weights)

    assert np.allclose(float(total_np), float(total_jx), atol=1e-5)
    assert set(terms_np) == set(terms_jx)
    for name in terms_np:
        assert np.allclose(float(terms_np[name]), float(terms_jx[name]),
                           atol=1e-5), name


def test_reward_is_jittable_and_vmappable():
    """If either fails, the MJX environment cannot use the shared reward."""
    weights = R.resolve_weights(None)

    fn = jax.jit(lambda: R.compute(make_state(jnp), weights)[0])
    assert np.isfinite(float(fn()))

    batched = jax.vmap(lambda _: R.compute(make_state(jnp), weights)[0])
    assert batched(jnp.arange(4)).shape == (4,)


def test_gpu_path_does_not_import_the_cpu_stack():
    """The MJX backend must not drag gymnasium or mujoco.viewer in with it.

    envs/__init__.py used to import Go2Env eagerly, so `from envs import gait`
    executed it and pulled gymnasium along. That made the GPU training path die
    with ModuleNotFoundError in a WSL venv holding JAX and MJX and having no
    reason to want a Gymnasium wrapper or an OpenGL viewer. envs/__init__.py is
    lazy now (PEP 562) and resolve_asset_path lives in the dependency-free
    envs/paths.py rather than in go2_env.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %r)
        import envs, envs.gait, envs.rewards, envs.commands, envs.paths
        import mjx.mjx_env
        leaked = [m for m in ("gymnasium", "mujoco.viewer", "stable_baselines3")
                  if m in sys.modules]
        print("LEAKED:" + ",".join(leaked))
        """
    ) % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, timeout=300)
    assert "LEAKED:" in out.stdout, out.stderr[-1500:]
    leaked = out.stdout.split("LEAKED:")[1].strip()
    assert leaked == "", "GPU path pulled in the CPU stack: %s" % leaked


def test_gait_schedule_agrees_between_backends():
    from envs.gait import desired_contact, gait_params

    offsets, duty = gait_params("trot")
    for phase in np.linspace(0.0, 1.0, 21, endpoint=False):
        a = desired_contact(phase, np.asarray(offsets), duty)
        b = desired_contact(jnp.asarray(phase), jnp.asarray(offsets), duty)
        assert np.allclose(np.asarray(a), np.asarray(b)), phase


@pytest.mark.skipif(not os.path.exists(XML), reason="mujoco_menagerie not present")
def test_mjx_and_cpu_environments_agree_on_every_shared_default():
    """The two backends must optimise the same problem, not just the same reward.

    This caught real drift: after the B20 fix raised the CPU action scale to
    0.40 and added the command feasibility clamp, the MJX environment was still
    on 0.30 with no clamp at all - so the GPU path would have reintroduced B20
    while every reward test still passed. Sharing envs/rewards.py guarantees the
    same objective; it guarantees nothing about the dynamics or the command
    distribution, which is what this test is for.
    """
    import inspect

    from envs.go2_env import Go2Env
    from mjx.mjx_env import Go2MJXEnv

    cpu = inspect.signature(Go2Env).parameters
    gpu = inspect.signature(Go2MJXEnv).parameters
    shared = sorted(set(cpu) & set(gpu) - {"xml_path", "render_mode"})
    assert len(shared) >= 10, shared        # guard against comparing nothing

    mismatched = {
        name: (cpu[name].default, gpu[name].default)
        for name in shared
        if cpu[name].default != gpu[name].default
    }
    assert not mismatched, "MJX and CPU defaults have drifted: %s" % mismatched


@pytest.mark.skipif(not os.path.exists(XML), reason="mujoco_menagerie not present")
def test_mjx_command_sampling_respects_the_feasibility_limit():
    """Bug B20 on the GPU side: no command may exceed stride x frequency."""
    from mjx.mjx_env import Go2MJXEnv

    env = Go2MJXEnv()
    sample = jax.jit(jax.vmap(env.sample_command))
    cmd, freq, _ = sample(jax.random.split(jax.random.PRNGKey(0), 512))
    cmd, freq = np.asarray(cmd), np.asarray(freq)

    v_max = np.minimum(env.max_speed_per_hz * freq, env.max_speed)
    v_max = v_max * env.feasibility_margin
    assert np.all(np.abs(cmd[:, 0]) <= v_max + 1e-5)
    assert np.all(np.abs(cmd[:, 1]) <= 0.5 * v_max + 1e-5)


@pytest.mark.skipif(not os.path.exists(XML), reason="mujoco_menagerie not present")
def test_mjx_env_resets_steps_and_vmaps():
    """End-to-end: the environment traces, jits and vmaps, and the physics is sane.

    Deliberately small (4 environments, 3 steps). On CPU jax this is slow and
    proves nothing about throughput - it proves the program is well-formed,
    which is the part that can be checked without a GPU.
    """
    pytest.importorskip("mujoco.mjx")
    from mjx.mjx_env import Go2MJXEnv

    env = Go2MJXEnv()
    assert env.obs_size == 50
    assert env.action_size == 12
    # Bug B1 again, on the MJX side: the nominal pose must be the home keyframe.
    assert np.allclose(np.asarray(env.default_joint_pos),
                       np.tile([0.0, 0.9, -1.8], 4), atol=1e-6)

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    state = reset(jax.random.split(jax.random.PRNGKey(0), 4))
    assert state["obs"].shape == (4, 50)
    assert np.all(np.isfinite(np.asarray(state["obs"])))

    action = jnp.zeros((4, 12))
    for _ in range(3):
        state, terms = step(state, action)

    assert np.all(np.isfinite(np.asarray(state["reward"])))
    heights = np.asarray(state["data"].qpos[:, 2])
    assert np.all((heights > 0.1) & (heights < 0.6)), heights
    # Every configured non-zero term must be reported, same as the CPU env.
    expected = {k for k, w in env.weights.items() if w != 0.0}
    assert set(terms) == expected
