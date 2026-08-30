"""Environments.

Imports here are **lazy**, via PEP 562's module ``__getattr__``.

The reason is concrete rather than stylistic. `mjx/mjx_env.py` needs
``envs.gait``, ``envs.rewards``, ``envs.commands`` and ``envs.paths`` — none of
which depend on anything heavier than numpy. But a plain
``from envs import gait`` executes this file first, and if this file eagerly
imports ``Go2Env`` it pulls in gymnasium and ``mujoco.viewer`` along with it.
That made the GPU training path fail with ``ModuleNotFoundError: No module
named 'gymnasium'`` in a WSL venv that has JAX and MJX and no reason whatsoever
to need a Gymnasium wrapper or an OpenGL viewer.

So the heavy names still resolve — ``from envs import Go2Env`` works exactly as
before — but only when someone actually asks for them.
"""

from envs.commands import (  # noqa: F401  (numpy only)
    Command,
    CommandCurriculum,
    CommandRanges,
    CommandSampler,
)
from envs.gait import FOOT_ORDER, GAIT_NAMES, GAITS  # noqa: F401  (numpy only)
from envs.paths import REPO_ROOT, resolve_asset_path  # noqa: F401  (stdlib only)

__all__ = [
    "Go2Env",
    "Command",
    "CommandRanges",
    "CommandCurriculum",
    "CommandSampler",
    "GAITS",
    "GAIT_NAMES",
    "FOOT_ORDER",
    "REPO_ROOT",
    "resolve_asset_path",
]

_LAZY = {"Go2Env": "envs.go2_env"}
_GYM_ID_REGISTERED = False


def __getattr__(name):
    """Resolve the gymnasium-dependent names on first use (PEP 562)."""
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value
        _register_gym_id()
        return value
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def _register_gym_id():
    """Register Go2Walk-v0 with Gymnasium, once, and only if gymnasium is here.

    Registration used to happen at import time, which is another reason this
    file could not be imported without gymnasium installed.
    """
    global _GYM_ID_REGISTERED
    if _GYM_ID_REGISTERED:
        return
    try:
        import gymnasium as gym

        if "Go2Walk-v0" not in gym.registry:
            gym.register(
                id="Go2Walk-v0",
                entry_point="envs.go2_env:Go2Env",
                max_episode_steps=1000,
            )
        _GYM_ID_REGISTERED = True
    except ImportError:
        pass


def __dir__():
    return sorted(__all__)
