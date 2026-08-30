"""Asset path resolution, with no heavy dependencies.

This lives in its own module rather than in ``go2_env`` on purpose. The MJX
backend needs it, and importing it from ``go2_env`` would drag gymnasium,
``mujoco.viewer`` and the whole CPU environment into the GPU training path -
which then fails on a machine that has JAX but no gymnasium, for no reason
connected to what it was trying to do.

Nothing here imports anything beyond the standard library.
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_asset_path(path):
    """Resolve a model path relative to the repository, not the shell's cwd.

    The default XML path is relative, which silently made every script in this
    repository only runnable from the repository root - a script invoked by
    absolute path from anywhere else died in ``MjModel.from_xml_path`` with
    "Error opening file", which names neither the file nor the reason.

    An absolute path, or a relative one that exists from the current directory,
    is used unchanged; otherwise it is tried relative to the repo root.
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(REPO_ROOT, path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        "Could not find the MuJoCo model %r.\n"
        "Looked in the current directory (%s)\n"
        "and in the repository root (%s).\n"
        "If mujoco_menagerie is missing, fetch it with:\n"
        "    git clone --depth 1 "
        "https://github.com/google-deepmind/mujoco_menagerie.git"
        % (path, os.getcwd(), REPO_ROOT)
    )
