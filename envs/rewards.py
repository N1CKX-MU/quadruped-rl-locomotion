"""Reward terms, one pure function each.

Design rationale
----------------
In v1 the reward was a single 40-line method with eight inline terms and their
weights hard-coded in the middle of the arithmetic. Three consequences, all of
which we hit:

* You cannot ablate a term without editing code.
* You cannot log a term without editing code, so you cannot see *which* term is
  driving behaviour, so debugging is guesswork.
* A broken term (v1's gait reward, which a motionless robot maximised) hides in
  the noise because nothing plots it separately.

Here every term is a standalone pure function of a ``RewardState`` snapshot,
returning an *unweighted* scalar. Weights live in YAML. The environment
multiplies, sums, and logs each term individually.

Sign convention: a function named like a penalty (``torques``, ``action_rate``)
returns a NON-NEGATIVE cost, and its configured weight is negative. Keeping the
sign in the config rather than the function means a weight's sign always tells
you whether the term is a carrot or a stick.

Scaling convention: the environment multiplies the weighted sum by ``dt``. So
all weights are "reward per second", and changing the control rate does not
silently rescale the whole objective. This is the convention used by
legged_gym and it is worth keeping for comparability.

Backend note: everything is written with array operations and ``xp.where``
rather than Python ``if`` on values, so the identical file runs under
``jax.numpy`` for the MJX training path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from envs.gait import _backend


@dataclass
class RewardState:
    """Everything the reward terms are allowed to look at, in one snapshot.

    Collecting this explicitly (rather than letting terms reach into MuJoCo's
    ``data``) is what makes the terms testable in isolation and reusable under
    MJX, where there is no ``mjData`` to reach into.

    All velocities are expressed in the BASE frame. v1's bug B2 was reading
    ``qvel[0:3]``, which is the WORLD-frame linear velocity of the free joint;
    tracking that means "forward" stops meaning forward as soon as the robot
    yaws.
    """

    # Base state (base frame unless noted)
    lin_vel_b: np.ndarray            # (3,) linear velocity, base frame
    ang_vel_b: np.ndarray            # (3,) angular velocity, base frame
    proj_gravity: np.ndarray         # (3,) gravity unit vector in base frame
    base_height: float               # world-frame height of the trunk origin

    # Joints
    joint_pos: np.ndarray            # (12,)
    joint_vel: np.ndarray            # (12,)
    joint_vel_prev: np.ndarray       # (12,)
    default_joint_pos: np.ndarray    # (12,)
    soft_joint_limits: np.ndarray    # (12, 2) lower/upper, already softened
    torques: np.ndarray              # (12,) applied joint torques (Nm)

    # Actions
    action: np.ndarray               # (12,) current, normalised
    prev_action: np.ndarray          # (12,)

    # Feet, ordered (FL, FR, RL, RR)
    contact: np.ndarray              # (4,) 1.0 if loaded against the floor
    desired_contact: np.ndarray      # (4,) 1.0 if the gait schedule wants stance
    feet_air_time: np.ndarray        # (4,) seconds since this foot last landed
    feet_first_contact: np.ndarray   # (4,) 1.0 on the step a foot touches down
    feet_vel_xy: np.ndarray          # (4, 2) horizontal foot velocity, world
    feet_height: np.ndarray          # (4,) foot height above the ground, world

    # Command
    cmd: np.ndarray                  # (3,) [vx, vy, yaw_rate], base frame
    cmd_base_height: float

    # Misc
    undesired_contacts: float        # count of non-foot links touching the floor
    dt: float

    # Kernel widths for the exponential tracking terms. Exposed so that
    # docs/10-reward-engineering.md can show what changing them does, and kept
    # in sync with configs/training_config.yaml so a test that omits them is
    # testing the same objective the robot is trained on.
    lin_vel_sigma: float = 0.20
    ang_vel_sigma: float = 0.25

    # Target height for a swing foot, metres. See feet_clearance().
    target_foot_clearance: float = 0.08


# --------------------------------------------------------------------------- #
#  Task terms: what we actually want the robot to do                          #
# --------------------------------------------------------------------------- #


def track_lin_vel_xy(s):
    """Exponential kernel on planar velocity tracking error. In (0, 1].

        r = exp( -||v_cmd_xy - v_xy||^2 / sigma )

    Why exponential and not negative-squared-error: a bare ``-e^2`` is unbounded
    below, so early in training (when the error is large) it swamps every other
    term and the fastest way to improve is to stop moving. The exponential
    saturates near zero for a bad error, which makes the gradient strongest
    exactly where the policy is already close - a shaped, bounded carrot.

    Both axes are tracked. v1 tracked only x and separately punished all lateral
    motion, which made a sideways command literally impossible to satisfy.
    """
    xp = _backend(s.lin_vel_b)
    err = xp.sum(xp.square(s.cmd[:2] - s.lin_vel_b[:2]))
    return xp.exp(-err / s.lin_vel_sigma)


def track_ang_vel_yaw(s):
    """Exponential kernel on yaw-rate tracking error. In (0, 1]."""
    xp = _backend(s.ang_vel_b)
    err = xp.square(s.cmd[2] - s.ang_vel_b[2])
    return xp.exp(-err / s.ang_vel_sigma)


def gait_phase(s):
    """Agreement between actual foot contacts and the gait schedule. In [0, 1].

        r = mean_i [ c_i * d_i + (1 - c_i) * (1 - d_i) ]

    where ``c_i`` is the actual contact and ``d_i`` the scheduled one. This is
    just "fraction of feet doing the right thing".

    This term replaces v1's ``abs(diag1 - diag2)``, which had no time
    dependence at all: a robot frozen on one diagonal pair scored the maximum
    forever. The fix is not a better contact expression, it is the introduction
    of a *clock* the contact can be compared against. See envs/gait.py.

    Because the schedule is an input, the same trained policy can trot, pace or
    bound - the gait is commanded, not baked in.
    """
    xp = _backend(s.contact)
    match = s.contact * s.desired_contact + (1.0 - s.contact) * (1.0 - s.desired_contact)
    return xp.mean(match)


def feet_air_time(s):
    """Reward long swing phases, credited on touchdown. Can be negative.

        r = sum_i (t_air_i - 0.5) * first_contact_i     [gated on ||cmd|| > 0.1]

    Without this the cheapest way to satisfy the gait schedule is a rapid,
    tiny-amplitude shuffle: contacts toggle on cue, but the robot barely moves
    and the feet never really leave the ground. Paying for *airtime* forces
    genuine strides. The 0.5 s offset makes short hops cost rather than pay.

    Gated on a non-zero command, otherwise it bribes a standing robot to fidget.
    """
    xp = _backend(s.feet_air_time)
    moving = xp.where(xp.linalg.norm(s.cmd) > 0.1, 1.0, 0.0)
    return xp.sum((s.feet_air_time - 0.5) * s.feet_first_contact) * moving


def feet_clearance(s):
    """Cost on how far a SWING foot is from its target height. In [0, inf).

        r = sum_i (1 - d_i) * (h_i - h*)^2

    This is the term that makes a gait discoverable, and it is worth
    understanding why ``gait_phase`` is not sufficient on its own.

    ``gait_phase`` compares a BINARY contact flag against the schedule. Its
    value is 0.5 for a fully-planted robot under a trot schedule, and it stays
    at exactly 0.5 until a foot physically breaks contact. Raising a foot from
    9.5 mm to 9.4 mm earns nothing. The gradient is ZERO all the way up to a
    discontinuity - which is precisely the region a policy that has never
    stepped has to cross.

    Foot height is continuous, so this term pulls the swing foot upward from the
    first millimetre. Measured on the standing robot: foot geoms sit at 0.0095 m
    and the target is 0.08 m, so a planted swing foot pays about 0.005 per foot
    and can earn all of it back by lifting.

    Gated implicitly: when the command is a stand, the schedule wants every foot
    in stance, ``1 - d_i`` is zero, and the term vanishes.
    """
    xp = _backend(s.feet_height)
    swing = 1.0 - s.desired_contact
    return xp.sum(swing * xp.square(s.feet_height - s.target_foot_clearance))


# --------------------------------------------------------------------------- #
#  Regularisation terms: how we want it to look while doing that              #
# --------------------------------------------------------------------------- #


def lin_vel_z(s):
    """Cost on vertical bouncing of the trunk."""
    xp = _backend(s.lin_vel_b)
    return xp.square(s.lin_vel_b[2])


def ang_vel_xy(s):
    """Cost on roll and pitch *rate*: stops the body wobbling as it walks."""
    xp = _backend(s.ang_vel_b)
    return xp.sum(xp.square(s.ang_vel_b[:2]))


def orientation(s):
    """Cost on trunk tilt, measured by the horizontal part of gravity.

    ``proj_gravity`` is the world gravity direction expressed in the base
    frame. Perfectly level, it is (0, 0, -1) and the xy part is zero. This is a
    better attitude error than roll/pitch angles because it has no
    trigonometric singularity and no yaw dependence - yaw is a commanded
    quantity here, so the attitude cost must be blind to it.
    """
    xp = _backend(s.proj_gravity)
    return xp.sum(xp.square(s.proj_gravity[:2]))


def base_height(s):
    """Cost on deviation from the commanded ride height.

    Also the mechanism by which ride height becomes *commandable*: the target
    is part of the command vector, so the same policy can walk tall or crouch.
    """
    xp = _backend(s.proj_gravity)
    return xp.square(s.base_height - s.cmd_base_height)


def torques(s):
    """Energy proxy: sum of squared joint torques."""
    xp = _backend(s.torques)
    return xp.sum(xp.square(s.torques))


def joint_acceleration(s):
    """Cost on joint acceleration, estimated by finite difference.

    Penalising acceleration rather than only velocity is what removes the
    high-frequency judder that looks fine in a plot and destroys real gearboxes.
    The weight is tiny because the quantity is large: at 50 Hz a 1 rad/s change
    in one step is 50 rad/s^2, squared is 2500.
    """
    xp = _backend(s.joint_vel)
    return xp.sum(xp.square((s.joint_vel - s.joint_vel_prev) / s.dt))


def action_rate(s):
    """Cost on how fast the *command* to the PD controller changes."""
    xp = _backend(s.action)
    return xp.sum(xp.square(s.action - s.prev_action))


def joint_limits(s):
    """Cost on pushing joints past a softened version of their travel limits.

    MuJoCo enforces the hard limit for us, so without this term the policy
    happily saturates against the endstop and learns a gait that depends on
    leaning on it - which no real actuator will reproduce.
    """
    xp = _backend(s.joint_pos)
    below = xp.clip(s.soft_joint_limits[:, 0] - s.joint_pos, 0.0, None)
    above = xp.clip(s.joint_pos - s.soft_joint_limits[:, 1], 0.0, None)
    return xp.sum(below + above)


def collision(s):
    """Cost per non-foot link in contact with the ground (knees, shins, trunk)."""
    return s.undesired_contacts


def feet_slip(s):
    """Cost on horizontal foot velocity while that foot is loaded.

    A foot that is both in contact and moving is skating. Physically it means
    the friction cone is being violated in the real world even though MuJoCo
    let it happen; it is also the classic way a policy fakes locomotion.
    """
    xp = _backend(s.feet_vel_xy)
    return xp.sum(xp.sum(xp.square(s.feet_vel_xy), axis=-1) * s.contact)


def stand_still(s):
    """Cost on drifting away from the nominal pose when told to hold still.

    Gated the opposite way to ``feet_air_time``: active only when the command
    is (near) zero.
    """
    xp = _backend(s.joint_pos)
    standing = xp.where(xp.linalg.norm(s.cmd) < 0.1, 1.0, 0.0)
    return xp.sum(xp.abs(s.joint_pos - s.default_joint_pos)) * standing


def alive(s):
    """Constant survival bonus.

    Small on purpose. v1 used 0.5 per step against a total positive reward of
    ~2.5, and its own logs record the result: the policy learned to stand still
    because standing is the safest way to keep collecting it. Here it exists
    only to make sure the sum of the penalty terms cannot make *existing* worse
    than terminating.
    """
    xp = _backend(s.joint_pos)
    return xp.ones(())


# Registry. The keys are the names used in configs/training_config.yaml and in
# the TensorBoard traces, so renaming one here renames it everywhere.
REWARD_TERMS = {
    "track_lin_vel_xy": track_lin_vel_xy,
    "track_ang_vel_yaw": track_ang_vel_yaw,
    "gait_phase": gait_phase,
    "feet_clearance": feet_clearance,
    "feet_air_time": feet_air_time,
    "lin_vel_z": lin_vel_z,
    "ang_vel_xy": ang_vel_xy,
    "orientation": orientation,
    "base_height": base_height,
    "torques": torques,
    "joint_acceleration": joint_acceleration,
    "action_rate": action_rate,
    "joint_limits": joint_limits,
    "collision": collision,
    "feet_slip": feet_slip,
    "stand_still": stand_still,
    "alive": alive,
}

# Sensible defaults, per second (the environment multiplies by dt).
# Positive = carrot, negative = stick. Tuned starting from the legged_gym Go1
# values, with gait_phase and base_height added for v2.
DEFAULT_WEIGHTS = {
    "track_lin_vel_xy": 1.5,
    "track_ang_vel_yaw": 0.75,
    "gait_phase": 1.5,
    "feet_clearance": -30.0,
    "feet_air_time": 2.0,
    "alive": 0.25,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "orientation": -1.0,
    "base_height": -5.0,
    "torques": -2.0e-4,
    "joint_acceleration": -2.5e-7,
    "action_rate": -0.01,
    "joint_limits": -10.0,
    "collision": -1.0,
    "feet_slip": -0.05,
    "stand_still": -0.5,
}


def resolve_weights(cfg):
    """Merge a YAML ``reward.weights`` mapping over the defaults.

    Unknown keys raise rather than being silently ignored: a typo in a reward
    weight is otherwise invisible and costs you a whole training run.
    """
    weights = dict(DEFAULT_WEIGHTS)
    if cfg:
        unknown = set(cfg) - set(REWARD_TERMS)
        if unknown:
            raise KeyError(
                "unknown reward terms in config: %s. Known terms: %s"
                % (sorted(unknown), sorted(REWARD_TERMS))
            )
        weights.update({k: float(v) for k, v in cfg.items()})
    return weights


def compute(state, weights):
    """Evaluate every non-zero-weighted term. Returns ``(total, per_term)``.

    ``total`` is the weighted sum BEFORE the dt scaling, which the caller
    applies. ``per_term`` holds the already-weighted contributions, which is
    what you want in TensorBoard: it answers "how much of my reward came from
    this term" directly, without mentally multiplying by the weight.
    """
    per_term = {}
    total = 0.0
    for name, weight in weights.items():
        if weight == 0.0:
            continue
        value = REWARD_TERMS[name](state) * weight
        per_term[name] = value
        total = total + value
    return total, per_term
