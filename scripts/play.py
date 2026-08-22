"""Drive the trained policy by hand in the MuJoCo viewer.

This is the payoff script for v2. v1 could only be watched going forwards at a
fixed speed; here you steer it.

    python scripts/play.py --model models/go2_ppo_final.zip

Controls
    W / S      forward / backward velocity
    A / D      strafe left / right
    Q / E      yaw rate left / right
    SPACE      stop (zero command, switches to the standing schedule)
    1..5       gait: trot, pace, bound, walk, pronk
    [ / ]      step frequency down / up
    - / =      body height down / up
    R          reset the episode
    P          print the current command
    ESC        quit (handled by the viewer)

Notes on how this works: the command is just three numbers in the observation,
so changing it live is enough - no retraining, no separate policies per
direction. The gait is a phase schedule the policy was trained to match, so
switching gait mid-stride is likewise just an input change.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from envs.commands import CommandRanges  # noqa: E402
from envs.gait import GAIT_NAMES  # noqa: E402
from envs.go2_env import Go2Env  # noqa: E402

GAIT_KEYS = {
    ord("1"): "trot",
    ord("2"): "pace",
    ord("3"): "bound",
    ord("4"): "walk",
    ord("5"): "pronk",
}

VEL_STEP = 0.1
YAW_STEP = 0.2
FREQ_STEP = 0.1
HEIGHT_STEP = 0.01


class Teleop:
    """Mutable command state driven by the viewer's key callback."""

    def __init__(self, env):
        self.env = env
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.gait = "trot"
        self.freq = 2.0
        self.height = 0.30
        self.reset_requested = False

    def apply(self):
        self.env.set_command(
            lin_vel_x=self.vx,
            lin_vel_y=self.vy,
            ang_vel_yaw=self.yaw,
            # A zero command uses the standing schedule, so the robot plants its
            # feet instead of marching on the spot.
            gait="stand" if self._is_stopped() else self.gait,
            gait_frequency=self.freq,
            base_height=self.height,
        )

    def _is_stopped(self):
        return abs(self.vx) < 0.05 and abs(self.vy) < 0.05 and abs(self.yaw) < 0.05

    def describe(self):
        return (
            "vx=%+.2f  vy=%+.2f  yaw=%+.2f  gait=%-6s  freq=%.1fHz  h=%.2fm"
            % (self.vx, self.vy, self.yaw,
               "stand" if self._is_stopped() else self.gait,
               self.freq, self.height)
        )

    def on_key(self, keycode):
        if keycode == ord("W"):
            self.vx = min(self.vx + VEL_STEP, 2.0)
        elif keycode == ord("S"):
            self.vx = max(self.vx - VEL_STEP, -1.5)
        elif keycode == ord("A"):
            self.vy = min(self.vy + VEL_STEP, 1.0)
        elif keycode == ord("D"):
            self.vy = max(self.vy - VEL_STEP, -1.0)
        elif keycode == ord("Q"):
            self.yaw = min(self.yaw + YAW_STEP, 2.0)
        elif keycode == ord("E"):
            self.yaw = max(self.yaw - YAW_STEP, -2.0)
        elif keycode == ord(" "):
            self.vx = self.vy = self.yaw = 0.0
        elif keycode in GAIT_KEYS:
            self.gait = GAIT_KEYS[keycode]
        elif keycode == ord("["):
            self.freq = max(self.freq - FREQ_STEP, 0.8)
        elif keycode == ord("]"):
            self.freq = min(self.freq + FREQ_STEP, 4.0)
        elif keycode == ord("-"):
            self.height = max(self.height - HEIGHT_STEP, 0.22)
        elif keycode == ord("="):
            self.height = min(self.height + HEIGHT_STEP, 0.38)
        elif keycode == ord("R"):
            self.reset_requested = True
        elif keycode == ord("P"):
            print(self.describe())
            return
        else:
            return
        self.apply()
        print("\r" + self.describe() + "   ", end="", flush=True)


def load_policy(model_path, vecnormalize_path):
    """Return ``predict(obs) -> action``.

    The VecNormalize statistics are part of the policy: the network was trained
    on normalised observations, so replaying it on raw ones produces garbage.
    This is the same failure as bug B12, just at inference time.
    """
    import pickle

    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")

    obs_rms = None
    clip_obs = 10.0
    epsilon = 1e-8
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        # VecNormalize.load wants a live venv to attach to. We only need the
        # running statistics, so unpickle the object directly.
        with open(vecnormalize_path, "rb") as f:
            vec = pickle.load(f)
        obs_rms = vec.obs_rms
        clip_obs = vec.clip_obs
        epsilon = vec.epsilon
        print("Loaded observation normalisation from " + vecnormalize_path)
    else:
        print(
            "WARNING: no VecNormalize statistics found. The policy was trained "
            "on normalised observations; without them it will not walk."
        )

    def predict(obs):
        x = obs
        if obs_rms is not None:
            x = np.clip(
                (obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                -clip_obs,
                clip_obs,
            ).astype(np.float32)
        action, _ = model.predict(x, deterministic=True)
        return action

    return predict


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="models/go2_ppo_final.zip")
    p.add_argument("--vecnormalize", default=None)
    p.add_argument("--xml", default="mujoco_menagerie/unitree_go2/scene.xml")
    p.add_argument("--random", action="store_true",
                   help="Drive with random actions; for checking the viewer "
                        "and the teleop plumbing without a trained model.")
    p.add_argument("--realtime", type=float, default=1.0,
                   help="Playback speed multiplier.")
    args = p.parse_args()

    env = Go2Env(
        xml_path=args.xml,
        randomize_dynamics=False,
        push_enabled=False,
        max_episode_steps=10 ** 9,      # never truncate; you are driving
        command_ranges=CommandRanges(),
        command_resample_interval_s=0.0,  # teleop owns the command
    )

    if args.random:
        def predict(_obs):
            return env.action_space.sample()
    else:
        vecnorm = args.vecnormalize
        if vecnorm is None:
            guess = os.path.splitext(args.model)[0].replace("_final", "")
            guess += "_vecnormalize.pkl"
            vecnorm = guess if os.path.exists(guess) else None
        predict = load_policy(args.model, vecnorm)

    teleop = Teleop(env)
    obs, _ = env.reset(seed=0)
    teleop.apply()

    print(__doc__.split("Controls")[1].split("Notes")[0])
    print("Available gaits: " + ", ".join(GAIT_NAMES))
    print(teleop.describe())

    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=teleop.on_key
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.base_body_id
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -15

        while viewer.is_running():
            tic = time.perf_counter()

            if teleop.reset_requested:
                obs, _ = env.reset()
                teleop.apply()
                teleop.reset_requested = False

            action = predict(obs)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                print("\nfell over - resetting")
                obs, _ = env.reset()
                teleop.apply()

            viewer.sync()
            elapsed = time.perf_counter() - tic
            sleep = env.dt / max(args.realtime, 1e-6) - elapsed
            if sleep > 0:
                time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    main()
