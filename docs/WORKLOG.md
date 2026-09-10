# Worklog

A running record of what has been done, what is true right now, and what is
next. Updated as work happens.

This is the *narrative*. For the **explanation** of how any of it works — the
RL maths, the robot, the reward design — see [the book](README.md), 17 chapters
starting at [chapter 1](01-what-is-rl.md). For the **defect list** with evidence
and reasoning, see [chapter 14](14-debugging-log.md). This file exists so that
someone returning after a month can reconstruct the state in five minutes
without reading either.

---

## Status at a glance

*Last updated: 2026-09-11*

| | |
|---|---|
| Branch | `master`, pushed |
| Tests | 79 (78 run on CPU, 1 needs a GPU), all passing (`make test`) |
| Trained policy | `models/go2_ppo_final.zip` + `models/go2_ppo_vecnormalize.pkl` (CPU, trained before the B21 fix) |
| Training run | 18.76M steps, 16 envs, seed 0, ~8 h at ~600 steps/s |
| CPU training | works |
| GPU training | **works since B22** — 2M-step smoke run learns cleanly; no GPU policy evaluated yet |

**What the policy does.** Tracks velocity commands to 0.021 m/s (forward),
0.028 m/s (lateral) and 0.035 rad/s (yaw) over 30 random commands from the
trained envelope, 100% survival, mean 2.00 of 4 feet in contact. Trots at
96.0% schedule match with a duty factor of exactly 0.500 and stride frequency
exactly the commanded 2.00 Hz. Drivable by hand with `make play`.

**What it does not do.** Change gait on command (trained on trot only — the
gait identity is not in its observation, see [11 §11.8](11-gaits-and-phase.md)).
Exceed ~0.85 m/s. Transfer to hardware ([16](16-sim-to-real.md)).

---

## Decision log

Choices that would be expensive to rediscover, with why and whether they still
hold.

| Decision | Rationale | Status |
|---|---|---|
| Nominal pose from the `home` keyframe | `qpos0` puts the calf joints outside their own limits (B1) | settled |
| All velocities in the base frame | world-frame tracking makes turning incoherent (B2) | settled |
| Gait as a commanded phase clock, not emergent | emergent gaits are uncontrollable and exploration cannot find a limit cycle unaided ([11](11-gaits-and-phase.md)) | settled |
| Reward terms as separate pure functions | ablation by config, per-term logging, unit-testable in isolation | settled |
| Weights are per second, scaled by `dt` | changing control rate must not silently rescale the objective | settled |
| $k_p=55$, $\alpha=0.40$ | measured: sag eats swing clearance (B17), stride sets top speed (B20), and $k_p\alpha=22$ N·m caps against the 23.7 N·m actuator limit | settled |
| Commands clamped to a measured speed envelope | infeasible commands saturate the tracking kernel and become noise (B20) | settled |
| CPU MuJoCo env stays canonical; MJX is an accelerator | MJX uses simplified collision geometry, so evaluation must happen on the unsimplified model | settled |
| Reward maths shared between backends via `_backend()` | two implementations of a 17-term reward will diverge silently | settled, tested |
| Actor keeps privileged base linear velocity | sim-to-real is out of scope; with no asymmetry there is nothing for an asymmetric critic to exploit ([2 §2.6](02-value-functions-and-policies.md)) | revisit for hardware |
| `gaits: ["trot"]` only | multi-gait needs the per-foot clock in the observation | **open — next task** |

---

## Log

### 2026-09-11 — B22: the GPU robot was never being controlled

The first real GPU smoke run went NaN at 491k steps. The earlier signal was
better: brax's step-0 eval runs a near-zero-action policy, and those episodes
lasted 27 steps and got *shorter* with training.

Driving `Go2MJXEnv` directly at zero action (256 envs, no brax wrappers) showed
the robot jumping 9 cm off the floor by step 11 and 239 of 256 environments
dead by step 30 — identically with randomisation and pushes on or off, and with
no NaN in the physics. Cause: Menagerie's `scene_mjx.xml` uses *position*
actuators (ctrl is a joint angle), `scene.xml` uses *motors* (ctrl is a torque),
and the env writes torques. Every calf was driven to its straight-leg limit.
Full write-up in [chapter 14, B22](14-debugging-log.md); B21 is written up there
now too.

Fix: `make_torque_actuators()` converts the MJX actuators to motors with the CPU
model's limits, so the PD law stays identical across backends. After it, zero
action gives 0 of 256 terminated, 4.00 of 4 feet down, and a settled height of
0.252 m against the CPU's 0.254 m. Two regression tests, both confirmed to fail
with the fix reverted.

2M-step smoke run after the fix (2048 envs, seed 0, all robustness features on):

| step | eval reward | episode length |
|---|---|---|
| 0 | 20.47 | 1000 |
| 0.98M | 35.56 | 1000 |
| 2.21M | 56.22 | 1000 |

Monotonic at every one of ten evals, full-length episodes throughout, no NaN.
About 30 minutes wall-clock, of which roughly 10 was XLA compiling the eval
loop (it prints a "very slow compile" warning; that is normal here). Saved as
`models/go2_mjx_smoke2m.pkl`.

This proves the pipeline learns. It does not prove the robot walks: there is no
way yet to run a brax policy in the CPU environment (open item 1).

Two earlier smoke attempts that day died as clean VM shutdowns with no error.
That was WSL being migrated to a new drive mid-run, not the code — noted so
nobody chases it.

### 2026-08-31 — GPU training path (in progress)

Assessed the hardware for the MJX/GPU path.

- **GPU: RTX 3050 Laptop, 4 GB VRAM.** This matters. The literature figure of
  4096 parallel environments assumes a datacentre card; 4 GB will support far
  fewer. The env count needs measuring, not assuming — same discipline as B20.
- WSL2 Ubuntu present, CUDA driver visible inside it
  (`/usr/lib/wsl/lib/libcuda.so`), Python 3.12.3, 937 GB free.
- No `sudo` needed for the install itself: JAX ships CUDA as pip wheels, so a
  plain venv suffices.
- **Blocked:** `/etc/wsl.conf` sets `generateResolvConf = false` and
  `/etc/resolv.conf` does not exist, so WSL has no DNS. Routing is fine (TCP to
  `1.1.1.1:53` succeeds). `sudo` requires a password. Fix is one command, run by
  the user:
  ```
  sudo bash -c 'printf "nameserver 1.1.1.1\nnameserver 8.8.8.8\n" > /etc/resolv.conf'
  ```
  Chosen over re-enabling `generateResolvConf` because that flag was set
  deliberately and is probably load-bearing for the ROS setup on this machine.

### 2026-08-31 — `743cc64` Model paths resolve against the repository

Running any script by absolute path from outside the repo root died in
`MjModel.from_xml_path` with `ParseXML: Error opening file`, naming neither the
file nor the reason. `resolve_asset_path()` now falls back to the repo root and,
on genuine absence, raises an error naming both directories searched and the
`git clone` that fixes it. Two regression tests.

Also capped the demo GIF: it was 38 MB, now 7.7 MB.

### 2026-08-31 — `6feb8e7` Results

Training reached 18.76M of 20M steps before the process died with the session;
the save-on-exit path held. Evaluated properly for the first time. Numbers in
[chapter 15](15-results.md); headline is 0.021 / 0.028 m/s and 0.035 rad/s
tracking with 100% survival.

Found and fixed a flaw in my own measurement: `evaluate.py` was sampling random
commands *wider than the trained envelope*, which measures extrapolation rather
than tracking, and was inflating the reported forward error from 0.021 to
0.152 m/s. It now reads the trained ranges from the config, with
`--full-envelope` to probe extrapolation deliberately.

### 2026-08-28 — `3956d74` B20, and the end of three stalls

Three successive runs stalled with the robot standing or trotting on the spot.
Each stall had a distinct cause, and each was found by measurement rather than
by tuning:

1. **B18** — `gait_phase` is built on a binary contact flag, so its gradient
   toward stepping is exactly zero until a foot breaks contact. Added
   `feet_clearance`, a smooth cost on swing-foot height.
2. **B19** — the stride reward, inherited from legged_gym, made a *correct trot
   score four times worse than standing still*. Its 0.5 s offset assumes an
   emergent gait frequency; here frequency is commanded, so at 2 Hz the
   scheduled swing is 0.25 s and every correctly-timed step scored −0.25.
3. **B20** — the command envelope asked for 1.5 m/s when the leg geometry
   delivers ~0.4 m/s at 2 Hz at the then-current action scale. Unreachable
   commands saturate the exponential tracking kernel, so the policy gets no
   gradient and the command channel becomes noise.

The generalisable check, now the first thing `make check` prints: **score your
target behaviour under your reward and compare it against doing nothing.** If
the thing you want does not out-score the trivial policy, no algorithm finds it.

### 2026-08-28 — `f4640fb` The v2 rewrite

Sixteen defects in the v1 environment and training script, none of which
produced an error message, all of which survived five documented training runs.
Full evidence in [chapter 14](14-debugging-log.md). The root cause of v1's
ceiling: the nominal pose was `qpos0`, which places all four calf joints
**outside their own travel limits**, and the policy's reachable target set did
not intersect the calf's legal range at all.

Added in the same pass: command-conditioned locomotion, the gait phase clock,
17 reward terms as pure functions, a closed-loop curriculum, an annotated
from-scratch PPO, the MJX backend, 67 tests, and the 17-chapter book.

---

## Open items

Ordered by what I would do next.

1. **Evaluate a GPU policy on the CPU environment.** Nothing can do this yet:
   `scripts/evaluate.py` and `scripts/play.py` load SB3 checkpoints only. Needs
   a loader for brax params (`models/go2_mjx_*.pkl`) that rebuilds the policy
   network and applies brax's observation normaliser, then drives `Go2Env`.
   Until it exists, a GPU run's reward curve says the pipeline learns, not that
   the robot walks - and MJX's simplified collision model means the CPU
   environment is the one whose verdict counts.
1b. **Full GPU run**, then evaluate it as above. Replaces the B21 caveat in the
   README with a clean result.
2. **Multi-gait.** The mechanism exists and is unused. Swap the observation's
   2-number global clock for the 8-number per-foot clock (`clock_signal` in
   `envs/gait.py`), bump `obs_dim`, mirror in `mjx/mjx_env.py`, widen `gaits` in
   the config, update the test asserting 50 dims. Then retrain. This is the last
   capability gap the README admits to.
3. ~~Push the branch.~~ Done.
4. **From-scratch PPO parity run.** `ppo_from_scratch/` has never been run
   against SB3 on a matched seed. That comparison is the payoff for writing it.
5. **Rough terrain.** `scripts/generate_terrain.py` and
   `configs/terrain_rough.xml` survive from v1 as a starting point.
6. **Sim-to-real.** The ordered list is [chapter 16 §16.8](16-sim-to-real.md).
   Step 1 is dropping base linear velocity from the actor, which everything
   else depends on.

## Known traps on this machine

- **WSL has no DNS** unless `/etc/resolv.conf` is written by hand;
  `generateResolvConf = false` is set in `/etc/wsl.conf` and WSL therefore never
  creates the file. Restore WSL's own backup:
  `wsl -d Ubuntu -u root cp /etc/resolv.conf.wslbak /etc/resolv.conf`.
  All four candidate resolvers (`10.255.255.254`, the gateway, `1.1.1.1`,
  `8.8.8.8`) answer once the file exists - the missing file is the whole fault,
  so `generateResolvConf = false` can stay as set.
- **`sudo` in WSL requires a password, but `wsl -u root` does not.** Any admin
  step is a passwordless one-liner from PowerShell.
- **`python3-venv` is not installed** and `apt` needs root, so
  `python3 -m venv` fails on `ensurepip`. Sidestep it without root:
  `python3 -m venv --without-pip`, then bootstrap pip inside the venv with
  PyPA's `get-pip.py`.
- **Put WSL venvs on ext4 (`~`), not `/mnt/d`.** The 9p filesystem makes
  imports and package installs painfully slow.
- **PowerShell 5.1 does not accept `&&`** as a statement separator. Use `;`.
- **`make` needs the venv path on Windows:** `make PY=venv/Scripts/python.exe …`
- **`brax` 0.14.2 calls `jax.device_put_replicated`**, removed in JAX 0.11.
  `requirements-mjx.txt` pins JAX 0.7.2 for this reason.
- **`pip install -r requirements.txt` gives the CPU build of PyTorch.**
- **brax's `num_evals` defaults to 1.** `progress_fn` fires once per eval, so a
  long run prints *nothing* until it finishes. Combined with piping through
  `grep` (which block-buffers on a non-tty), a healthy 2-hour run is
  indistinguishable from a hung one. `mjx/train_mjx.py` now defaults
  `--num-evals 20` and flushes; never pipe a long run through `grep`.
