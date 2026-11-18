# PolicyIK-OMX

A research implementation of learned pose tracking for the five-axis ROBOTIS OM-X arm. It follows the idea of [PolicyIK](https://arxiv.org/abs/2611.05372) and uses an OM-X task-space interface. The policy reads a desired pinch pose, measured joint position and velocity, and the last accepted joint reference. It outputs a reference for the five arm joints and preserves the sixth channel for the gripper.

## Checkpoints

- **Sim-IK** combines a PPO actor, a learned FK-based workspace correction, and a holding correction trained on simulated actuator deflection. The three CPU-loadable weights are in [checkpoints/sim](checkpoints/sim). They can be reused with the same OM-X kinematics and action interface. Validate the mounting, workspace, actuator model, and limits for each deployment.

OM-X actions use `[x, y, z, pitch, heading, gripper]` in meters and radians. Goals are absolute in the calibrated arm frame. Encoder-derived poses use the analytic **forward** kinematics. The learned controller produces joint references. Pass them through the robot's limits, clearance, speed, and acceleration checks before sending them.

## Run a CPU example

```bash
python -m pip install -e '.[sim]'
python examples/shadow_sim.py
python -m unittest discover -s tests
```

```python
from policyik_omx.kinematics import forward
from policyik_omx.sim import load_sim

policy = load_sim("checkpoints/sim")
reference = policy.target(goal, q, dq, last_accepted, low, high, forward=forward)
```

The rolling reference limiter is available as `policyik_omx.reference.ReferenceEnvelope`. Supply clearance computed from your calibrated arm geometry. The `gripper_braking` option reserves the distance needed to stop before reaching the existing joint limits.

## Adapt the simulator checkpoints

Start from the supplied actor for the same five-axis geometry. The release actor was trained with PPO using reachable FK-generated goals, a running joint reference, coarse and fine pose-error features, and rewards for reaching and holding. Refit the workspace and holding modules when the simulated arm or motor dynamics change.

```bash
python -m policyik_omx.train_sim workspace data/joint-pairs.npz runs/workspace.pt --init checkpoints/sim/workspace.pt
python -m policyik_omx.train_sim hold data/hold-samples.npz runs/hold_prior.pt
```

The workspace file contains six-channel arrays `start` and `end` of nearby joint positions within the simulator's valid space. Training labels are the known joint change and the difference between their two FK poses. The holding file contains six-channel `command` and five-channel `deflection` arrays. Hold each reference until settled, then record command minus measured joint position. Split both datasets by pose region or recording session before selecting weights. Reproducing the original actor requires the original simulator setup and training trajectories.

## Citation and license

Please cite Yuhan Liu, Xinqi Liu, Zixuan Dong, Keith W. Ross, and Shenji Wan, **PolicyIK: High-Precision Inverse Kinematics for Robot Arms with Reinforcement Learning**, arXiv:2611.05372, 2026. See [CITATION.cff](CITATION.cff) and the [original implementation](https://github.com/Tinaliu0123/PolicyIK). Code and released weights are licensed under [Apache 2.0](LICENSE).

## Evaluation snapshot

The fixed simulation panel contains 96 reachable initial/goal pairs, run both as stationary goals and as moving trajectories that end in a hold. Success requires staying within 1 mm and 0.2° throughout the final second, with no guard stop or desk contact.

| Controller | Fixed goal | Moving goal then hold | Guard stops, fixed / moving |
|---|---:|---:|---:|
| Analytical IK with holding correction | 69/96 | 60/96 | 2 / 1 |
| Earlier Sim-IK | 52/96 | 42/96 | 10 / 9 |
| Sim-IK with learned holding and braking checks | **75/96** | **65/96** | 7 / 5 |

The simulation benchmark used the simulator's robot mesh for table clearance and a braking-distance check around the controller. Its test targets are in [benchmarks/sim-goals.json](benchmarks/sim-goals.json). On 12 paired pick-and-place scenes, the gripper braking update reduced all guard stops from 7 to 3 and gripper limit stops from 5 to 0. Lifts increased from 7 to 8, and placement was 0/12 in both conditions. Both runs used the same high-level policy, initial scenes, gripper target interface, and hard motion bounds. Each tick keeps the gripper target requested by the policy. The executed reference changes during braking. See [results and file digests](checkpoints/manifest.json).
