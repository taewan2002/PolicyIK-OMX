"""Learned correction for a PolicyIK joint setpoint under the OMX Cartesian action contract."""
import numpy as np
import torch
from torch import nn


class CommandCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(15, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 25))

    def forward(self, joints, error):
        features = torch.cat((joints, torch.sin(joints), torch.cos(joints)), dim=-1)
        matrix = self.layers(features).reshape(-1, 5, 5)
        return torch.bmm(matrix, error.unsqueeze(-1)).squeeze(-1)


def scaled_error(goal, pose):
    delta = np.asarray(goal)[..., :5] - np.asarray(pose)[..., :5]
    delta[..., 3:] = (delta[..., 3:] + np.pi) % (2*np.pi) - np.pi
    return delta * [10., 10., 10., 1., 1.]


class CommandSpacePolicyIK:
    """Start with the actor command and refine its FK against the same current waypoint."""
    def __init__(self, actor, correction, *, max_step, max_lead):
        self.actor = actor
        self.correction = correction.eval()
        self.max_step = max_step
        self.max_lead = max_lead

    def reset(self):
        self.actor.reset()

    @torch.inference_mode()
    def target(self, goal, q, dq, last, lo, hi, *, forward):
        cmd = self.actor.target(goal, q, dq, last, lo, hi, forward=forward)
        lower = np.maximum(lo[:5], np.maximum(last[:5]-self.max_step, q[:5]-self.max_lead))
        upper = np.minimum(hi[:5], np.minimum(last[:5]+self.max_step, q[:5]+self.max_lead))
        if np.any(lower > upper):
            return cmd
        err = scaled_error(goal, forward(cmd))
        # The residual is trained on joint commands paired with their FK, independent of the task.
        for _ in range(3):
            joints = torch.as_tensor(cmd[None,:5], dtype=torch.float32)
            delta = self.correction(joints, torch.as_tensor(err[None], dtype=torch.float32))[0].numpy()
            if not np.isfinite(delta).all():
                raise ValueError('Nonfinite learned joint correction')
            trial = cmd.copy()
            trial[:5] = np.clip(cmd[:5]+delta, lower, upper)
            trial_err = scaled_error(goal, forward(trial))
            if np.dot(trial_err,trial_err) >= np.dot(err,err):
                break
            cmd, err = trial, trial_err
        return cmd



class HoldingOffsetPolicyIK:
    """Estimate steady servo deflection from accepted references and encoders at rest."""

    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        self.policy.reset()
        self.last_reference = None
        self.last_goal = None
        self.offset = np.zeros(5)
        self.settled_ticks = 0

    def target(self, goal, q, dq, last, lo, hi, *, forward):
        nominal = self.policy.target(goal, q, dq, last, lo, hi, forward=forward)
        stable = False
        if self.last_goal is not None:
            change = scaled_error(goal, self.last_goal)
            stable = (np.linalg.norm(change[:3]) <= .001 and np.linalg.norm(change[3:]) <= .001
                      and abs(goal[5] - self.last_goal[5]) <= .005
                      and np.max(np.abs(last[:5] - self.last_reference)) <= .0015
                      and np.max(np.abs(dq[:5])) <= .05)
        self.last_goal = goal.copy()
        self.last_reference = last[:5].copy()
        self.settled_ticks = self.settled_ticks + 1 if stable else 0
        if not stable:
            self.offset.fill(0.)
        if self.settled_ticks < 4:
            return nominal
        deflection = np.clip(last[:5] - q[:5], -.025, .025)
        self.offset += .20 * (deflection - self.offset)
        lower = np.maximum(lo[:5], np.maximum(last[:5] - self.policy.max_step, q[:5] - self.policy.max_lead))
        upper = np.minimum(hi[:5], np.minimum(last[:5] + self.policy.max_step, q[:5] + self.policy.max_lead))
        if np.all(lower <= upper):
            nominal[:5] = np.clip(nominal[:5] + self.offset, lower, upper)
        return nominal
