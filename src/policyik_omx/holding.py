"""OMX simulation position-hold compensation from servo feedback."""

import numpy as np
import torch
from torch import nn

from .command import scaled_error


class HoldingPrior(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(15,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,5))

    def forward(self,q):
        return .025 * torch.tanh(super().forward(torch.cat((q,q.sin(),q.cos()),dim=-1)))


class SmoothHolding:
    """Ramp in a pose-dependent bias and retain measured correction while holding."""

    def __init__(self, policy, prior):
        self.policy = policy
        self.prior = prior
        self.reset()

    def reset(self):
        self.policy.reset()
        self.target_pose = None
        self.offset = np.zeros(5)
        self.settled_ticks = 0

    def target(self, goal, q, dq, last, lo, hi, *, forward):
        command = self.policy.target(goal,q,dq,last,lo,hi,forward=forward)
        same_goal = self.target_pose is not None and np.max(np.abs(scaled_error(goal,self.target_pose))) <= .001
        slow = np.max(np.abs(dq[:5]))<=.075
        close = np.max(np.abs(scaled_error(goal,forward(q))))<=.060
        self.settled_ticks = self.settled_ticks+1 if same_goal and slow and close else 0
        self.target_pose = goal.copy()
        bias = np.zeros(5)
        if self.settled_ticks>=3:
            prediction = self.prior(torch.as_tensor(q[None,:5],dtype=torch.float32))[0].numpy()
            measured_bias = np.clip(last[:5]-q[:5],-.025,.025)
            weight = .3 / (1.+(self.settled_ticks-3)/4.)
            bias = weight*prediction+(1.-weight)*measured_bias
        self.offset += np.clip(.32*(bias-self.offset),-.0012,.0012)
        lo_arm = np.maximum(lo[:5],np.maximum(last[:5]-self.policy.max_step,q[:5]-self.policy.max_lead))
        hi_arm = np.minimum(hi[:5],np.minimum(last[:5]+self.policy.max_step,q[:5]+self.policy.max_lead))
        if np.all(lo_arm<=hi_arm):
            command[:5] = np.clip(command[:5]+self.offset,lo_arm,hi_arm)
        return command
