"""Run a no-hardware inference example."""
from pathlib import Path

import numpy as np

from policyik_omx.kinematics import forward
from policyik_omx.sim import load_sim

root = Path(__file__).resolve().parents[1]
policy = load_sim(root/"checkpoints/sim")
q = np.array([0.,-.50,-.60,.70,0.,.30])
destination = q + [.015,.020,0.,0.,.025,0.]
low = np.array([-np.pi,-1.868,-np.pi/2,-1.745,-np.pi,0.])
high = np.array([np.pi,np.pi/2,np.pi/2,1.745,np.pi,.74])
goal = forward(destination)
command = policy.target(goal,q,np.zeros(6),q,low,high,forward=forward)
print(dict(goal=goal.tolist(),joint_reference=command.tolist()))
