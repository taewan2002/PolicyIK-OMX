"""OMX port of PolicyIK's accumulated joint reference; no analytical inverse."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import kinematics as ee

ABI = 'omx-direct-policyik-pure-ppo-v1'
STEP_RAD = .0625
REF_LEAD_RAD = .20
POS_TOL = .001
ANG_TOL = np.deg2rad(.2)
OBS = 44


def pose_error(q, goal, forward=ee.forward):
    pose = forward(q)
    e = goal[..., :5] - pose[..., :5]
    e[..., 3:] = ee.wrap(e[..., 3:])
    return pose, e


def rho(e):
    return np.maximum(np.linalg.norm(e[..., :3], axis=-1)/POS_TOL,
                      np.linalg.norm(e[..., 3:], axis=-1)/ANG_TOL)


def keypoints(pose, radius):
    xyz = pose[..., :3]
    p,y = pose[...,3],pose[...,4]
    f = np.stack([np.cos(p)*np.cos(y), np.cos(p)*np.sin(y), -np.sin(p)], axis=-1)
    s = np.stack([-np.sin(y), np.cos(y), np.zeros_like(y)], axis=-1)
    return np.stack([xyz, xyz+radius*f, xyz+radius*s],axis=-2)


def observation(q,dq,ref,prev,goal,pose,e,best):
    kp=(keypoints(goal,.30)-keypoints(pose,.30)).reshape(e.shape[:-1]+(9,))
    h=np.stack((np.clip(best/20,0,1),np.clip((rho(e)-best)/20,0,1)),axis=-1)
    return np.concatenate((q[...,:5]/np.pi,dq[...,:5]/3,goal[...,:3]/.4,kp/.08,
                   e/np.array([.01]*3+[np.deg2rad(20)]*2),
                   e/np.array([.0005]*3+[np.deg2rad(.25)]*2),
                   h,(ref[...,:5]-q[...,:5])/.15,prev),axis=-1).clip(-12,12).astype(np.float32)


def action_scale(e):
    pg = (np.linalg.norm(e[...,:3],axis=-1)-.0005)/(.00931-.0005)
    ag = (np.rad2deg(np.linalg.norm(e[...,3:],axis=-1))-.415)/(2.328-.415)
    gain=.10+.90*np.maximum(pg,ag).clip(0,1)
    return STEP_RAD*gain[...,None]


def apply(q,ref,action,e,goal,lo,hi):
    delta=action_scale(e)*np.clip(action,-1,1)
    old=ref[...,:5]
    measured=q[...,:5]
    lower=np.maximum(lo[:5],np.maximum(old-STEP_RAD,measured-REF_LEAD_RAD))
    upper=np.minimum(hi[:5],np.minimum(old+STEP_RAD,measured+REF_LEAD_RAD))
    # If a disturbed arm is already outside the reference-lead bound, walk
    # the existing setpoint toward the measured arm at the same velocity cap.
    normal=np.minimum(np.maximum(old+delta,lower),upper)
    catch_up=old+np.clip(np.clip(old,measured-REF_LEAD_RAD,measured+REF_LEAD_RAD)-old,-STEP_RAD,STEP_RAD)
    target=np.where(lower<=upper,normal,np.clip(catch_up,lo[:5],hi[:5]))
    return np.concatenate((target,np.clip(goal[...,5:6],lo[5],hi[5])),axis=-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(OBS,256),nn.ELU(),nn.Linear(256,256),nn.ELU())
        self.mu=nn.Linear(256,5)
        self.val=nn.Linear(256,1)
        self.log_std=nn.Parameter(torch.zeros(5))

    def forward(self,x):
        h=self.body(x)
        return self.mu(h),self.val(h).squeeze(-1)

    def actor(self,x):
        return self.mu(self.body(x))


class DirectIK:
    def __init__(self, path: Path):
        ck=torch.load(path,map_location='cpu',weights_only=True)
        assert ck['abi']==ABI
        self.steps=ck['steps']
        self.net=Model().eval()
        self.net.load_state_dict(ck['weights'],strict=True)
        self.reset()

    def reset(self):
        self.prev=np.zeros(5)
        self.last_goal=None
        self.best=0.

    @torch.inference_mode()
    def target(self,goal,q,dq,last,lo,hi, *, forward=ee.forward):
        if not all(np.isfinite(x).all() for x in (goal,q,dq,last)):
            raise ValueError('Nonfinite direct IK input')
        pose,e=pose_error(q,goal,forward)
        if self.last_goal is None or not np.array_equal(goal[:5],self.last_goal):
            self.best=rho(e)
        action=self.net.actor(torch.from_numpy(observation(q,dq,last,self.prev,goal,pose,e,self.best))).numpy()
        if not np.isfinite(action).all():raise ValueError('Nonfinite direct IK action')
        self.prev=np.clip(action,-1,1)
        self.last_goal=goal[:5].copy()
        self.best=min(self.best,rho(e))
        return apply(q,last,action,e,goal,lo,hi)

    def decode(self,delta,q,dq,last,lo,hi):
        # Same EE command interface as the existing decoder. No geometric inverse.
        goal=ee.goal_from_delta(ee.forward(q),delta)
        return self.target(goal,q,dq,last,lo,hi),1.
