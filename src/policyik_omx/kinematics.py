"""Cartesian action ABI for the pinned 5-DoF OMX MuJoCo arm, in m/rad."""

import numpy as np


SPACE = "omx_pinch_delta_world_xyz_pitch_yaw_gripper_v1"
# World pivot of shoulder; straight-chain link lengths from simulation/base/scene.xml.
ORIGIN = np.array([0., -.23625, .0975])
BASE_ANGLE = 1.57079632679
UPPER_X, UPPER_Z, LOWER, TOOL_X, TOOL_Y = .0415, .11315, .162, .1137, -.00165
UPPER = np.hypot(UPPER_X, UPPER_Z)
OFFSET = np.arctan2(UPPER_Z, UPPER_X)
TAU = 2 * np.pi


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def forward(q):
    """Pinch center, tool pitch, tool-axis heading, absolute gripper target.

    Heading is the top-down jaw heading continued along the arm's reachable
    orientation manifold, i.e. base heading minus rotation around tool X.
    """
    base, j2, j3, j4, wrist, grip = np.moveaxis(np.asarray(q, np.float64), -1, 0)
    b, p = BASE_ANGLE + base, j2 + j3 + j4
    sh = TOOL_Y * np.sin(wrist)
    radius = (
        UPPER_X * np.cos(j2) + UPPER_Z * np.sin(j2)
        + LOWER * np.cos(j2 + j3) + TOOL_X * np.cos(p) + sh * np.sin(p)
    )
    side = TOOL_Y * np.cos(wrist)
    z = (UPPER_Z * np.cos(j2) - UPPER_X * np.sin(j2) - LOWER * np.sin(j2 + j3)
         - TOOL_X * np.sin(p) + sh * np.cos(p))
    xyz = np.stack([np.cos(b)*radius - np.sin(b)*side,
                    np.sin(b)*radius + np.cos(b)*side, z], axis=-1) + ORIGIN
    return np.concatenate([xyz, np.stack([p, wrap(b-wrist), grip], axis=-1)], axis=-1)


def labels(states, commands):
    observed, wanted = forward(states), forward(commands)
    out = wanted.copy()
    out[..., :3] -= observed[..., :3]
    out[..., 3:5] = wrap(out[..., 3:5] - observed[..., 3:5])
    return out.astype(np.float32)


def goal_from_delta(pose, delta, scale=1.):
    goal = np.array(delta, dtype=np.float64, copy=True)
    goal[:5] = pose[:5] + scale*goal[:5]
    goal[4] = wrap(goal[4])
    return goal
