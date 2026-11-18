"""Regression for gripper target reversals near the joint boundary."""
import unittest

import numpy as np

from policyik_omx.braking import braked_gripper, stopping_distance
from policyik_omx.reference import ReferenceEnvelope


class TestBraking(unittest.TestCase):
    def test_random_waypoints_retain_space_to_stop(self):
        dt, speed, acceleration = .04, .8, 2.
        rng = np.random.default_rng(146)
        for start in (0., .12, .37, .739, .74):
            angle, velocity = start, 0.
            for goal in np.r_[np.tile([.74, .66, .74, 0., .09, 0.], 20), rng.uniform(0., .74, 500)]:
                delta = braked_gripper(angle,velocity,goal,0.,.74,dt,speed,acceleration)
                new_speed = delta/dt
                self.assertLessEqual(abs(new_speed-velocity),acceleration*dt+1e-7)
                self.assertLessEqual(abs(new_speed),speed)
                stop = angle+stopping_distance(new_speed,dt,acceleration)
                self.assertGreaterEqual(stop,-1e-10)
                self.assertLessEqual(stop,.74+1e-10)
                angle += delta
                velocity = new_speed

    def test_original_gripper_target_is_unchanged(self):
        start = np.array([0.,-.5,-.6,.7,0.,.73])
        target = start.copy()
        target[5] = .74
        limits = (np.r_[np.full(5,-np.pi),0.], np.r_[np.full(5,np.pi),.74])
        envelope = ReferenceEnvelope(start,*limits,lambda _: .04,gripper_braking=True)
        for _ in range(20):
            pos = envelope.accept(target,envelope.position.copy(),.04)
        self.assertEqual(target[5],.74)
        self.assertAlmostEqual(pos[5],.74,places=6)


if __name__=="__main__":
    unittest.main()
