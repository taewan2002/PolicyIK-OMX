"""Portable checks of the pose and independent IK interface."""
import unittest

import numpy as np

from policyik_omx.kinematics import forward, wrap


class TestInterface(unittest.TestCase):
    def setUp(self):
        self.q = np.array([0.,-.5,-.6,.7,0.,.30])

    def test_pose_shape_and_gripper(self):
        poses = forward(np.stack((self.q,self.q)))
        self.assertEqual(poses.shape,(2,6))
        np.testing.assert_array_equal(poses[:,5],.30)

    def test_orientation_wrap(self):
        self.assertAlmostEqual(wrap(-np.pi+.01-(np.pi-.01)),.02)


if __name__=="__main__":
    unittest.main()
