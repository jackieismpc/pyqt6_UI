"""双目标定的纯数据回归测试。"""

import unittest

import numpy as np

from calibration.patterns import PatternDetection
from calibration.stereo import _aligned_points


class StereoCalibrationTests(unittest.TestCase):
    def test_charuco_points_are_aligned_by_common_ids(self):
        left = PatternDetection(
            object_points=np.asarray([[i, 0, 0] for i in range(7)], dtype=np.float32),
            image_points=np.asarray([[10 + i * 10, 10] for i in range(7)], dtype=np.float32),
            debug_image=None,
            debug_info={},
            point_ids=np.asarray([4, 7, 9, 10, 11, 13, 15], dtype=np.int32),
        )
        right = PatternDetection(
            object_points=np.asarray([[i, 0, 0] for i in range(7)], dtype=np.float32),
            image_points=np.asarray([[40 + i * 10, 10] for i in range(7)], dtype=np.float32),
            debug_image=None,
            debug_info={},
            point_ids=np.asarray([7, 9, 10, 11, 13, 15, 16], dtype=np.int32),
        )

        aligned = _aligned_points(left, right)
        self.assertIsNotNone(aligned)
        object_points, left_points, right_points = aligned
        self.assertEqual(len(object_points), 6)
        np.testing.assert_array_equal(left_points, [[20, 10], [30, 10], [40, 10], [50, 10], [60, 10], [70, 10]])
        np.testing.assert_array_equal(right_points, [[40, 10], [50, 10], [60, 10], [70, 10], [80, 10], [90, 10]])

    def test_unmatched_views_are_rejected(self):
        left = PatternDetection(
            np.zeros((2, 3), np.float32), np.zeros((2, 2), np.float32), None, {},
            point_ids=np.asarray([1, 2], np.int32),
        )
        right = PatternDetection(
            np.zeros((2, 3), np.float32), np.zeros((2, 2), np.float32), None, {},
            point_ids=np.asarray([3, 4], np.int32),
        )
        self.assertIsNone(_aligned_points(left, right))


if __name__ == "__main__":
    unittest.main()
