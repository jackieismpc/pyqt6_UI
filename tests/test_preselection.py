"""首帧预选定位回归测试（不加载深度模型）。"""

import unittest

import numpy as np

from backend.crystalvol.config import LocalizeConfig
from backend.crystalvol.localize import _normalised_target_box, locate_crystal


class PreselectionLocalizationTests(unittest.TestCase):
    def test_normalised_box_is_scaled_to_current_image(self):
        self.assertEqual(
            _normalised_target_box((0.2, 0.3, 0.6, 0.8), 1000, 500),
            (200, 150, 600, 400),
        )

    def test_manual_anchor_rejects_distractor_outside_search_window(self):
        height = width = 200
        image = np.zeros((height, width, 3), np.uint8)
        # 真实目标在下方；右上角放置更亮、更容易触发全图显著性的干扰物。
        image[130:170, 80:120] = 210
        image[15:65, 150:195] = 255
        edges = np.zeros((height, width), np.uint8)
        edges[130:170, 80:120] = 255
        edges[15:65, 150:195] = 255

        config = LocalizeConfig(
            preselection_enabled=True,
            preselection_roi=(0.35, 0.60, 0.65, 0.90),
        )
        result = locate_crystal(image, edges, None, config)

        self.assertTrue(result.found)
        self.assertEqual(result.mode, "manual_anchor")
        self.assertEqual(result.component_bbox, (80, 130, 120, 170))
        self.assertEqual(result.bbox, (40, 90, 160, 200))
        self.assertNotIn("跳转", "".join(result.warnings))

    def test_manual_anchor_stays_local_when_no_candidate_exists(self):
        image = np.zeros((200, 200, 3), np.uint8)
        edges = np.zeros((200, 200), np.uint8)
        config = LocalizeConfig(
            preselection_enabled=True,
            preselection_roi=(0.35, 0.60, 0.65, 0.90),
        )
        result = locate_crystal(image, edges, None, config)

        self.assertFalse(result.found)
        self.assertEqual(result.mode, "manual_anchor")
        self.assertEqual(result.bbox, (40, 90, 160, 200))
        self.assertEqual(result.component_bbox, (70, 120, 130, 180))
        self.assertTrue(any("禁止跳转" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
