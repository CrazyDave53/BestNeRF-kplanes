import unittest

import numpy as np

from scripts.render_semantic_query_neu3d import (
    colorize_heatmap,
    normalize_scores,
    overlay_heatmap,
)


class RenderSemanticQueryNeu3DTest(unittest.TestCase):
    def test_normalize_scores_maps_range_to_unit_interval(self):
        scores = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

        normalized = normalize_scores(scores)

        np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0], atol=1e-6)

    def test_normalize_scores_handles_constant_input(self):
        scores = np.full((2, 3), 0.7, dtype=np.float32)

        normalized = normalize_scores(scores)

        np.testing.assert_array_equal(normalized, np.zeros_like(scores))

    def test_colorize_heatmap_returns_uint8_rgb(self):
        heatmap = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

        color = colorize_heatmap(heatmap)

        self.assertEqual(color.shape, (1, 3, 3))
        self.assertEqual(color.dtype, np.uint8)
        np.testing.assert_array_equal(color[0, 0], [0, 0, 255])
        np.testing.assert_array_equal(color[0, 2], [255, 0, 0])

    def test_overlay_heatmap_blends_rgb_and_heatmap(self):
        rgb = np.full((1, 1, 3), 100, dtype=np.uint8)
        heatmap_color = np.array([[[200, 0, 0]]], dtype=np.uint8)

        overlay = overlay_heatmap(rgb, heatmap_color, alpha=0.25)

        self.assertEqual(overlay.dtype, np.uint8)
        np.testing.assert_array_equal(overlay[0, 0], [125, 75, 75])


if __name__ == "__main__":
    unittest.main()
