import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from plenoxels.datasets.openseg_cache import OpenSegFeatureCache, map_pixels_to_feature_pixels


class OpenSegFeatureCacheTest(unittest.TestCase):
    def test_map_pixels_to_feature_pixels_clamps_bounds(self):
        x = torch.tensor([0, 675, 676])
        y = torch.tensor([0, 506, 507])

        feat_x, feat_y = map_pixels_to_feature_pixels(
            x=x,
            y=y,
            rgb_h=507,
            rgb_w=676,
            feat_h=126,
            feat_w=169,
        )

        self.assertEqual(feat_x.tolist(), [0, 168, 168])
        self.assertEqual(feat_y.tolist(), [0, 125, 125])

    def test_lookup_returns_batch_features_by_camera_and_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam00 = np.lib.format.open_memmap(
                root / "cam00.npy",
                mode="w+",
                dtype=np.float16,
                shape=(2, 2, 3, 4),
            )
            cam01 = np.lib.format.open_memmap(
                root / "cam01.npy",
                mode="w+",
                dtype=np.float16,
                shape=(2, 2, 3, 4),
            )
            cam00[:] = 1
            cam01[:] = 2
            cam00.flush()
            cam01.flush()
            del cam00
            del cam01

            cache = OpenSegFeatureCache(root, expected_feature_dim=4)
            out = cache.lookup(
                camera_names=["cam00", "cam01"],
                frame_ids=torch.tensor([0, 1]),
                x=torch.tensor([0, 2]),
                y=torch.tensor([0, 1]),
                rgb_h=2,
                rgb_w=3,
            )

        self.assertEqual(tuple(out.shape), (2, 4))
        np.testing.assert_array_equal(out[0].numpy(), np.ones(4, dtype=np.float16))
        np.testing.assert_array_equal(out[1].numpy(), np.full(4, 2, dtype=np.float16))

    def test_missing_camera_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = OpenSegFeatureCache(Path(tmp), expected_feature_dim=4)
            with self.assertRaisesRegex(FileNotFoundError, "cam99.npy"):
                cache.lookup(
                    camera_names=["cam99"],
                    frame_ids=torch.tensor([0]),
                    x=torch.tensor([0]),
                    y=torch.tensor([0]),
                    rgb_h=2,
                    rgb_w=3,
                )


if __name__ == "__main__":
    unittest.main()
