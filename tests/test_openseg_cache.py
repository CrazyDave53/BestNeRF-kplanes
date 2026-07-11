import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from plenoxels.datasets.openseg_cache import OpenSegFeatureCache, map_pixels_to_feature_pixels


class OpenSegFeatureCacheTest(unittest.TestCase):
    def _write_shard(self, path, shape=(2, 2, 3, 4)):
        shard = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
        values = np.arange(np.prod(shape), dtype=np.float16).reshape(shape)
        shard[:] = values
        shard.flush()
        del shard
        return values

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
            cam00_values = self._write_shard(root / "cam00.npy")
            cam01_values = self._write_shard(root / "cam01.npy") + 100
            cam01 = np.lib.format.open_memmap(
                root / "cam01.npy", mode="r+", dtype=np.float16, shape=(2, 2, 3, 4)
            )
            cam01[:] = cam01_values
            cam01.flush()
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
        np.testing.assert_array_equal(out[0].numpy(), cam00_values[0, 0, 0])
        np.testing.assert_array_equal(out[1].numpy(), cam01_values[1, 1, 2])

    def test_lookup_uses_nested_openseg_camckpts_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_root = root / "openseg_camckpts"
            shard_root.mkdir()
            cam00_values = self._write_shard(shard_root / "cam00.npy")

            cache = OpenSegFeatureCache(root, expected_feature_dim=4)
            out = cache.lookup(
                camera_names=["cam00"],
                frame_ids=torch.tensor([1]),
                x=torch.tensor([2]),
                y=torch.tensor([1]),
                rgb_h=2,
                rgb_w=3,
            )

        np.testing.assert_array_equal(out[0].numpy(), cam00_values[1, 1, 2])

    def test_lookup_loads_repeated_camera_once_per_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_shard(root / "cam00.npy")
            cache = OpenSegFeatureCache(root, expected_feature_dim=4)
            original_load_shard = cache._load_shard
            load_calls = []

            def counted_load_shard(camera_name):
                load_calls.append(camera_name)
                return original_load_shard(camera_name)

            cache._load_shard = counted_load_shard

            cache.lookup(
                camera_names=["cam00", "cam00", "cam00"],
                frame_ids=torch.tensor([0, 0, 1]),
                x=torch.tensor([0, 1, 2]),
                y=torch.tensor([0, 1, 1]),
                rgb_h=2,
                rgb_w=3,
            )

        self.assertEqual(load_calls, ["cam00"])

    def test_lookup_reuses_open_mmap_across_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_shard(root / "cam00.npy")
            cache = OpenSegFeatureCache(root, expected_feature_dim=4)
            original_open_shard = cache._open_shard
            open_calls = []

            def counted_open_shard(camera_name):
                open_calls.append(camera_name)
                return original_open_shard(camera_name)

            cache._open_shard = counted_open_shard

            for _ in range(2):
                cache.lookup(
                    camera_names=["cam00", "cam00"],
                    frame_ids=torch.tensor([0, 1]),
                    x=torch.tensor([0, 2]),
                    y=torch.tensor([0, 1]),
                    rgb_h=2,
                    rgb_w=3,
                )
            cache.close()

        self.assertEqual(open_calls, ["cam00"])

    def test_negative_frame_id_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_shard(root / "cam00.npy")
            cache = OpenSegFeatureCache(root, expected_feature_dim=4)

            with self.assertRaisesRegex(IndexError, "cam00.*-1.*2"):
                cache.lookup(
                    camera_names=["cam00"],
                    frame_ids=torch.tensor([-1]),
                    x=torch.tensor([0]),
                    y=torch.tensor([0]),
                    rgb_h=2,
                    rgb_w=3,
                )

    def test_too_large_frame_id_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_shard(root / "cam00.npy")
            cache = OpenSegFeatureCache(root, expected_feature_dim=4)

            with self.assertRaisesRegex(IndexError, "cam00.*2.*2"):
                cache.lookup(
                    camera_names=["cam00"],
                    frame_ids=torch.tensor([2]),
                    x=torch.tensor([0]),
                    y=torch.tensor([0]),
                    rgb_h=2,
                    rgb_w=3,
                )

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
