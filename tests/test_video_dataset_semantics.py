import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from plenoxels.datasets.intrinsics import Intrinsics
from plenoxels.datasets.video_datasets import Video360Dataset


class VideoDatasetSemanticsTest(unittest.TestCase):
    def _write_openseg_shard(self, root: Path):
        values = np.arange(2 * 4 * 4 * 4, dtype=np.float16).reshape(2, 4, 4, 4)
        shard = np.lib.format.open_memmap(
            root / "cam01.npy",
            mode="w+",
            dtype=np.float16,
            shape=values.shape,
        )
        shard[:] = values
        shard.flush()
        del shard
        return values

    def test_train_batch_includes_openseg_features_and_pixel_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_values = self._write_openseg_shard(root)
            intrinsics = Intrinsics(
                width=4,
                height=4,
                focal_x=4.0,
                focal_y=4.0,
                center_x=2.0,
                center_y=2.0,
            )
            cam_pose = torch.eye(4)[:3].unsqueeze(0)
            near_fars = torch.tensor([[0.1, 1.0]])
            video_paths = [str(root / "videos" / "cam01.mp4")]
            frame_poses = cam_pose.repeat(2, 1, 1)
            images = torch.arange(2 * 4 * 4 * 3, dtype=torch.uint8).reshape(2, 4, 4, 3)
            timestamps = torch.tensor([0.0, 1.0])
            median_images = images[:1]

            with mock.patch(
                "plenoxels.datasets.video_datasets.load_llffvideo_poses",
                return_value=(cam_pose, near_fars, intrinsics, video_paths),
            ), mock.patch(
                "plenoxels.datasets.video_datasets.load_llffvideo_data",
                return_value=(frame_poses, images, timestamps, median_images),
            ):
                dataset = Video360Dataset(
                    str(root),
                    split="train",
                    batch_size=3,
                    ndc=True,
                    openseg_cache_dir=str(root),
                    openseg_feature_dim=4,
                )

            dataset.perm = torch.tensor([0, 5, 31])
            batch = dataset[0]

        self.assertEqual(tuple(batch["openseg_features"].shape), (3, 4))
        self.assertEqual(batch["camera_ids"].tolist(), [0, 0, 0])
        self.assertEqual(batch["frame_ids"].tolist(), [0, 0, 1])
        self.assertEqual(batch["pixel_x"].tolist(), [0, 1, 3])
        self.assertEqual(batch["pixel_y"].tolist(), [0, 1, 3])
        np.testing.assert_array_equal(batch["openseg_features"][0].numpy(), cache_values[0, 0, 0])
        np.testing.assert_array_equal(batch["openseg_features"][1].numpy(), cache_values[0, 1, 1])
        np.testing.assert_array_equal(batch["openseg_features"][2].numpy(), cache_values[1, 3, 3])


if __name__ == "__main__":
    unittest.main()
