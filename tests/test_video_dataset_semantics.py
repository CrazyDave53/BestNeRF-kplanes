import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from plenoxels.datasets.intrinsics import Intrinsics
from plenoxels.datasets.video_datasets import Video360Dataset


class VideoDatasetSemanticsTest(unittest.TestCase):
    def _intrinsics(self):
        return Intrinsics(
            width=4,
            height=4,
            focal_x=4.0,
            focal_y=4.0,
            center_x=2.0,
            center_y=2.0,
        )

    def _write_openseg_shard(self, root: Path, camera_name: str, camera_offset: int):
        values = np.zeros((31, 2, 2, 4), dtype=np.float16)
        for frame_id in (0, 30):
            for y in range(2):
                for x in range(2):
                    base = camera_offset + frame_id * 100 + y * 10 + x
                    values[frame_id, y, x] = np.array(
                        [base, base + 1, base + 2, base + 3], dtype=np.float16
                    )
        shard = np.lib.format.open_memmap(
            root / f"{camera_name}.npy",
            mode="w+",
            dtype=np.float16,
            shape=values.shape,
        )
        shard[:] = values
        shard.flush()
        del shard
        return values

    def _mock_llff_dataset(
        self,
        root: Path,
        split: str,
        num_cameras: int = 2,
        raw_frame_ids: tuple[int, ...] = (0, 30),
        openseg_cache_dir: str | None = None,
    ):
        intrinsics = self._intrinsics()
        cam_poses = torch.eye(4)[:3].unsqueeze(0).repeat(num_cameras, 1, 1)
        near_fars = torch.tensor([[0.1, 1.0]] * num_cameras)
        video_paths = [str(root / "videos" / f"cam{i:02d}.mp4") for i in range(num_cameras)]
        images = torch.arange(
            num_cameras * len(raw_frame_ids) * 4 * 4 * 3,
            dtype=torch.uint8,
        ).reshape(num_cameras * len(raw_frame_ids), 4, 4, 3)
        frame_poses = cam_poses.repeat_interleave(len(raw_frame_ids), dim=0)
        timestamps = torch.tensor(list(raw_frame_ids) * num_cameras)
        median_images = images[::len(raw_frame_ids)]

        with mock.patch(
            "plenoxels.datasets.video_datasets.load_llffvideo_poses",
            return_value=(cam_poses, near_fars, intrinsics, video_paths),
        ), mock.patch(
            "plenoxels.datasets.video_datasets.load_llffvideo_data",
            return_value=(frame_poses, images, timestamps, median_images),
        ):
            return Video360Dataset(
                str(root),
                split=split,
                batch_size=4 if split == "train" else None,
                ndc=True,
                keyframes=True,
                openseg_cache_dir=openseg_cache_dir,
                openseg_feature_dim=4,
            )

    def test_train_batch_uses_camera_shards_raw_frame_ids_and_downsampled_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam00_values = self._write_openseg_shard(root, "cam00", camera_offset=0)
            cam01_values = self._write_openseg_shard(root, "cam01", camera_offset=10000)
            dataset = self._mock_llff_dataset(
                root,
                split="train",
                openseg_cache_dir=str(root),
            )

            dataset.perm = torch.tensor([0, 31, 32, 63])
            batch = dataset[0]

        self.assertEqual(tuple(batch["openseg_features"].shape), (4, 4))
        self.assertEqual(batch["camera_ids"].tolist(), [0, 0, 1, 1])
        self.assertEqual(batch["frame_ids"].tolist(), [0, 30, 0, 30])
        self.assertEqual(batch["pixel_x"].tolist(), [0, 3, 0, 3])
        self.assertEqual(batch["pixel_y"].tolist(), [0, 3, 0, 3])
        np.testing.assert_array_equal(batch["openseg_features"][0].numpy(), cam00_values[0, 0, 0])
        np.testing.assert_array_equal(batch["openseg_features"][1].numpy(), cam00_values[30, 1, 1])
        np.testing.assert_array_equal(batch["openseg_features"][2].numpy(), cam01_values[0, 0, 0])
        np.testing.assert_array_equal(batch["openseg_features"][3].numpy(), cam01_values[30, 1, 1])

    def test_train_batch_without_cache_keeps_rgb_only_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._mock_llff_dataset(Path(tmp), split="train")
            dataset.perm = torch.tensor([0, 1, 2, 3])

            batch = dataset[0]

        self.assertNotIn("openseg_features", batch)
        self.assertEqual(tuple(batch["imgs"].shape), (4, 3))
        self.assertEqual(batch["camera_ids"].tolist(), [0, 0, 0, 0])
        self.assertEqual(batch["pixel_x"].tolist(), [0, 1, 2, 3])

    def test_test_split_does_not_attach_openseg_features_when_cache_is_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_openseg_shard(root, "cam00", camera_offset=0)
            dataset = self._mock_llff_dataset(
                root,
                split="test",
                num_cameras=1,
                raw_frame_ids=(0,),
                openseg_cache_dir=str(root),
            )

            batch = dataset[0]

        self.assertNotIn("openseg_features", batch)
        self.assertEqual(tuple(batch["imgs"].shape), (16, 3))


if __name__ == "__main__":
    unittest.main()
