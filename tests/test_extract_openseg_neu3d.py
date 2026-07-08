import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.extract_openseg_neu3d import (
    compute_feature_shape,
    find_camera_videos,
    save_feature_shard,
)


class ExtractOpenSegNeu3DTest(unittest.TestCase):
    def test_find_camera_videos_returns_sorted_cam_mp4_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["cam10.mp4", "notes.txt", "cam02.mp4", "cam01.mp4"]:
                (root / name).write_text("")

            videos = find_camera_videos(root)

        self.assertEqual([p.name for p in videos], ["cam01.mp4", "cam02.mp4", "cam10.mp4"])

    def test_compute_feature_shape_uses_integer_downsample(self):
        self.assertEqual(compute_feature_shape(2028, 2704, 8), (253, 338))

    def test_save_feature_shard_streams_and_trims_written_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "cam00.npy"

            def frames():
                for value in [1, 2]:
                    yield np.full((2, 3, 4), value, dtype=np.float16)

            info = save_feature_shard(
                output_path,
                frames(),
                max_frames=5,
                feature_shape=(2, 3, 4),
                chunk_frames=1,
            )

            saved = np.load(output_path, mmap_mode="r")
            saved_shape = saved.shape
            saved_first = np.array(saved[0])
            saved_second = np.array(saved[1])
            del saved

        self.assertEqual(info["shape"], [2, 2, 3, 4])
        self.assertEqual(saved_shape, (2, 2, 3, 4))
        np.testing.assert_array_equal(saved_first, np.full((2, 3, 4), 1, dtype=np.float16))
        np.testing.assert_array_equal(saved_second, np.full((2, 3, 4), 2, dtype=np.float16))


if __name__ == "__main__":
    unittest.main()
