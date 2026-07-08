import tempfile
import unittest
from pathlib import Path

from scripts.extract_openseg_neu3d import compute_feature_shape, find_camera_videos


class ExtractOpenSegNeu3DTest(unittest.TestCase):
    def test_find_camera_videos_returns_sorted_cam_mp4_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["cam10.mp4", "notes.txt", "cam02.mp4", "cam01.mp4"]:
                (root / name).write_text("")

            videos = find_camera_videos(root)

        self.assertEqual([p.name for p in videos], ["cam01.mp4", "cam02.mp4", "cam10.mp4"])

    def test_compute_feature_shape_uses_integer_downsample(self):
        self.assertEqual(compute_feature_shape(1014, 1352, 4), (253, 338))


if __name__ == "__main__":
    unittest.main()

