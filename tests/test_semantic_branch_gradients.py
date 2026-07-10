import unittest

import torch

from plenoxels.models.semantic_kplane_field import SemanticKPlaneField


class SemanticBranchGradientTest(unittest.TestCase):
    def test_semantic_field_outputs_per_sample_features(self):
        field = SemanticKPlaneField(
            aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
            grid_config=[
                {
                    "grid_dimensions": 2,
                    "input_coordinate_dim": 4,
                    "output_coordinate_dim": 4,
                    "resolution": [4, 4, 4, 4],
                }
            ],
            concat_features_across_scales=False,
            multiscale_res=[1],
            semantic_feature_dim=8,
            spatial_distortion=None,
            linear_decoder_layers=1,
        )
        pts = torch.zeros((2, 3, 3), dtype=torch.float32)
        timestamps = torch.zeros((2,), dtype=torch.float32)

        out = field(pts, timestamps=timestamps)

        self.assertEqual(tuple(out.shape), (2, 3, 8))


if __name__ == "__main__":
    unittest.main()
