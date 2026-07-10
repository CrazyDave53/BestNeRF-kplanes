import unittest

import torch

from plenoxels.models.semantic_kplane_field import SemanticKPlaneField


def make_semantic_field(**kwargs):
    params = {
        "aabb": torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        "grid_config": [
            {
                "grid_dimensions": 2,
                "input_coordinate_dim": 4,
                "output_coordinate_dim": 4,
                "resolution": [4, 4, 4, 4],
            }
        ],
        "concat_features_across_scales": False,
        "multiscale_res": [1],
        "semantic_feature_dim": 8,
        "spatial_distortion": None,
        "linear_decoder_layers": 1,
    }
    params.update(kwargs)
    return SemanticKPlaneField(**params)


class SemanticBranchGradientTest(unittest.TestCase):
    def test_semantic_field_outputs_per_sample_features(self):
        field = make_semantic_field()
        pts = torch.zeros((2, 3, 3), dtype=torch.float32)
        timestamps = torch.zeros((2,), dtype=torch.float32)

        out = field(pts, timestamps=timestamps)

        self.assertEqual(tuple(out.shape), (2, 3, 8))

    def test_semantic_field_supports_openseg_sized_features(self):
        field = make_semantic_field(semantic_feature_dim=768)
        pts = torch.zeros((1, 1, 3), dtype=torch.float32)
        timestamps = torch.zeros((1,), dtype=torch.float32)

        out = field(pts, timestamps=timestamps)

        self.assertEqual(tuple(out.shape), (1, 1, 768))

    def test_semantic_field_accepts_string_grid_config(self):
        grid_config = str(
            [
                {
                    "grid_dimensions": 2,
                    "input_coordinate_dim": 4,
                    "output_coordinate_dim": 4,
                    "resolution": [4, 4, 4, 4],
                }
            ]
        )

        field = make_semantic_field(grid_config=grid_config)

        self.assertEqual(field.grid_config[0]["input_coordinate_dim"], 4)

    def test_get_params_covers_each_trainable_parameter_once(self):
        field = make_semantic_field()

        grouped_params = (
            field.get_params()["field"]
            + field.get_params()["nn"]
            + field.get_params()["other"]
        )
        trainable_grouped_params = [p for p in grouped_params if p.requires_grad]
        trainable_named_params = [p for p in field.parameters() if p.requires_grad]

        self.assertEqual(len(trainable_grouped_params), len(set(trainable_grouped_params)))
        self.assertEqual(set(trainable_grouped_params), set(trainable_named_params))

    def test_backward_reaches_semantic_field_params(self):
        field = make_semantic_field()
        pts = torch.zeros((2, 3, 3), dtype=torch.float32)
        timestamps = torch.zeros((2,), dtype=torch.float32)

        field(pts, timestamps=timestamps).sum().backward()

        trainable_params = [
            p
            for group in field.get_params().values()
            for p in group
            if p.requires_grad
        ]
        self.assertTrue(trainable_params)
        for param in trainable_params:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())


if __name__ == "__main__":
    unittest.main()
