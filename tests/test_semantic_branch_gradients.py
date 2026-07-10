import unittest
import tempfile

import torch

from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.models.semantic_kplane_field import SemanticKPlaneField
from plenoxels.runners import base_trainer
from plenoxels.runners.base_trainer import BaseTrainer
from plenoxels.utils.ema import EMA


class TinyFreezeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_param = torch.nn.Parameter(torch.tensor([1.0]))
        self.semantic_param = torch.nn.Parameter(torch.tensor([2.0]))
        self.freeze_called = False

    def get_params(self, lr):
        return [
            {"params": [self.rgb_param], "lr": lr},
            {"params": [self.semantic_param], "lr": lr},
        ]

    def freeze_rgb_parameters(self):
        self.freeze_called = True
        self.rgb_param.requires_grad = False

    def step_before_iter(self, _):
        pass

    def step_after_iter(self, _):
        pass


class TinyTrainer(BaseTrainer):
    def eval_step(self, data, **kwargs):
        return {}

    def validate(self):
        pass

    def init_epoch_info(self):
        return {"mse": EMA(), "psnr": EMA(), "semantic": EMA()}

    def init_model(self, **kwargs):
        return kwargs["model"]


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


def make_lowrank_model(**kwargs):
    grid_config = [
        {
            "grid_dimensions": 2,
            "input_coordinate_dim": 4,
            "output_coordinate_dim": 4,
            "resolution": [4, 4, 4, 4],
        }
    ]
    params = {
        "grid_config": grid_config,
        "is_ndc": False,
        "is_contracted": False,
        "aabb": torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        "multiscale_res": [1],
        "concat_features_across_scales": False,
        "linear_decoder": True,
        "linear_decoder_layers": 1,
        "num_proposal_iterations": 1,
        "use_same_proposal_network": False,
        "proposal_net_args_list": [
            {
                "resolution": [4, 4, 4, 4],
                "num_input_coords": 4,
                "num_output_coords": 4,
            }
        ],
        "num_proposal_samples": (2,),
        "num_samples": 2,
        "single_jitter": True,
        "semantic_enabled": True,
        "semantic_feature_dim": 8,
        "semantic_detach_geometry": True,
        "semantic_linear_decoder_layers": 1,
    }
    params.update(kwargs)
    return LowrankModel(**params)


def make_rays():
    rays_o = torch.tensor([[0.0, 0.0, -0.5], [0.1, 0.0, -0.5]])
    rays_d = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.1, 1.0]])
    bg_color = torch.zeros((2, 3))
    near_far = torch.tensor([[0.1, 0.9], [0.1, 0.9]])
    timestamps = torch.zeros((2,), dtype=torch.float32)
    return rays_o, rays_d, bg_color, near_far, timestamps


class SemanticBranchGradientTest(unittest.TestCase):
    def test_cosine_semantic_loss_is_zero_for_matching_features(self):
        self.assertTrue(hasattr(base_trainer, "semantic_cosine_loss"))
        preds = torch.tensor([[2.0, 0.0, 0.0], [0.0, -3.0, 0.0]])
        targets = torch.tensor([[7.0, 0.0, 0.0], [0.0, -0.5, 0.0]])

        loss = base_trainer.semantic_cosine_loss(preds, targets)

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0), atol=1e-6))

    def test_optimizer_excludes_frozen_rgb_params_after_trainer_freeze(self):
        model = TinyFreezeModel()

        with tempfile.TemporaryDirectory() as logdir:
            trainer = TinyTrainer(
                train_data_loader=[],
                num_steps=1,
                logdir=logdir,
                expname="semantic-freeze-test",
                train_fp16=False,
                save_every=-1,
                valid_every=-1,
                save_outputs=False,
                device="cpu",
                model=model,
                optim_type="adam",
                lr=1e-3,
                scheduler_type="cosine",
                freeze_rgb_for_semantic=True,
            )

        optimized_params = {
            param
            for group in trainer.optimizer.param_groups
            for param in group["params"]
        }
        self.assertTrue(model.freeze_called)
        self.assertFalse(model.rgb_param.requires_grad)
        self.assertNotIn(model.rgb_param, optimized_params)
        self.assertIn(model.semantic_param, optimized_params)

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

    def test_lowrank_semantic_loss_does_not_grad_rgb_field(self):
        model = make_lowrank_model()
        model.train()

        out = model(*make_rays())
        out["semantic_features"].sum().backward()

        rgb_params = [p for p in model.field.parameters() if p.requires_grad]
        semantic_params = [
            p for p in model.semantic_field.parameters() if p.requires_grad
        ]
        self.assertTrue(rgb_params)
        self.assertTrue(semantic_params)
        for param in rgb_params:
            self.assertIsNone(param.grad)
        self.assertTrue(any(param.grad is not None for param in semantic_params))
        for param in semantic_params:
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_freeze_rgb_parameters_keeps_semantic_trainable(self):
        model = make_lowrank_model()

        model.freeze_rgb_parameters()

        self.assertTrue(list(model.field.parameters()))
        self.assertTrue(list(model.proposal_networks.parameters()))
        self.assertTrue(list(model.semantic_field.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.field.parameters()))
        self.assertFalse(
            any(p.requires_grad for p in model.proposal_networks.parameters())
        )
        self.assertTrue(any(p.requires_grad for p in model.semantic_field.parameters()))


if __name__ == "__main__":
    unittest.main()
