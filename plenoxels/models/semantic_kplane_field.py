import ast
import logging as log
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import tinycudann as tcnn

from plenoxels.models.kplane_field import (
    init_grid_param,
    interpolate_ms_features,
    normalize_aabb,
)
from plenoxels.raymarching.spatial_distortions import SpatialDistortion


class SemanticKPlaneField(nn.Module):
    def __init__(
        self,
        aabb: torch.Tensor,
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        multiscale_res: Optional[Sequence[int]],
        semantic_feature_dim: int,
        spatial_distortion: Optional[SpatialDistortion],
        linear_decoder_layers: Optional[int],
    ) -> None:
        super().__init__()

        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        if isinstance(grid_config, str):
            self.grid_config: List[Dict] = ast.literal_eval(grid_config)
        else:
            self.grid_config: List[Dict] = grid_config
        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]
        self.concat_features = concat_features_across_scales
        self.semantic_feature_dim = semantic_feature_dim

        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in self.multiscale_res_multipliers:
            config = self.grid_config[0].copy()
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)
        log.info(f"Initialized semantic model grids: {self.grids}")

        assert linear_decoder_layers is not None
        self.semantic_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=self.semantic_feature_dim,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": linear_decoder_layers,
            },
        )

    def forward(
        self,
        pts: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2
        else:
            pts = normalize_aabb(pts, self.aabb)

        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]
            pts = torch.cat((pts, timestamps), dim=-1)

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts,
            ms_grids=self.grids,
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features,
            num_levels=None,
        )
        semantic_features = self.semantic_net(features).to(pts)
        return semantic_features.view(n_rays, n_samples, self.semantic_feature_dim)

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = {
            k: v for k, v in self.semantic_net.named_parameters(prefix="semantic_net")
        }
        other_params = {
            k: v
            for k, v in self.named_parameters()
            if k not in nn_params.keys() and k not in field_params.keys()
        }
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }
