# Semantic Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add detached OpenSeg semantic supervision to K-Planes video training while preserving RGB-only behavior and preventing semantic gradients from updating RGB geometry.

**Architecture:** Add an optional semantic branch inside `LowrankModel`, backed by a separate semantic K-Plane field. `Video360Dataset` supplies memory-mapped OpenSeg feature targets by camera/frame/pixel, and `BaseTrainer` adds cosine semantic loss only when semantic targets are present.

**Tech Stack:** Python 3.12, PyTorch, tiny-cuda-nn, NumPy memmap `.npy`, unittest, existing K-Planes trainer/config structure.

---

## File Structure

- Create `plenoxels/datasets/openseg_cache.py`: OpenSeg shard discovery, mmap loading, pixel mapping, and batch lookup.
- Modify `plenoxels/datasets/video_datasets.py`: preserve selected `camXX` names, compute ray metadata, and attach semantic targets when configured.
- Modify `plenoxels/runners/video_trainer.py`: pass semantic cache config into the train dataset.
- Create `plenoxels/models/semantic_kplane_field.py`: semantic K-Plane grids and decoder returning `[n_rays, n_samples, semantic_feature_dim]`.
- Modify `plenoxels/models/lowrank_model.py`: instantiate optional semantic branch, render semantic features using detached final RGB weights and detached positions, and expose semantic params separately.
- Modify `plenoxels/runners/base_trainer.py`: move semantic targets to device, add semantic cosine loss, respect `train_rgb` / `train_semantic`, freeze RGB when requested, and log semantic metrics.
- Create `plenoxels/configs/local/dynerf_cm_semantic_smoke.py`: 20-step semantic smoke run.
- Create `plenoxels/configs/local/dynerf_cm_semantic_64f_ds16.py`: first real 64-frame semantic run.
- Create tests:
  - `tests/test_openseg_cache.py`
  - `tests/test_video_dataset_semantics.py`
  - `tests/test_semantic_branch_gradients.py`
- Modify `docs/kaggle_runbook.md`: add semantic smoke/full commands after implementation works.

---

## Task 1: OpenSeg Cache Loader

**Files:**
- Create: `tests/test_openseg_cache.py`
- Create: `plenoxels/datasets/openseg_cache.py`

- [ ] **Step 1: Write failing cache tests**

Create `tests/test_openseg_cache.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m unittest tests.test_openseg_cache -v
```

Expected: fail with `ModuleNotFoundError: No module named 'plenoxels.datasets.openseg_cache'`.

- [ ] **Step 3: Implement cache loader**

Create `plenoxels/datasets/openseg_cache.py`:

```python
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


def map_pixels_to_feature_pixels(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    rgb_h: int,
    rgb_w: int,
    feat_h: int,
    feat_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    feat_x = torch.floor(x.to(torch.float32) / float(rgb_w) * float(feat_w)).to(torch.long)
    feat_y = torch.floor(y.to(torch.float32) / float(rgb_h) * float(feat_h)).to(torch.long)
    feat_x = torch.clamp(feat_x, 0, feat_w - 1)
    feat_y = torch.clamp(feat_y, 0, feat_h - 1)
    return feat_x, feat_y


class OpenSegFeatureCache:
    def __init__(self, root: str | Path, expected_feature_dim: int = 768):
        self.root = Path(root)
        self.shard_dir = self.root / "openseg_camckpts" if (self.root / "openseg_camckpts").is_dir() else self.root
        self.expected_feature_dim = expected_feature_dim
        self._shards: dict[str, np.ndarray] = {}

    def _path_for_camera(self, camera_name: str) -> Path:
        camera_name = Path(camera_name).stem
        return self.shard_dir / f"{camera_name}.npy"

    def _load_shard(self, camera_name: str) -> np.ndarray:
        camera_name = Path(camera_name).stem
        if camera_name not in self._shards:
            path = self._path_for_camera(camera_name)
            if not path.is_file():
                raise FileNotFoundError(f"OpenSeg shard not found: {path.name} in {self.shard_dir}")
            shard = np.load(path, mmap_mode="r")
            if shard.ndim != 4:
                raise ValueError(f"Expected 4D OpenSeg shard for {camera_name}, got shape {shard.shape}")
            if shard.shape[-1] != self.expected_feature_dim:
                raise ValueError(
                    f"Expected feature dim {self.expected_feature_dim} for {camera_name}, got {shard.shape[-1]}"
                )
            self._shards[camera_name] = shard
        return self._shards[camera_name]

    def lookup(
        self,
        *,
        camera_names: Sequence[str],
        frame_ids: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        rgb_h: int,
        rgb_w: int,
    ) -> torch.Tensor:
        if len(camera_names) != int(frame_ids.numel()):
            raise ValueError(f"camera_names length {len(camera_names)} does not match batch {frame_ids.numel()}")

        features = []
        for camera_name in sorted(set(camera_names)):
            shard = self._load_shard(camera_name)
            break
        first_shard = self._load_shard(camera_names[0])
        feat_h, feat_w = first_shard.shape[1:3]
        feat_x, feat_y = map_pixels_to_feature_pixels(x=x, y=y, rgb_h=rgb_h, rgb_w=rgb_w, feat_h=feat_h, feat_w=feat_w)

        for i, camera_name in enumerate(camera_names):
            shard = self._load_shard(camera_name)
            frame_id = int(frame_ids[i].item())
            if frame_id < 0 or frame_id >= shard.shape[0]:
                raise IndexError(f"Frame {frame_id} is out of range for {camera_name}; shard has {shard.shape[0]} frames")
            features.append(shard[frame_id, int(feat_y[i]), int(feat_x[i])])

        return torch.from_numpy(np.stack(features, axis=0))
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
python -m unittest tests.test_openseg_cache -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add plenoxels/datasets/openseg_cache.py tests/test_openseg_cache.py
git commit -m "feat: add openseg feature cache"
```

---

## Task 2: Dataset Semantic Targets

**Files:**
- Modify: `plenoxels/datasets/video_datasets.py`
- Modify: `plenoxels/runners/video_trainer.py`
- Create: `tests/test_video_dataset_semantics.py`

- [ ] **Step 1: Write failing dataset-focused tests**

Create `tests/test_video_dataset_semantics.py`:

```python
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from plenoxels.datasets.intrinsics import Intrinsics
from plenoxels.datasets.video_datasets import Video360Dataset


class VideoDatasetSemanticsTest(unittest.TestCase):
    def test_train_batch_includes_openseg_features_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            cache_root.mkdir()
            shard = np.lib.format.open_memmap(
                cache_root / "cam01.npy",
                mode="w+",
                dtype=np.float16,
                shape=(2, 2, 2, 4),
            )
            shard[0] = 3
            shard[1] = 5
            shard.flush()
            del shard

            imgs = torch.zeros((2, 4, 4, 3), dtype=torch.uint8)
            poses = torch.eye(4)[:3].repeat(2, 1, 1)
            timestamps = torch.tensor([0, 1], dtype=torch.int32)
            median = torch.zeros((1, 4, 4, 3), dtype=torch.uint8)

            with mock.patch(
                "plenoxels.datasets.video_datasets.load_llffvideo_poses",
                return_value=(
                    torch.eye(4)[:3].unsqueeze(0),
                    torch.tensor([[0.0, 2.6]]),
                    Intrinsics(width=4, height=4, focal_x=1.0, focal_y=1.0, center_x=2.0, center_y=2.0),
                    ["cam01.mp4"],
                ),
            ), mock.patch(
                "plenoxels.datasets.video_datasets.load_llffvideo_data",
                return_value=(poses, imgs, timestamps, median),
            ):
                dset = Video360Dataset(
                    "data/neu3d/coffee_martini",
                    split="train",
                    batch_size=2,
                    downsample=4,
                    max_tsteps=2,
                    contraction=False,
                    ndc=True,
                    scene_bbox=[[-1, -1, -1], [1, 1, 1]],
                    openseg_cache_dir=str(cache_root),
                    openseg_feature_dim=4,
                )
                batch = dset[0]

        self.assertIn("openseg_features", batch)
        self.assertEqual(tuple(batch["openseg_features"].shape), (2, 4))
        self.assertIn("camera_ids", batch)
        self.assertIn("frame_ids", batch)
        self.assertIn("pixel_x", batch)
        self.assertIn("pixel_y", batch)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m unittest tests.test_video_dataset_semantics -v
```

Expected: fail because `Video360Dataset.__init__()` does not accept `openseg_cache_dir`.

- [ ] **Step 3: Add dataset constructor fields**

In `plenoxels/datasets/video_datasets.py`, add import:

```python
from pathlib import Path
from .openseg_cache import OpenSegFeatureCache
```

Extend `Video360Dataset.__init__` arguments:

```python
                 ndc_far: float = 2.6,
                 openseg_cache_dir: Optional[str] = None,
                 openseg_feature_dim: int = 768):
```

After `load_llffvideo_poses(...)` returns `videopaths`, store camera names:

```python
                self.camera_names = [Path(path).stem for path in videopaths]
```

After `load_llffvideo_data(...)`, store frames per camera:

```python
                self.num_frames_per_camera = imgs.shape[0] // max(1, len(self.camera_names))
```

After `super().__init__(...)`, create cache:

```python
        self.openseg_cache = None
        if openseg_cache_dir is not None and split == "train":
            self.openseg_cache = OpenSegFeatureCache(openseg_cache_dir, expected_feature_dim=openseg_feature_dim)
```

For non-LLFF branches, set safe defaults before branching:

```python
        self.camera_names = []
        self.num_frames_per_camera = None
```

- [ ] **Step 4: Add ray metadata and semantic lookup**

In `Video360Dataset.__getitem__`, preserve integer pixel coordinates before adding `0.5`:

```python
            pixel_x = x.clone()
            pixel_y = y.clone()
            x, y = x + 0.5, y + 0.5
```

After train `camera_id` is computed, compute frame ids and camera names:

```python
            frame_id = torch.remainder(image_id, num_frames_per_camera)
            out["camera_ids"] = camera_id
            out["frame_ids"] = frame_id
            out["pixel_x"] = pixel_x.to(torch.long)
            out["pixel_y"] = pixel_y.to(torch.long)
            if self.openseg_cache is not None:
                camera_names = [self.camera_names[int(i)] for i in camera_id.tolist()]
                out["openseg_features"] = self.openseg_cache.lookup(
                    camera_names=camera_names,
                    frame_ids=frame_id.to(torch.long),
                    x=pixel_x.to(torch.long),
                    y=pixel_y.to(torch.long),
                    rgb_h=h,
                    rgb_w=w,
                )
```

In the non-train branch, define metadata variables only if needed and do not attach OpenSeg targets.

- [ ] **Step 5: Pass semantic config from video trainer**

In `plenoxels/runners/video_trainer.py`, pass semantic cache options into train dataset:

```python
        openseg_cache_dir=kwargs.get("openseg_cache_dir", None),
        openseg_feature_dim=kwargs.get("semantic_feature_dim", 768),
```

Only add this to `init_tr_data`; do not add it to test data yet.

- [ ] **Step 6: Run tests**

Run:

```powershell
python -m unittest tests.test_openseg_cache tests.test_video_dataset_semantics -v
```

Expected: both test modules pass.

- [ ] **Step 7: Commit**

```powershell
git add plenoxels/datasets/video_datasets.py plenoxels/runners/video_trainer.py tests/test_video_dataset_semantics.py
git commit -m "feat: attach openseg targets to video batches"
```

---

## Task 3: Semantic K-Plane Field

**Files:**
- Create: `plenoxels/models/semantic_kplane_field.py`
- Create or extend: `tests/test_semantic_branch_gradients.py`

- [ ] **Step 1: Write failing semantic field shape test**

Create `tests/test_semantic_branch_gradients.py` with the first test:

```python
import unittest

import torch

from plenoxels.models.semantic_kplane_field import SemanticKPlaneField


class SemanticBranchGradientTest(unittest.TestCase):
    def test_semantic_field_outputs_per_sample_features(self):
        field = SemanticKPlaneField(
            aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
            grid_config=[{
                "grid_dimensions": 2,
                "input_coordinate_dim": 4,
                "output_coordinate_dim": 4,
                "resolution": [4, 4, 4, 4],
            }],
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
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m unittest tests.test_semantic_branch_gradients -v
```

Expected: fail with `ModuleNotFoundError` for `semantic_kplane_field`.

- [ ] **Step 3: Implement semantic field**

Create `plenoxels/models/semantic_kplane_field.py`:

```python
from typing import Dict, List, Optional, Sequence, Union

import logging as log
import torch
import torch.nn as nn
import tinycudann as tcnn

from plenoxels.models.kplane_field import init_grid_param, interpolate_ms_features, normalize_aabb
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
        linear_decoder_layers: int = 1,
    ) -> None:
        super().__init__()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = eval(grid_config) if isinstance(grid_config, str) else grid_config
        self.concat_features = concat_features_across_scales
        self.multiscale_res_multipliers = multiscale_res or [1]
        self.semantic_feature_dim = semantic_feature_dim

        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in self.multiscale_res_multipliers:
            config = self.grid_config[0].copy()
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
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
        log.info(f"Initialized semantic K-Plane grids: {self.grids}")

        self.semantic_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=self.semantic_feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": linear_decoder_layers,
            },
        )

    def forward(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        semantic = self.semantic_net(features).to(pts)
        return semantic.view(n_rays, n_samples, self.semantic_feature_dim)

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="semantic_grids")}
        nn_params = {k: v for k, v in self.semantic_net.named_parameters(prefix="semantic_net")}
        other_params = {
            k: v for k, v in self.named_parameters()
            if k not in field_params.keys() and k not in nn_params.keys()
        }
        return {
            "field": list(field_params.values()),
            "nn": list(nn_params.values()),
            "other": list(other_params.values()),
        }
```

- [ ] **Step 4: Run semantic field test**

Run:

```powershell
python -m unittest tests.test_semantic_branch_gradients -v
```

Expected: semantic field shape test passes. If local CPU lacks tiny-cuda-nn support, run on Kaggle after Task 4; do not replace the semantic field with a CPU-only implementation.

- [ ] **Step 5: Commit**

```powershell
git add plenoxels/models/semantic_kplane_field.py tests/test_semantic_branch_gradients.py
git commit -m "feat: add semantic kplane field"
```

---

## Task 4: LowrankModel Semantic Forward

**Files:**
- Modify: `plenoxels/models/lowrank_model.py`
- Extend: `tests/test_semantic_branch_gradients.py`

- [ ] **Step 1: Add failing model isolation test**

Append this test to `SemanticBranchGradientTest`:

```python
    def test_lowrank_semantic_loss_does_not_grad_rgb_field(self):
        from plenoxels.models.lowrank_model import LowrankModel

        model = LowrankModel(
            grid_config=[{
                "grid_dimensions": 2,
                "input_coordinate_dim": 4,
                "output_coordinate_dim": 4,
                "resolution": [4, 4, 4, 4],
            }],
            is_ndc=True,
            is_contracted=False,
            aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
            multiscale_res=[1],
            concat_features_across_scales=False,
            linear_decoder=True,
            linear_decoder_layers=1,
            num_proposal_iterations=1,
            proposal_net_args_list=[{
                "num_input_coords": 4,
                "num_output_coords": 4,
                "resolution": [4, 4, 4, 4],
            }],
            num_proposal_samples=[4],
            num_samples=4,
            semantic_enabled=True,
            semantic_feature_dim=8,
            semantic_detach_geometry=True,
            semantic_grid_config=[{
                "grid_dimensions": 2,
                "input_coordinate_dim": 4,
                "output_coordinate_dim": 4,
                "resolution": [4, 4, 4, 4],
            }],
            semantic_multiscale_res=[1],
        )
        model.train()
        rays_o = torch.zeros((2, 3))
        rays_d = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        bg_color = torch.ones((1, 3))
        near_far = torch.tensor([[0.1, 1.0], [0.1, 1.0]])
        timestamps = torch.zeros((2,))

        out = model(rays_o, rays_d, bg_color=bg_color, near_far=near_far, timestamps=timestamps)
        loss = out["semantic_features"].sum()
        loss.backward()

        rgb_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.field.parameters())
        semantic_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.semantic_field.parameters())

        self.assertFalse(rgb_has_grad)
        self.assertTrue(semantic_has_grad)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m unittest tests.test_semantic_branch_gradients -v
```

Expected: fail because `LowrankModel` does not accept semantic config or return `semantic_features`.

- [ ] **Step 3: Add semantic config to LowrankModel**

In `plenoxels/models/lowrank_model.py`, import:

```python
from plenoxels.models.semantic_kplane_field import SemanticKPlaneField
```

Add constructor arguments:

```python
                 semantic_enabled: bool = False,
                 semantic_feature_dim: int = 768,
                 semantic_detach_geometry: bool = True,
                 semantic_grid_config: Optional[Union[str, List[Dict]]] = None,
                 semantic_multiscale_res: Optional[Sequence[int]] = None,
                 semantic_linear_decoder_layers: int = 1,
```

After proposal sampler setup, instantiate:

```python
        self.semantic_enabled = semantic_enabled
        self.semantic_feature_dim = semantic_feature_dim
        self.semantic_detach_geometry = semantic_detach_geometry
        self.semantic_field = None
        if self.semantic_enabled:
            self.semantic_field = SemanticKPlaneField(
                aabb=aabb,
                grid_config=semantic_grid_config or self.config,
                concat_features_across_scales=self.concat_features_across_scales,
                multiscale_res=semantic_multiscale_res or self.multiscale_res,
                semantic_feature_dim=self.semantic_feature_dim,
                spatial_distortion=self.spatial_distortion,
                linear_decoder_layers=semantic_linear_decoder_layers,
            )
```

- [ ] **Step 4: Render semantic features in forward**

Add static renderer:

```python
    @staticmethod
    def render_features(features: torch.Tensor, weights: torch.Tensor):
        return torch.sum(weights * features, dim=-2)
```

In `forward`, after `outputs` is created and before eval-only storage:

```python
        if self.semantic_enabled and self.semantic_field is not None and self.training:
            semantic_positions = ray_samples.get_positions()
            semantic_weights = weights
            semantic_timestamps = timestamps
            if self.semantic_detach_geometry:
                semantic_positions = semantic_positions.detach()
                semantic_weights = semantic_weights.detach()
                if semantic_timestamps is not None:
                    semantic_timestamps = semantic_timestamps.detach()
            semantic_per_sample = self.semantic_field(semantic_positions, timestamps=semantic_timestamps)
            outputs["semantic_features"] = self.render_features(
                features=semantic_per_sample,
                weights=semantic_weights,
            )
```

- [ ] **Step 5: Add semantic params to get_params**

At the end of `get_params`, append semantic parameter groups:

```python
        params = [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
        ]
        if self.semantic_field is not None:
            semantic_params = self.semantic_field.get_params()
            params.extend([
                {"params": semantic_params["field"], "lr": lr},
                {"params": semantic_params["nn"], "lr": lr},
                {"params": semantic_params["other"], "lr": lr},
            ])
        return params
```

Replace the existing direct return with this `params` return.

- [ ] **Step 6: Add RGB freeze helper**

Add this method to `LowrankModel`:

```python
    def freeze_rgb_parameters(self):
        for param in self.field.parameters():
            param.requires_grad_(False)
        for network in self.proposal_networks:
            for param in network.parameters():
                param.requires_grad_(False)
```

This intentionally does not freeze `self.semantic_field`.

- [ ] **Step 7: Add freeze behavior test**

Append this test to `SemanticBranchGradientTest`:

```python
    def test_freeze_rgb_parameters_keeps_semantic_trainable(self):
        from plenoxels.models.lowrank_model import LowrankModel

        model = LowrankModel(
            grid_config=[{
                "grid_dimensions": 2,
                "input_coordinate_dim": 4,
                "output_coordinate_dim": 4,
                "resolution": [4, 4, 4, 4],
            }],
            is_ndc=True,
            is_contracted=False,
            aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
            multiscale_res=[1],
            concat_features_across_scales=False,
            linear_decoder=True,
            linear_decoder_layers=1,
            num_proposal_iterations=1,
            proposal_net_args_list=[{
                "num_input_coords": 4,
                "num_output_coords": 4,
                "resolution": [4, 4, 4, 4],
            }],
            num_proposal_samples=[4],
            num_samples=4,
            semantic_enabled=True,
            semantic_feature_dim=8,
            semantic_grid_config=[{
                "grid_dimensions": 2,
                "input_coordinate_dim": 4,
                "output_coordinate_dim": 4,
                "resolution": [4, 4, 4, 4],
            }],
            semantic_multiscale_res=[1],
        )

        model.freeze_rgb_parameters()

        self.assertFalse(any(p.requires_grad for p in model.field.parameters()))
        self.assertFalse(any(p.requires_grad for n in model.proposal_networks for p in n.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.semantic_field.parameters()))
```

- [ ] **Step 8: Run tests**

Run:

```powershell
python -m unittest tests.test_semantic_branch_gradients -v
```

Expected: tests pass. If this test is too slow locally due tiny-cuda-nn, run it on Kaggle after pulling the branch.

- [ ] **Step 9: Commit**

```powershell
git add plenoxels/models/lowrank_model.py tests/test_semantic_branch_gradients.py
git commit -m "feat: render detached semantic features"
```

---

## Task 5: Trainer Semantic Loss And Modes

**Files:**
- Modify: `plenoxels/runners/base_trainer.py`
- Extend: `tests/test_semantic_branch_gradients.py`

- [ ] **Step 1: Add trainer loss helper test**

Append this test to `SemanticBranchGradientTest`:

```python
    def test_cosine_semantic_loss_is_zero_for_matching_features(self):
        from plenoxels.runners.base_trainer import semantic_cosine_loss

        pred = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        target = torch.tensor([[2.0, 0.0], [0.0, 1.0]])

        loss = semantic_cosine_loss(pred, target)

        self.assertLess(float(loss), 1e-6)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m unittest tests.test_semantic_branch_gradients -v
```

Expected: fail because `semantic_cosine_loss` does not exist.

- [ ] **Step 3: Add semantic loss helper and config fields**

In `plenoxels/runners/base_trainer.py`, import:

```python
import torch.nn.functional as F
```

Add helper near `losses_to_postfix`:

```python
def semantic_cosine_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = F.normalize(preds.float(), dim=-1)
    targets = F.normalize(targets.float(), dim=-1)
    return (1.0 - torch.sum(preds * targets, dim=-1)).mean()
```

In `BaseTrainer.__init__`, set:

```python
        self.train_rgb = kwargs.get("train_rgb", True)
        self.train_semantic = kwargs.get("train_semantic", False)
        self.freeze_rgb_for_semantic = kwargs.get("freeze_rgb_for_semantic", False)
        self.semantic_loss_weight = float(kwargs.get("semantic_loss_weight", 0.0))
```

- [ ] **Step 4: Freeze RGB before optimizer creation when requested**

In `BaseTrainer.__init__`, after `self.model = self.init_model(**self.extra_args)` and before `self.optimizer = self.init_optim(**self.extra_args)`, add:

```python
        if self.freeze_rgb_for_semantic:
            if not hasattr(self.model, "freeze_rgb_parameters"):
                raise RuntimeError("freeze_rgb_for_semantic requires a model with freeze_rgb_parameters()")
            self.model.freeze_rgb_parameters()
```

This ensures frozen RGB parameters enter the optimizer with `requires_grad=False`; semantic-only training can still load RGB weights with the existing checkpoint path.

- [ ] **Step 5: Filter frozen parameters in optimizer groups**

In `BaseTrainer.init_optim`, replace the Adam creation with:

```python
            param_groups = []
            for group in self.model.get_params(kwargs["lr"]):
                params = [p for p in group["params"] if p.requires_grad]
                if params:
                    group = dict(group)
                    group["params"] = params
                    param_groups.append(group)
            optim = torch.optim.Adam(params=param_groups, eps=1e-15)
```

This prevents semantic-only mode from carrying frozen RGB parameters in the optimizer state.

- [ ] **Step 6: Move semantic targets to device**

In `_move_data_to_device`, add:

```python
        if "openseg_features" in data:
            data["openseg_features"] = data["openseg_features"].to(self.device)
```

- [ ] **Step 7: Add semantic loss to train_step**

Replace reconstruction/loss block in `train_step` with:

```python
            recon_loss = self.criterion(fwd_out["rgb"], data["imgs"])
            loss = torch.zeros((), dtype=recon_loss.dtype, device=recon_loss.device)
            if self.train_rgb:
                loss = loss + recon_loss
                for r in self.regularizers:
                    reg_loss = r.regularize(self.model, model_out=fwd_out)
                    loss = loss + reg_loss
            semantic_loss = None
            if self.train_semantic and "openseg_features" in data:
                if "semantic_features" not in fwd_out:
                    raise RuntimeError("Semantic training is enabled, but model did not return semantic_features")
                semantic_loss = semantic_cosine_loss(fwd_out["semantic_features"], data["openseg_features"])
                loss = loss + self.semantic_loss_weight * semantic_loss
```

Keep regularizer reporting guarded by `self.train_rgb`:

```python
                if self.train_rgb:
                    for r in self.regularizers:
                        r.report(self.loss_info)
                if semantic_loss is not None:
                    self.loss_info["semantic"].update(semantic_loss.item())
```

- [ ] **Step 8: Run trainer helper test**

Run:

```powershell
python -m unittest tests.test_semantic_branch_gradients -v
```

Expected: tests pass.

- [ ] **Step 9: Run RGB-only smoke unit path**

Run:

```powershell
python -m unittest tests.test_extract_openseg_neu3d tests.test_openseg_cache tests.test_video_dataset_semantics tests.test_semantic_branch_gradients -v
```

Expected: tests pass in an environment with tiny-cuda-nn import support. If local GPU/tiny-cuda-nn is unavailable, record that the full command must run on Kaggle before merge.

- [ ] **Step 10: Commit**

```powershell
git add plenoxels/runners/base_trainer.py tests/test_semantic_branch_gradients.py
git commit -m "feat: add semantic training loss"
```

---

## Task 6: Semantic Configs

**Files:**
- Create: `plenoxels/configs/local/dynerf_cm_semantic_smoke.py`
- Create: `plenoxels/configs/local/dynerf_cm_semantic_64f_ds16.py`

- [ ] **Step 1: Create semantic smoke config**

Create `plenoxels/configs/local/dynerf_cm_semantic_smoke.py`:

```python
from plenoxels.configs.local.dynerf_cm_baseline_smoke import config

config = dict(config)

config["expname"] = "cm_semantic_smoke"
config["data_downsample"] = 4
config["batch_size"] = 64
config["num_steps"] = 20
config["save_every"] = 20
config["max_train_cameras"] = 1
config["max_train_tsteps"] = 4
config["max_test_cameras"] = 1
config["max_test_tsteps"] = 2

config["semantic_enabled"] = True
config["train_rgb"] = True
config["train_semantic"] = True
config["freeze_rgb_for_semantic"] = False
config["semantic_detach_geometry"] = True
config["semantic_feature_dim"] = 768
config["semantic_loss_weight"] = 0.1
config["openseg_cache_dir"] = "/kaggle/input/coffee-martini-openseg-ds16-64f"

config["semantic_grid_config"] = [{
    "grid_dimensions": 2,
    "input_coordinate_dim": 4,
    "output_coordinate_dim": 8,
    "resolution": [16, 16, 16, 16],
}]
config["semantic_multiscale_res"] = [1]
config["semantic_linear_decoder_layers"] = 1
```

- [ ] **Step 2: Create 64-frame config**

Create `plenoxels/configs/local/dynerf_cm_semantic_64f_ds16.py`:

```python
from plenoxels.configs.local.dynerf_cm_rgb_real_64f_10k import config

config = dict(config)

config["expname"] = "cm_semantic_64f_ds16"
config["data_downsample"] = 4
config["batch_size"] = 1024
config["num_steps"] = 10000
config["save_every"] = 10000
config["valid_every"] = -1

config["semantic_enabled"] = True
config["train_rgb"] = True
config["train_semantic"] = True
config["freeze_rgb_for_semantic"] = False
config["semantic_detach_geometry"] = True
config["semantic_feature_dim"] = 768
config["semantic_loss_weight"] = 0.1
config["openseg_cache_dir"] = "/kaggle/input/coffee-martini-openseg-ds16-64f"

config["semantic_grid_config"] = [{
    "grid_dimensions": 2,
    "input_coordinate_dim": 4,
    "output_coordinate_dim": 8,
    "resolution": [64, 64, 64, 64],
}]
config["semantic_multiscale_res"] = [1, 2]
config["semantic_linear_decoder_layers"] = 1
```

- [ ] **Step 3: Import configs locally**

Run:

```powershell
python - <<'PY'
import importlib.util
for path in [
    "plenoxels/configs/local/dynerf_cm_semantic_smoke.py",
    "plenoxels/configs/local/dynerf_cm_semantic_64f_ds16.py",
]:
    spec = importlib.util.spec_from_file_location("cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(path, mod.config["expname"], mod.config["semantic_enabled"])
PY
```

Expected:

```text
plenoxels/configs/local/dynerf_cm_semantic_smoke.py cm_semantic_smoke True
plenoxels/configs/local/dynerf_cm_semantic_64f_ds16.py cm_semantic_64f_ds16 True
```

- [ ] **Step 4: Commit**

```powershell
git add plenoxels/configs/local/dynerf_cm_semantic_smoke.py plenoxels/configs/local/dynerf_cm_semantic_64f_ds16.py
git commit -m "chore: add semantic coffee martini configs"
```

---

## Task 7: Runbook And Kaggle Verification

**Files:**
- Modify: `docs/kaggle_runbook.md`

- [ ] **Step 1: Add Kaggle commands to runbook**

Append this section to `docs/kaggle_runbook.md`:

```markdown
## Semantic Branch Smoke Training

After runtime setup and after mounting/downloading the OpenSeg cache dataset:

```bash
cd /kaggle/working/BestNeRF/k-planes

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LIBRARY_PATH
export TCNN_CUDA_ARCHITECTURES=75

find /kaggle/input/coffee-martini-openseg-ds16-64f -maxdepth 1 -type f -name 'cam*.npy' | wc -l

PYTHONPATH=. python plenoxels/main.py \
  --config-path plenoxels/configs/local/dynerf_cm_semantic_smoke.py
```

Expected smoke output:

```text
20/20
semantic=<finite value>
Saving model checkpoint to: ./logs/baseline/cm_semantic_smoke/model.pth
```

Run the 64-frame semantic config only after smoke succeeds:

```bash
PYTHONPATH=. python plenoxels/main.py \
  --config-path plenoxels/configs/local/dynerf_cm_semantic_64f_ds16.py
```
```

- [ ] **Step 2: Run local test suite**

Run:

```powershell
python -m unittest tests.test_extract_openseg_neu3d tests.test_openseg_cache tests.test_video_dataset_semantics tests.test_semantic_branch_gradients -v
```

Expected: pass in an environment where tiny-cuda-nn imports successfully. If local CUDA is unavailable, run the same command on Kaggle after pushing.

- [ ] **Step 3: Run Kaggle smoke**

On Kaggle:

```bash
cd /kaggle/working/BestNeRF/k-planes
git pull
bash scripts/kaggle_setup_runtime.sh

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat:$LIBRARY_PATH
export TCNN_CUDA_ARCHITECTURES=75

PYTHONPATH=. python plenoxels/main.py \
  --config-path plenoxels/configs/local/dynerf_cm_semantic_smoke.py
```

Expected: 20 steps complete, semantic loss appears in the progress bar, and `logs/baseline/cm_semantic_smoke/model.pth` exists.

- [ ] **Step 4: Commit runbook**

```powershell
git add docs/kaggle_runbook.md
git commit -m "docs: add semantic training commands"
```

---

## Self-Review Checklist

Spec coverage:

- Detached joint training: Tasks 4 and 5.
- Semantic-only from checkpoint support: Task 4 adds `freeze_rgb_parameters`; Task 5 wires `freeze_rgb_for_semantic`, filters frozen optimizer parameters, and keeps existing `--log-dir` checkpoint loading.
- OpenSeg cache mmap and coordinate mapping: Task 1.
- Camera/frame/pixel metadata: Task 2.
- Separate semantic field: Task 3.
- Configs and Kaggle commands: Tasks 6 and 7.
- Gradient isolation test: Task 4.

Implementation guardrails:

- Do not change `plenoxels/main.py` unless checkpoint loading cannot handle semantic state with `strict=False`.
- Do not import TensorFlow or OpenNeRF in training code.
- Keep semantic disabled by default.
- Run RGB-only smoke after semantic changes to confirm no regression.
