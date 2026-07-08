# Semantic Branch Design

Date: 2026-07-09

## Goal

Add OpenSeg semantic supervision to the K-Planes video training pipeline for Coffee Martini while keeping RGB learning isolated from semantic learning.

The first implementation target is detached joint training from scratch:

- Train RGB and semantic branches in the same loop.
- RGB loss updates only RGB field, density, and proposal networks.
- Semantic loss updates only the semantic branch.
- Semantic branch may reuse final RGB sample positions and weights, but semantic gradients must not flow into RGB, density, or proposal parameters.

The design also supports semantic-only training from an RGB checkpoint as a second mode.

## Inputs And Assumptions

The first semantic cache is the Kaggle dataset:

```text
lenguyenminhchau/coffee-martini-openseg-ds16-64f
```

Expected files:

```text
cam00.npy
cam01.npy
cam02.npy
cam04.npy
cam05.npy
cam06.npy
cam07.npy
cam08.npy
cam09.npy
cam10.npy
cam11.npy
cam12.npy
cam13.npy
cam14.npy
cam16.npy
cam18.npy
cam19.npy
cam20.npy
openseg_camckpts_manifest.json
```

Expected shard shape and dtype:

```text
(64, 126, 169, 768) float16
```

The first semantic training configuration uses:

```text
RGB data_downsample = 4
RGB resolution      = 507 x 676
OpenSeg resolution  = 126 x 169
OpenSeg target dim  = 768
max_train_tsteps    = 64
```

At this resolution, OpenSeg features are approximately 4x lower than the RGB training grid. Higher-resolution inference can be tested later by rendering denser camera rays, but the first validation target is the training resolution.

## Supported Modes

### Mode 1: Detached Joint From Scratch

This is the first implementation and smoke-test target.

Configuration intent:

```text
train_rgb = true
train_semantic = true
semantic_enabled = true
semantic_detach_geometry = true
load checkpoint = false
```

Behavior:

- RGB branch trains normally from RGB reconstruction loss.
- Semantic branch trains from OpenSeg feature loss.
- Semantic branch uses final RGB ray samples and final RGB weights after detaching them.
- Semantic loss cannot affect RGB color, density, or proposal sampling.
- RGB loss cannot affect semantic branch parameters.

### Mode 2: Semantic-Only From RGB Checkpoint

This is a supported fallback and later workflow.

Configuration intent:

```text
train_rgb = false
train_semantic = true
semantic_enabled = true
semantic_detach_geometry = true
load checkpoint = true
freeze_rgb_for_semantic = true
```

Behavior:

- Load an RGB checkpoint.
- Freeze RGB field, density, and proposal networks.
- Run RGB geometry forward only to produce sample positions and weights.
- Train only semantic branch parameters.

### Out Of Scope For First Implementation

Semantic-only training from scratch without RGB reconstruction is not a first target. It would require semantic loss to create geometry, which conflicts with the current isolation goal.

## Architecture

Add a separate semantic K-Plane field inside `LowrankModel` as a sibling to the existing RGB field.

Existing RGB path remains responsible for:

- proposal sampling
- density
- RGB field output
- final RGB weights
- RGB rendering
- depth and accumulation outputs

New semantic path is responsible for:

- semantic K-Plane interpolation
- semantic decoder output
- semantic feature rendering

Forward pass during training:

1. Build the usual RGB ray bundle.
2. Generate proposal and final ray samples as current K-Planes does.
3. Evaluate the RGB field and final density.
4. Compute final RGB weights.
5. Render RGB, depth, and accumulation normally.
6. If semantic training is enabled:
   - take final sample positions and timestamps,
   - detach geometry inputs required to enforce semantic isolation,
   - evaluate the semantic field,
   - render semantic features with detached final RGB weights,
   - return `semantic_features` with shape `[batch, 768]`.

The semantic branch should be disabled by default. Existing configs without semantic options must preserve current behavior.

## Implementation Touchpoints

This section lists the expected files for implementation. The exact line-level plan belongs in the implementation plan, but the design should make the ownership boundaries clear.

### Files To Edit Or Create

```text
plenoxels/datasets/openseg_cache.py
```

Create this file. It owns OpenSeg cache loading, shard validation, memory mapping, and RGB-pixel-to-feature-pixel lookup. Keeping this separate prevents semantic cache logic from bloating the video dataset.

```text
plenoxels/datasets/video_datasets.py
```

Edit this file. It owns Coffee Martini camera selection, video frame loading, random ray sampling, and train batch construction. It must preserve selected `camXX` names, compute train-ray metadata, and attach OpenSeg targets when semantic cache options are configured.

```text
plenoxels/runners/video_trainer.py
```

Edit this file. It creates `Video360Dataset` instances and should pass semantic cache configuration into the train dataset. It may also hold video-specific semantic training defaults if needed.

```text
plenoxels/runners/base_trainer.py
```

Edit this file. It owns the core training step, loss calculation, optimizer setup, device transfer, checkpoint save/load, and metric logging. It should add semantic loss while preserving RGB-only behavior.

```text
plenoxels/models/semantic_kplane_field.py
```

Create this file, unless implementation proves a small class inside `kplane_field.py` is cleaner. It should own semantic K-Plane grids and the decoder that predicts OpenSeg features.

```text
plenoxels/models/lowrank_model.py
```

Edit this file. It owns ray sampling, RGB field evaluation, final weights, and rendering outputs. It should instantiate the optional semantic branch and render `semantic_features` from detached final geometry.

```text
plenoxels/configs/local/dynerf_cm_semantic_smoke.py
plenoxels/configs/local/dynerf_cm_semantic_64f_ds16.py
```

Create these configs for the first smoke run and the first real 64-frame semantic run.

```text
tests/test_openseg_cache.py
tests/test_semantic_branch_gradients.py
tests/test_video_dataset_semantics.py
```

Create focused tests for cache lookup, gradient isolation, and dataset semantic targets. These test filenames can be adjusted if a simpler layout emerges, but the coverage areas are required.

```text
docs/kaggle_runbook.md
```

Edit after implementation commands are known. It should document semantic smoke/full training commands and checkpoint/render verification commands.

### Files To Use As References

```text
scripts/extract_openseg_neu3d.py
tests/test_extract_openseg_neu3d.py
```

Reference for cache file format, feature shape, feature-downsample assumptions, and existing extraction tests. These should not need semantic training edits unless the cache format changes.

```text
plenoxels/models/kplane_field.py
plenoxels/models/density_fields.py
```

Reference for grid initialization, K-Plane interpolation, decoder patterns, and parameter grouping. The semantic field should follow these conventions instead of inventing a different grid style.

```text
plenoxels/raymarching/ray_samplers.py
```

Reference for `RayBundle`, `RaySamples`, final sample positions, and weights. The semantic branch should reuse final sample outputs from `LowrankModel` rather than resampling rays independently.

```text
plenoxels/configs/local/dynerf_cm_rgb_real_64f_10k.py
plenoxels/configs/local/dynerf_cm_baseline_smoke.py
plenoxels/configs/final/DyNeRF/dynerf_hybrid.py
```

Reference for known-good Coffee Martini settings, smoke-run patterns, and upstream DyNeRF K-Planes defaults.

```text
plenoxels/main.py
```

Reference for config loading, `--validate-only`, `--render-only`, and checkpoint loading behavior. Avoid changing this unless semantic checkpoint/render behavior requires it.

```text
../opennerf/opennerf/data/utils/openseg_extractor.py
```

Reference only. OpenNeRF should remain an external feature-extraction dependency. K-Planes semantic training should consume cached `.npy` features and should not depend on TensorFlow/OpenSeg at training time.

## Data Flow

### Camera Names

The video loader must preserve selected camera names after the train/test split.

This is required because Coffee Martini:

- reserves `cam00` for test,
- removes one unsynchronized camera,
- may use a camera subset during smoke tests.

Therefore semantic cache lookup must be filename-based, such as `cam01.npy`, not based on blind camera array position.

### Ray Metadata

For train batches, `Video360Dataset.__getitem__` should retain or compute:

- `image_id`
- `camera_id`
- `camera_name`
- `frame_id`
- RGB pixel `x`
- RGB pixel `y`

This metadata maps a sampled RGB ray to one OpenSeg feature vector.

### OpenSeg Cache Loader

Add an `OpenSegFeatureCache` component with a small interface:

```text
lookup(camera_name, frame_id, rgb_x, rgb_y, rgb_h, rgb_w) -> semantic feature
```

Responsibilities:

- Open `.npy` shards with `np.load(..., mmap_mode="r")`.
- Avoid loading the full cache into RAM.
- Validate shard existence.
- Validate expected feature dimension.
- Map RGB pixel coordinates to OpenSeg coordinates.
- Clamp mapped coordinates to valid feature-map bounds.

For the first cache:

```text
rgb_h = 507
rgb_w = 676
feat_h = 126
feat_w = 169
```

Coordinate mapping should be proportional rather than hardcoded:

```text
feat_y = floor(rgb_y / rgb_h * feat_h)
feat_x = floor(rgb_x / rgb_w * feat_w)
```

Then clamp:

```text
0 <= feat_y < feat_h
0 <= feat_x < feat_w
```

The dataset returns semantic targets as `[batch, 768]`. CPU storage may remain `float16`; trainer/device transfer can convert as needed under mixed precision.

## Loss And Optimizer

RGB loss remains the existing MSE reconstruction loss:

```text
rgb_loss = mse(pred_rgb, target_rgb)
```

Semantic loss starts as cosine distance:

```text
semantic_loss = mean(1 - cosine_similarity(pred_semantic, target_openseg))
```

Cosine loss is preferred first because OpenSeg/CLIP feature direction is more important than raw magnitude. MSE may be logged later, but it is not the first optimization target.

Total loss for detached joint mode:

```text
total_loss = rgb_loss + semantic_loss_weight * semantic_loss
```

Gradient isolation is mandatory:

- `rgb_loss.backward()` must produce gradients only for RGB field, density, and proposal parameters.
- `semantic_loss.backward()` must produce gradients only for semantic branch parameters.
- Detached final weights and detached sample positions must prevent semantic loss from updating RGB geometry.
- Parameter groups or explicit `requires_grad` filtering must prevent optimizer leakage between branches.

Semantic-only mode uses the same semantic loss, but RGB parameters are frozen and no RGB loss is optimized.

## Configuration

Add local configs:

```text
plenoxels/configs/local/dynerf_cm_semantic_smoke.py
plenoxels/configs/local/dynerf_cm_semantic_64f_ds16.py
```

The smoke config should use:

```text
num_steps = 20
max_train_cameras = 1
max_train_tsteps = 4
batch_size small enough for Kaggle T4
save_every = 20
valid_every = -1
```

The first real config should use:

```text
data_downsample = 4
max_train_cameras = None
max_train_tsteps = 64
semantic_feature_dim = 768
semantic_loss_weight configured explicitly
semantic_cache_dir = /kaggle/input/coffee-martini-openseg-ds16-64f
```

The exact semantic grid size and decoder width should start conservatively for memory. A later projected 128-dimensional version can reuse the same architecture by changing the semantic output dimension and target preprocessing.

## Testing And Verification

Unit tests:

- `OpenSegFeatureCache` loads shard metadata without reading full arrays.
- Pixel mapping maps RGB `507 x 676` coordinates into feature `126 x 169` coordinates.
- Missing camera shard raises a clear error.
- Dataset batch includes semantic target shape `[batch, 768]` when cache is enabled.
- Semantic loss gradients reach semantic parameters.
- Semantic loss gradients do not reach RGB field or proposal parameters.
- Existing RGB-only smoke config still runs without semantic options.

Kaggle verification:

1. Run RGB-only smoke to prove no regression.
2. Run semantic smoke:

   ```text
   20 steps, 1 train camera, 4 frames
   ```

3. Confirm logs include RGB metrics and semantic loss.
4. Confirm checkpoint saves.
5. Run the real 64-frame all-camera semantic config only after smoke passes.

## Risks

### Memory

Predicting 768-dimensional features for every final sample can be expensive. The first implementation should keep batch size conservative and avoid storing semantic per-sample tensors longer than needed.

### Camera Alignment

Incorrect camera-name mapping would silently train semantics against the wrong cache shard. This is why preserving `camXX` names in the dataset is a requirement.

### Frame Alignment

The cache contains the first 64 frames per camera. Training config must use `max_train_tsteps = 64` without temporal subsampling that changes frame IDs unexpectedly.

### Gradient Leakage

The most important correctness risk is semantic loss affecting RGB geometry. The implementation must include a gradient isolation test before full training.

## Later Work

- Add 128-dimensional semantic targets using PCA or a saved fixed projection.
- Add semantic-only-from-checkpoint config after detached joint smoke passes.
- Add semantic rendering/query tools for text or feature-space visualization.
- Upload important RGB and semantic checkpoints as Kaggle datasets so render checks survive instance restarts.
