#!/usr/bin/env python3
"""Build an interpolated camera fly path from Neu3D train camera poses."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np


def load_config(config_path: str, overrides: list[str]) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(os.path.basename(config_path), config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load config from {config_path}")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config = dict(cfg.config)
    for override in overrides:
        key, value = override.split("=", 1)
        config[key] = value
    return config


def infer_model_type(config: dict[str, Any]) -> str:
    if "keyframes" in config:
        return "video"
    if "appearance_embedding_dim" in config:
        return "phototourism"
    return "static"


def orthonormalize_pose(pose: np.ndarray) -> np.ndarray:
    out = pose.astype(np.float32, copy=True)
    rotation = out[:3, :3]
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    out[:3, :3] = rotation
    return out


def interpolate_poses(a: np.ndarray, b: np.ndarray, count: int, include_endpoint: bool) -> list[np.ndarray]:
    if count <= 0:
        return []
    if include_endpoint:
        ts = np.linspace(0.0, 1.0, count, dtype=np.float32)
    else:
        ts = np.linspace(0.0, 1.0, count + 1, dtype=np.float32)[:-1]
    poses = []
    for t in ts:
        pose = (1.0 - float(t)) * a + float(t) * b
        poses.append(orthonormalize_pose(pose))
    return poses


def parse_camera_order(value: str | None, available: list[str]) -> list[str]:
    if not value:
        return available
    requested = [item.strip() for item in value.split(",") if item.strip()]
    missing = [camera for camera in requested if camera not in available]
    if missing:
        raise ValueError(f"camera(s) not in train dataset: {missing}; available={available}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-order", default=None)
    parser.add_argument("--frames-per-segment", type=int, default=12)
    parser.add_argument("--raw-frame", type=int, default=32)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("override", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    from plenoxels.main import load_data

    config = load_config(args.config_path, args.override)
    config = dict(config)
    config["openseg_cache_dir"] = None
    model_type = infer_model_type(config)
    data = load_data(model_type, validate_only=False, render_only=False, **config)
    dataset = data["train_dataset"]

    camera_names = list(dataset.camera_names)
    ordered_cameras = parse_camera_order(args.camera_order, camera_names)
    camera_to_index = {name: index for index, name in enumerate(camera_names)}

    key_poses = []
    for camera in ordered_cameras:
        camera_index = camera_to_index[camera]
        image_id = camera_index * dataset.num_frames_per_camera
        key_poses.append(dataset.poses[image_id].detach().cpu().numpy().astype(np.float32))

    path = []
    segments = len(key_poses) if args.loop else len(key_poses) - 1
    for index in range(segments):
        a = key_poses[index]
        b = key_poses[(index + 1) % len(key_poses)]
        path.extend(
            interpolate_poses(
                a,
                b,
                count=args.frames_per_segment,
                include_endpoint=False,
            )
        )
    if not args.loop:
        path.append(orthonormalize_pose(key_poses[-1]))

    poses = np.stack(path).astype(np.float32)
    timestamps = np.full((poses.shape[0],), float(args.raw_frame), dtype=np.float32)
    intrinsics = np.array(
        [
            dataset.intrinsics.width,
            dataset.intrinsics.height,
            dataset.intrinsics.focal_x,
            dataset.intrinsics.focal_y,
            dataset.intrinsics.center_x,
            dataset.intrinsics.center_y,
        ],
        dtype=np.float32,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        poses=poses,
        timestamps=timestamps,
        intrinsics=intrinsics,
        camera_order=np.array(ordered_cameras),
        key_poses=np.stack(key_poses).astype(np.float32),
        frames_per_segment=np.array(args.frames_per_segment, dtype=np.int32),
        raw_frame=np.array(args.raw_frame, dtype=np.int32),
        loop=np.array(bool(args.loop)),
    )
    print(f"Saved fly path: {output}")
    print(f"cameras: {', '.join(ordered_cameras)}")
    print(f"poses: {poses.shape}")
    print(f"raw_frame timestamp: {args.raw_frame}")


if __name__ == "__main__":
    main()
