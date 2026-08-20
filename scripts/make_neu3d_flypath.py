#!/usr/bin/env python3
"""Build a reusable interpolated camera fly path for Neu3D/LLFF videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_TRAIN_CAMERAS = (
    "cam01", "cam02", "cam04", "cam05", "cam06", "cam07", "cam08", "cam09",
    "cam10", "cam11", "cam12", "cam14", "cam16", "cam18", "cam19", "cam20",
)


def normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        raise ValueError(f"cannot normalize near-zero vector: {vec}")
    return vec / norm


def load_centered_llff_poses(data_dir: Path, downsample: float, near_scaling: float) -> np.ndarray:
    from plenoxels.datasets.llff_dataset import load_llff_poses_helper

    poses, _near_fars, _intrinsics = load_llff_poses_helper(
        str(data_dir),
        downsample=downsample,
        near_scaling=near_scaling,
    )
    return poses.astype(np.float64, copy=False)


def camera_names_for_pose_file(data_dir: Path) -> list[str]:
    names = sorted(path.stem for path in data_dir.glob("cam*.mp4"))
    if not names:
        raise FileNotFoundError(f"no cam*.mp4 files found in {data_dir}")
    return names


def pose_axes(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    right = normalize(c2w[:, 0])
    up = normalize(c2w[:, 1])
    back = normalize(c2w[:, 2])
    position = c2w[:, 3].astype(np.float64, copy=False)
    return position, right, up, back


def rebuild_pose(position: np.ndarray, forward: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = normalize(forward)
    back = -forward
    right = normalize(np.cross(up_hint, back))
    up = normalize(np.cross(back, right))
    pose = np.zeros((3, 4), dtype=np.float64)
    pose[:, 0] = right
    pose[:, 1] = up
    pose[:, 2] = back
    pose[:, 3] = position
    return pose


def interpolate_pose(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    pos_a, _right_a, up_a, back_a = pose_axes(a)
    pos_b, _right_b, up_b, back_b = pose_axes(b)
    position = (1.0 - t) * pos_a + t * pos_b
    forward = normalize((1.0 - t) * (-back_a) + t * (-back_b))
    up_hint = normalize((1.0 - t) * up_a + t * up_b)
    return rebuild_pose(position, forward, up_hint)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_flypath(
    poses_by_camera: dict[str, np.ndarray],
    camera_order: Sequence[str],
    steps_per_segment: int,
    close_loop: bool,
) -> list[dict]:
    if steps_per_segment < 1:
        raise ValueError("--steps-per-segment must be >= 1")

    missing = [camera for camera in camera_order if camera not in poses_by_camera]
    if missing:
        raise ValueError(f"missing requested cameras: {missing}")

    path = []
    segment_count = len(camera_order) if close_loop else len(camera_order) - 1
    for segment_index in range(segment_count):
        camera_a = camera_order[segment_index]
        camera_b = camera_order[(segment_index + 1) % len(camera_order)]
        pose_a = poses_by_camera[camera_a]
        pose_b = poses_by_camera[camera_b]
        for step_index in range(steps_per_segment):
            t = step_index / steps_per_segment
            pose = interpolate_pose(pose_a, pose_b, t)
            position, _right, up, back = pose_axes(pose)
            path.append({
                "index": len(path),
                "segment": [camera_a, camera_b],
                "segment_t": t,
                "position": position.tolist(),
                "forward": (-back).tolist(),
                "up": up.tolist(),
                "c2w": pose.tolist(),
            })

    final_camera = camera_order[0] if close_loop else camera_order[-1]
    final_pose = poses_by_camera[final_camera]
    position, _right, up, back = pose_axes(final_pose)
    path.append({
        "index": len(path),
        "segment": [final_camera, final_camera],
        "segment_t": 1.0,
        "position": position.tolist(),
        "forward": (-back).tolist(),
        "up": up.tolist(),
        "c2w": final_pose.tolist(),
    })
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Neu3D scene directory containing cam*.mp4 and poses_bounds.npy")
    parser.add_argument("--output", required=True, help="JSON path to write")
    parser.add_argument("--cameras", default=",".join(DEFAULT_TRAIN_CAMERAS))
    parser.add_argument("--downsample", type=float, default=4.0)
    parser.add_argument("--near-scaling", type=float, default=0.9)
    parser.add_argument("--steps-per-segment", type=int, default=8)
    parser.add_argument("--close-loop", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    camera_order = parse_csv(args.cameras)
    all_camera_names = camera_names_for_pose_file(data_dir)
    poses = load_centered_llff_poses(data_dir, args.downsample, args.near_scaling)
    if poses.shape[0] != len(all_camera_names):
        raise ValueError(
            f"pose/camera mismatch: poses_bounds has {poses.shape[0]} poses, "
            f"but found {len(all_camera_names)} camera videos"
        )

    poses_by_camera = dict(zip(all_camera_names, poses))
    frames = build_flypath(
        poses_by_camera=poses_by_camera,
        camera_order=camera_order,
        steps_per_segment=args.steps_per_segment,
        close_loop=args.close_loop,
    )
    payload = {
        "scene": data_dir.name,
        "source": "poses_bounds.npy centered with plenoxels.datasets.llff_dataset.load_llff_poses_helper",
        "downsample": args.downsample,
        "near_scaling": args.near_scaling,
        "camera_order": camera_order,
        "steps_per_segment": args.steps_per_segment,
        "close_loop": args.close_loop,
        "num_frames": len(frames),
        "frames": frames,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(frames)} fly-path frames to {output}")
    print(f"Cameras: {', '.join(camera_order)}")


if __name__ == "__main__":
    main()
