#!/usr/bin/env python3
"""Extract OpenSeg feature shards for Neural 3D Video camera mp4 files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

OPENSEG_FEATURE_DIM = 768


def find_camera_videos(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("cam*.mp4"))


def compute_feature_shape(height: int, width: int, downsample: int) -> tuple[int, int]:
    return height // downsample, width // downsample


def limited(items: Iterable[Path], max_items: Optional[int]) -> list[Path]:
    values = list(items)
    if max_items is None:
        return values
    return values[:max_items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/neu3d/coffee_martini"))
    parser.add_argument("--opennerf-root", type=Path, default=Path("../opennerf"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Defaults to <opennerf-root>/models/openseg_exported_clip.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <opennerf-root>/outputs/neu3d/coffee_martini/openseg_camckpts.",
    )
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument(
        "--feature-downsample",
        type=int,
        default=8,
        help=(
            "Feature-map downsample from raw mp4 resolution. Coffee Martini raw videos are "
            "2028x2704; K-Planes trains at data_downsample=2, and OpenSeg supervision is "
            "typically 4x below that, so the default is raw/8 = 253x338."
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_openseg_model(model_dir: Path):
    import tensorflow as tf2
    import tensorflow.compat.v1 as tf

    model_dir = model_dir.resolve()
    return tf2.saved_model.load(str(model_dir), tags=[tf.saved_model.tag_constants.SERVING])


def _import_extractor(opennerf_root: Path):
    sys.path.insert(0, str(opennerf_root.resolve()))
    from opennerf.data.utils.openseg_extractor import extract_openseg_img_feature

    return extract_openseg_img_feature


def _open_video(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def _write_temp_frame(frame_bgr: np.ndarray, path: Path, jpeg_quality: int) -> None:
    import cv2

    ok = cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError(f"Could not write temporary frame: {path}")


def save_feature_shard(
    output_path: Path,
    frame_features: Iterable[np.ndarray],
    *,
    max_frames: int,
    feature_shape: tuple[int, int, int],
    chunk_frames: int = 16,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}.tmp.npy")
    if temp_path.exists():
        temp_path.unlink()

    shard = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float16,
        shape=(max_frames, *feature_shape),
    )
    frame_count = 0
    try:
        for feature in frame_features:
            if frame_count >= max_frames:
                break
            feature = np.asarray(feature, dtype=np.float16)
            if feature.shape != feature_shape:
                raise RuntimeError(
                    f"Expected feature shape {feature_shape}, got {feature.shape}"
                )
            shard[frame_count] = feature
            frame_count += 1
    finally:
        shard.flush()
        del shard

    if frame_count == 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"No frames extracted for {output_path}")

    if output_path.exists():
        output_path.unlink()

    if frame_count == max_frames:
        temp_path.replace(output_path)
    else:
        source = np.load(temp_path, mmap_mode="r")
        trimmed = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float16,
            shape=(frame_count, *feature_shape),
        )
        for start in range(0, frame_count, chunk_frames):
            end = min(start + chunk_frames, frame_count)
            trimmed[start:end] = source[start:end]
        trimmed.flush()
        del trimmed
        del source
        temp_path.unlink()

    return {"shape": [frame_count, *feature_shape], "dtype": str(np.dtype(np.float16))}


def extract_camera(
    video_path: Path,
    output_path: Path,
    openseg_model,
    extract_openseg_img_feature,
    *,
    max_frames: int,
    feature_downsample: int,
    jpeg_quality: int,
) -> dict:
    if output_path.exists() and not output_path.is_file():
        raise RuntimeError(f"Output path exists and is not a file: {output_path}")

    cap = _open_video(video_path)
    src_width = int(cap.get(3))
    src_height = int(cap.get(4))
    feat_h, feat_w = compute_feature_shape(src_height, src_width, feature_downsample)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{video_path.stem}_openseg_"))

    def frame_features():
        frame_idx = 0
        while frame_idx < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            tmp_frame = tmp_dir / f"{frame_idx:06d}.jpg"
            _write_temp_frame(frame_bgr, tmp_frame, jpeg_quality)
            feat_chw = extract_openseg_img_feature(
                str(tmp_frame),
                openseg_model,
                img_size=[feat_h, feat_w],
            )
            yield feat_chw.permute(1, 2, 0).cpu().numpy().astype(np.float16)
            frame_idx += 1

    try:
        shard_info = save_feature_shard(
            output_path,
            frame_features(),
            max_frames=max_frames,
            feature_shape=(feat_h, feat_w, OPENSEG_FEATURE_DIM),
        )
    finally:
        cap.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "video": str(video_path),
        "output": str(output_path),
        "shape": shard_info["shape"],
        "dtype": shard_info["dtype"],
        "source_height": src_height,
        "source_width": src_width,
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    opennerf_root = args.opennerf_root.resolve()
    model_dir = args.model_dir or (opennerf_root / "models" / "openseg_exported_clip")
    output_dir = args.output_dir or (
        opennerf_root / "outputs" / "neu3d" / "coffee_martini" / "openseg_camckpts"
    )

    videos = limited(find_camera_videos(data_dir), args.max_cameras)
    if not videos:
        raise SystemExit(f"No cam*.mp4 files found in {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    extract_openseg_img_feature = _import_extractor(opennerf_root)
    openseg_model = _load_openseg_model(model_dir)

    manifest = {
        "data_dir": str(data_dir),
        "model_dir": str(model_dir.resolve()),
        "feature_downsample": args.feature_downsample,
        "max_frames": args.max_frames,
        "shards": [],
    }

    for video in videos:
        output_path = output_dir / f"{video.stem}.npy"
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing shard: {output_path}")
            continue
        print(f"Extracting {video.name} -> {output_path}")
        info = extract_camera(
            video,
            output_path,
            openseg_model,
            extract_openseg_img_feature,
            max_frames=args.max_frames,
            feature_downsample=args.feature_downsample,
            jpeg_quality=args.jpeg_quality,
        )
        print(f"Saved {output_path} shape={info['shape']} dtype={info['dtype']}")
        manifest["shards"].append(info)

    manifest_path = output_dir.parent / "openseg_camckpts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
