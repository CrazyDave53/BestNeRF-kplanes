"""Prepare Coffee Martini frames and manifest rows for human mask annotation.

This script does not run Grounded-SAM2 itself. It creates the stable folder
layout that a proposal model and human reviewer can fill:

  images/      extracted RGB frames
  proposals/   model-proposed masks
  masks/       human-reviewed final masks
  overlays/    review overlays
  manifest.csv annotation/review ledger
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


MANIFEST_FIELDS = [
    "image_path",
    "proposal_path",
    "mask_path",
    "overlay_path",
    "query",
    "camera",
    "frame",
    "status",
    "source",
    "reviewer",
    "notes",
]


def parse_csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def parse_frame_list(value: str) -> list[int]:
    frames = []
    for raw in parse_csv_values(value):
        try:
            frame = int(raw)
        except ValueError as exc:
            raise ValueError(f"frame id must be an integer, got {raw!r}") from exc
        if frame < 0:
            raise ValueError("frame ids must be non-negative")
        frames.append(frame)
    return sorted(set(frames))


def camera_sort_key(path_or_name: str | Path) -> tuple[str, int | str]:
    stem = Path(path_or_name).stem
    if stem.startswith("cam") and stem[3:].isdigit():
        return ("cam", int(stem[3:]))
    return ("other", stem)


def discover_cameras(data_dir: Path) -> list[str]:
    cameras = sorted(
        (path.stem for path in data_dir.glob("cam*.mp4")),
        key=camera_sort_key,
    )
    if not cameras:
        raise FileNotFoundError(f"no cam*.mp4 files found in {data_dir}")
    return cameras


def select_llff_split_cameras(
    cameras: Sequence[str],
    split: str,
    datadir_name: str,
) -> list[str]:
    cameras = sorted(cameras, key=camera_sort_key)
    if split == "train":
        split_ids = list(range(1, len(cameras)))
    elif split == "test":
        split_ids = [0]
    elif split == "all":
        split_ids = list(range(len(cameras)))
    else:
        raise ValueError("--split must be one of: train, test, all")

    if "coffee_martini" in datadir_name:
        split_ids = [index for index in split_ids if index != 12]
    return [cameras[index] for index in split_ids]


def resolve_cameras(
    data_dir: Path,
    cameras_arg: str,
    split: str,
    max_cameras: int | None = None,
) -> list[str]:
    if cameras_arg.strip().lower() == "auto":
        cameras = select_llff_split_cameras(
            discover_cameras(data_dir),
            split=split,
            datadir_name=str(data_dir),
        )
    elif cameras_arg.strip().lower() == "all":
        cameras = discover_cameras(data_dir)
    else:
        cameras = parse_csv_values(cameras_arg)
        missing = [name for name in cameras if not (data_dir / f"{name}.mp4").is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing camera videos in {data_dir}: {', '.join(missing)}"
            )
        cameras = sorted(cameras, key=camera_sort_key)
    if max_cameras is not None:
        if max_cameras < 1:
            raise ValueError("--max-cameras must be positive")
        cameras = cameras[:max_cameras]
    return cameras


def frame_stem(camera: str, frame: int) -> str:
    return f"{camera}_frame{frame:03d}"


def build_manifest_rows(
    cameras: Sequence[str],
    frames: Sequence[int],
    queries: Sequence[str],
    source: str,
) -> list[dict[str, str]]:
    rows = []
    for camera in cameras:
        for frame in frames:
            stem = frame_stem(camera, frame)
            for query in queries:
                rows.append({
                    "image_path": f"images/{stem}.png",
                    "proposal_path": f"proposals/{query}/{stem}.png",
                    "mask_path": f"masks/{query}/{stem}.png",
                    "overlay_path": f"overlays/{query}/{stem}_overlay.png",
                    "query": query,
                    "camera": camera,
                    "frame": str(frame),
                    "status": "pending",
                    "source": source,
                    "reviewer": "",
                    "notes": "",
                })
    return rows


def write_manifest(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def is_binary_mask_array(mask: np.ndarray) -> bool:
    if mask.dtype == np.bool_:
        return True
    values = np.unique(mask)
    return set(values.tolist()).issubset({0, 1, 255})


def ensure_annotation_dirs(output_dir: Path, queries: Iterable[str]) -> None:
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    for base in ("proposals", "masks", "overlays"):
        for query in queries:
            (output_dir / base / query).mkdir(parents=True, exist_ok=True)


def extract_frame(video_path: Path, frame_id: int, output_path: Path, downsample: float) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {frame_id} from {video_path}")
        if downsample != 1.0:
            if downsample <= 0:
                raise ValueError("--downsample must be positive")
            height, width = frame_bgr.shape[:2]
            target_width = max(1, int(round(width / downsample)))
            target_height = max(1, int(round(height / downsample)))
            frame_bgr = cv2.resize(
                frame_bgr,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), frame_bgr)
        if not ok:
            raise RuntimeError(f"could not write frame image: {output_path}")
    finally:
        cap.release()


def extract_selected_frames(
    data_dir: Path,
    output_dir: Path,
    cameras: Sequence[str],
    frames: Sequence[int],
    downsample: float,
    overwrite_images: bool,
) -> None:
    for camera in cameras:
        video_path = data_dir / f"{camera}.mp4"
        for frame in frames:
            image_path = output_dir / "images" / f"{frame_stem(camera, frame)}.png"
            if image_path.exists() and not overwrite_images:
                continue
            extract_frame(video_path, frame, image_path, downsample)


def write_readme(output_dir: Path) -> None:
    text = """# Coffee Martini Human Annotations

This folder is prepared for model-assisted human annotation.

Workflow:

1. Run a segmentation model such as Grounded-SAM2 to fill `proposals/<query>/`.
2. Review every proposal overlay.
3. Write the final reviewed binary masks to `masks/<query>/`.
4. Update `manifest.csv` status:
   - `accepted` if the proposal is correct;
   - `corrected` if the final mask was edited;
   - `rejected` if it should not be evaluated;
   - `ambiguous` if the query/object is unclear.
5. Only `accepted` and `corrected` rows should be used for metrics.

Masks should be binary PNGs: 0 for background, 255 for foreground.
"""
    (output_dir / "README.md").write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cameras",
        default="auto",
        help="comma-separated camera names, 'all', or 'auto' for the LLFF split cameras",
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument(
        "--frames",
        default="0,8,16,24,32,40,48,56",
        help="comma-separated source video frame ids",
    )
    parser.add_argument(
        "--queries",
        default="human,hand",
        help="comma-separated text queries to annotate",
    )
    parser.add_argument("--source", default="grounded_sam2")
    parser.add_argument("--downsample", type=float, default=2.0)
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    cameras = resolve_cameras(data_dir, args.cameras, args.split, args.max_cameras)
    frames = parse_frame_list(args.frames)
    queries = parse_csv_values(args.queries)

    ensure_annotation_dirs(output_dir, queries)
    extract_selected_frames(
        data_dir=data_dir,
        output_dir=output_dir,
        cameras=cameras,
        frames=frames,
        downsample=args.downsample,
        overwrite_images=args.overwrite_images or args.overwrite,
    )

    rows = build_manifest_rows(
        cameras=cameras,
        frames=frames,
        queries=queries,
        source=args.source,
    )
    write_manifest(output_dir / "manifest.csv", rows)
    write_readme(output_dir)

    print(f"Wrote {len(rows)} manifest rows to {output_dir / 'manifest.csv'}")
    print(f"Extracted {len(cameras) * len(frames)} RGB frames to {output_dir / 'images'}")
    print(f"Selected cameras for split {args.split}: {', '.join(cameras)}")
    print("Next: run Grounded-SAM2 proposals, then review masks into masks/<query>/")


if __name__ == "__main__":
    main()
