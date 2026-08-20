#!/usr/bin/env python3
"""Create a Teacher / GT / Student comparison video from saved eval renders.

The expected input is the image output produced by
``evaluate_human_annotations_neu3d.py`` or ``evaluate_human_annotations_neu3d_many.py``
with image saving enabled.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_LABELS = ("RGB", "OpenSeg teacher", "Human GT", "K-Planes student")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", default="human")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--label-height", type=int, default=42)
    parser.add_argument("--save-frames-dir", type=Path, default=None)
    return parser.parse_args()


def load_manifest(annotation_dir: Path, query: str, camera: str | None) -> list[dict[str, str]]:
    manifest_path = annotation_dir / "manifest.csv"
    with manifest_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    selected = []
    for row in rows:
        if row.get("query") != query:
            continue
        if camera is not None and row.get("camera") != camera:
            continue
        if row.get("status", "accepted") not in {"accepted", "corrected"}:
            continue
        selected.append(row)

    selected.sort(key=lambda row: (row["camera"], int(row["frame"])))
    return selected


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_keep_aspect(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * (width / image.width))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def make_gt_overlay(rgb: Image.Image, mask_path: Path) -> Image.Image:
    mask = Image.open(mask_path).convert("L").resize(rgb.size, Image.Resampling.NEAREST)
    rgb_arr = np.asarray(rgb).astype(np.float32)
    mask_arr = np.asarray(mask) > 0

    overlay = rgb_arr.copy()
    green = np.array([0, 255, 80], dtype=np.float32)
    overlay[mask_arr] = overlay[mask_arr] * 0.35 + green * 0.65

    edge = mask_arr.astype(np.uint8)
    eroded = (
        edge
        & np.roll(edge, 1, axis=0)
        & np.roll(edge, -1, axis=0)
        & np.roll(edge, 1, axis=1)
        & np.roll(edge, -1, axis=1)
    )
    boundary = mask_arr & ~eroded.astype(bool)
    overlay[boundary] = np.array([255, 255, 255], dtype=np.float32)
    return Image.fromarray(np.clip(np.round(overlay), 0, 255).astype(np.uint8))


def draw_label(panel: Image.Image, label: str, label_height: int) -> Image.Image:
    canvas = Image.new("RGB", (panel.width, panel.height + label_height), (18, 18, 18))
    canvas.paste(panel, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    x = max(0, (panel.width - (bbox[2] - bbox[0])) // 2)
    y = max(0, (label_height - (bbox[3] - bbox[1])) // 2 - 2)
    draw.text((x, y), label, fill=(245, 245, 245), font=font)
    return canvas


def build_frame(
    row: dict[str, str],
    annotation_dir: Path,
    render_dir: Path,
    panel_width: int,
    label_height: int,
) -> Image.Image:
    camera = row["camera"]
    frame = int(row["frame"])
    query = row["query"]
    stem = f"{camera}_frame{frame:03d}_{query}"

    rgb = open_rgb(render_dir / f"{stem}_rgb.png")
    teacher = open_rgb(render_dir / f"{stem}_teacher_overlay.png")
    student = open_rgb(render_dir / f"{stem}_overlay.png")
    gt = make_gt_overlay(rgb, annotation_dir / row["mask_path"])

    panels = [rgb, teacher, gt, student]
    resized = [resize_keep_aspect(panel, panel_width) for panel in panels]
    target_h = max(panel.height for panel in resized)
    padded = []
    for panel, label in zip(resized, PANEL_LABELS):
        canvas = Image.new("RGB", (panel_width, target_h), (0, 0, 0))
        y = (target_h - panel.height) // 2
        canvas.paste(panel, (0, y))
        padded.append(draw_label(canvas, label, label_height))

    out = Image.new("RGB", (panel_width * len(padded), target_h + label_height), (0, 0, 0))
    for i, panel in enumerate(padded):
        out.paste(panel, (i * panel_width, 0))
    return out


def write_video(frames: Iterable[Image.Image], output: Path, fps: float) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_list = list(frames)
    if not frame_list:
        raise RuntimeError("no frames to write")

    try:
        import imageio.v2 as imageio

        with imageio.get_writer(output, fps=fps, codec="libx264", quality=8) as writer:
            for frame in frame_list:
                writer.append_data(np.asarray(frame))
    except Exception:
        import cv2

        first = np.asarray(frame_list[0])
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {output}")
        try:
            for frame in frame_list:
                rgb = np.asarray(frame)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    return len(frame_list)


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.annotation_dir, query=args.query, camera=args.camera)
    if args.max_frames is not None:
        rows = rows[: args.max_frames]
    if not rows:
        raise RuntimeError("no manifest rows selected")

    frames = []
    for row in rows:
        frame = build_frame(
            row=row,
            annotation_dir=args.annotation_dir,
            render_dir=args.render_dir,
            panel_width=args.panel_width,
            label_height=args.label_height,
        )
        frames.append(frame)
        if args.save_frames_dir is not None:
            args.save_frames_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{row['camera']}_frame{int(row['frame']):03d}_{row['query']}"
            frame.save(args.save_frames_dir / f"{stem}.png")

    count = write_video(frames, args.output, fps=args.fps)
    print(f"Wrote {count} frames to {args.output}")


if __name__ == "__main__":
    main()
