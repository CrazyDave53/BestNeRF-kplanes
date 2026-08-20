#!/usr/bin/env python3
"""Render semantic query heatmap videos along a saved Neu3D fly path."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from render_query_samples_neu3d import (
    build_trainer,
    colorize_heatmap,
    encode_text_features,
    load_config,
    normalize_scores,
    overlay_heatmap,
    parse_csv,
    write_png,
)


def safe_query_name(query: str) -> str:
    return query.replace(" ", "_").replace("/", "_")


def pingpong_time(index: int, num_time_frames: int) -> int:
    if num_time_frames < 2:
        return 0
    period = 2 * num_time_frames - 2
    phase = index % period
    return phase if phase < num_time_frames else period - phase


def raw_time_for_index(
    index: int,
    mode: str,
    fixed_time_frame: int,
    num_time_frames: int,
) -> int:
    if mode == "fixed":
        return int(fixed_time_frame)
    if mode == "pingpong":
        return pingpong_time(index, int(num_time_frames))
    raise ValueError(f"unknown time mode: {mode}")


def normalized_video_timestamp(raw_frame: int) -> float:
    # Video360Dataset normalizes raw Neu3D frame ids with this exact convention.
    return (float(raw_frame) / 299.0) * 2.0 - 1.0


def load_flypath(path: Path, max_frames: int | None) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = list(payload["frames"])
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"fly path has no frames: {path}")
    return payload, frames


def render_flypath_frame(
    trainer,
    c2w_np: np.ndarray,
    raw_time_frame: int,
    text_features,
    batch_size: int,
    use_amp: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import torch
    from plenoxels.datasets.ray_utils import create_meshgrid, get_rays, stack_camera_dirs

    dataset = trainer.train_dataset
    h, w = dataset.img_h, dataset.img_w
    x, y = create_meshgrid(height=h, width=w, dev="cpu", add_half=True, flat=True)
    camera_dirs = stack_camera_dirs(x, y, dataset.intrinsics, True)
    c2w = torch.as_tensor(c2w_np, dtype=torch.float32)
    rays_o, rays_d = get_rays(
        camera_dirs,
        c2w,
        ndc=dataset.is_ndc,
        ndc_near=1.0,
        intrinsics=dataset.intrinsics,
        normalize_rd=True,
    )

    timestamp = torch.tensor([normalized_video_timestamp(raw_time_frame)], dtype=torch.float32)
    near_far = dataset.per_cam_near_fars[0:1].to(trainer.device)
    bg_color = torch.ones((1, 3), dtype=torch.float32, device=trainer.device)

    rgb_chunks = []
    score_chunks = {query: [] for query in text_features}
    was_training = trainer.model.training
    trainer.model.train()
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for start in range(0, rays_o.shape[0], batch_size):
                end = min(start + batch_size, rays_o.shape[0])
                rays_o_b = rays_o[start:end].to(trainer.device)
                rays_d_b = rays_d[start:end].to(trainer.device)
                timestamps_b = timestamp.expand(rays_o_b.shape[0]).to(trainer.device)
                outputs = trainer.model(
                    rays_o_b,
                    rays_d_b,
                    timestamps=timestamps_b,
                    bg_color=bg_color,
                    near_far=near_far,
                )
                if "semantic_features" not in outputs:
                    raise RuntimeError(
                        "model output did not include semantic_features; "
                        f"keys={list(outputs.keys())}"
                    )
                semantic = torch.nn.functional.normalize(outputs["semantic_features"].float(), dim=-1)
                rgb_chunks.append(outputs["rgb"].detach().cpu())
                for query, text_feature in text_features.items():
                    score_chunks[query].append((semantic @ text_feature).detach().cpu())
    finally:
        trainer.model.train(was_training)

    rgb = torch.cat(rgb_chunks, 0).reshape(h, w, 3).clamp(0, 1).numpy()
    scores = {
        query: torch.cat(chunks, 0).reshape(h, w).numpy()
        for query, chunks in score_chunks.items()
    }
    return rgb, scores


def ffmpeg_escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def run_ffmpeg(command: Sequence[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def make_forward_video(frames_dir: Path, output: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; skipping video creation")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        str(output),
    ])


def make_loop_video(frames: Sequence[Path], output: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; skipping loop video creation")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_path = output.with_suffix(".concat.txt")
    loop_frames = list(frames) + list(frames[-2:0:-1])
    concat_path.write_text(
        "".join(f"file '{ffmpeg_escape(path)}'\n" for path in loop_frames),
        encoding="utf-8",
    )
    run_ffmpeg([
        "ffmpeg", "-y",
        "-r", str(fps),
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        str(output),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--flypath", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--queries", default="hand,cup,bottle")
    parser.add_argument("--time-mode", choices=("fixed", "pingpong"), default="pingpong")
    parser.add_argument("--fixed-time-frame", type=int, default=32)
    parser.add_argument("--num-time-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--alpha", type=float, default=0.48)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("override", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    flypath_payload, flypath_frames = load_flypath(Path(args.flypath), args.max_frames)
    queries = parse_csv(args.queries)

    config = load_config(args.config_path, args.override)
    trainer = build_trainer(config, args.checkpoint)
    text_features = encode_text_features(
        str(trainer.device),
        queries=queries,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "flypath.json").write_text(
        json.dumps(flypath_payload, indent=2),
        encoding="utf-8",
    )

    metadata_rows = []
    for render_index, frame in enumerate(flypath_frames):
        raw_time = raw_time_for_index(
            render_index,
            mode=args.time_mode,
            fixed_time_frame=args.fixed_time_frame,
            num_time_frames=args.num_time_frames,
        )
        rgb, scores_by_query = render_flypath_frame(
            trainer=trainer,
            c2w_np=np.asarray(frame["c2w"], dtype=np.float32),
            raw_time_frame=raw_time,
            text_features=text_features,
            batch_size=args.batch_size,
            use_amp=args.amp,
        )
        rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        rgb_path = output_dir / "rgb" / f"frame_{render_index:04d}.png"
        write_png(rgb_path, rgb_u8)

        metadata_rows.append({
            "render_index": render_index,
            "flypath_index": frame.get("index", render_index),
            "time_frame": raw_time,
            "segment_start": frame.get("segment", ["", ""])[0],
            "segment_end": frame.get("segment", ["", ""])[1],
            "segment_t": frame.get("segment_t", ""),
        })

        for query, scores in scores_by_query.items():
            query_dir = output_dir / safe_query_name(query)
            heatmap = normalize_scores(scores)
            heatmap_color = colorize_heatmap(heatmap, args.cmap)
            overlay = overlay_heatmap(rgb_u8, heatmap_color, args.alpha)
            write_png(query_dir / "heatmaps" / f"frame_{render_index:04d}.png", heatmap_color)
            write_png(query_dir / "overlays" / f"frame_{render_index:04d}.png", overlay)
            (query_dir / "scores").mkdir(parents=True, exist_ok=True)
            np.save(
                query_dir / "scores" / f"frame_{render_index:04d}_t{raw_time:03d}.npy",
                scores.astype(np.float32),
            )
        print(
            f"rendered frame {render_index + 1}/{len(flypath_frames)} "
            f"time={raw_time:03d} queries={','.join(queries)}"
        )

    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["render_index", "flypath_index", "time_frame", "segment_start", "segment_end", "segment_t"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    if not args.no_video:
        for query in queries:
            query_dir = output_dir / safe_query_name(query)
            overlay_dir = query_dir / "overlays"
            overlay_frames = sorted(overlay_dir.glob("frame_*.png"))
            make_forward_video(
                overlay_dir,
                query_dir / f"{safe_query_name(query)}_forward.mp4",
                fps=args.fps,
            )
            make_loop_video(
                overlay_frames,
                query_dir / f"{safe_query_name(query)}_loop.mp4",
                fps=args.fps,
            )

    print(f"Saved flypath semantic demo outputs to {output_dir}")


if __name__ == "__main__":
    main()
