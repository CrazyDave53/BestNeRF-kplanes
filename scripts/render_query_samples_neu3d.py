#!/usr/bin/env python3
"""Render sample RGB + semantic query heatmaps for Neu3D train frames."""

from __future__ import annotations

import argparse
import ast
import glob
import importlib.util
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_override_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def load_config(config_path: str, overrides: Sequence[str]) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(os.path.basename(config_path), config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load config from {config_path}")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config = dict(cfg.config)
    for override in overrides:
        key, value = override.split("=", 1)
        config[key] = parse_override_value(value)
    return config


def infer_model_type(config: dict[str, Any]) -> str:
    if "keyframes" in config:
        return "video"
    if "appearance_embedding_dim" in config:
        return "phototourism"
    return "static"


def normalize_scores(scores: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    scores = scores.astype(np.float32, copy=False)
    lo = float(np.percentile(scores, low_pct))
    hi = float(np.percentile(scores, high_pct))
    if hi <= lo:
        lo = float(scores.min())
        hi = float(scores.max())
    if hi <= lo:
        return np.zeros_like(scores, dtype=np.float32)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


def colorize_heatmap(heatmap: np.ndarray, cmap: str) -> np.ndarray:
    heatmap = np.clip(heatmap.astype(np.float32, copy=False), 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        mapped = cm.get_cmap(cmap)(heatmap)[..., :3]
        return np.round(mapped * 255.0).astype(np.uint8)
    except Exception:
        color = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
        color[..., 0] = np.round(heatmap * 255.0).astype(np.uint8)
        color[..., 1] = np.round(np.sqrt(heatmap) * 220.0).astype(np.uint8)
        color[..., 2] = np.round((1.0 - heatmap) * 255.0).astype(np.uint8)
        return color


def overlay_heatmap(rgb: np.ndarray, heatmap_color: np.ndarray, alpha: float) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    heatmap_f = heatmap_color.astype(np.float32)
    out = rgb_f * (1.0 - alpha) + heatmap_f * alpha
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def write_png(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data).save(path)


def add_title(image: np.ndarray, title: str, height: int = 36) -> np.ndarray:
    pil = Image.fromarray(image)
    canvas = Image.new("RGB", (pil.width, pil.height + height), (18, 18, 18))
    canvas.paste(pil, (0, height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 9), title, fill=(245, 245, 245), font=font)
    return np.asarray(canvas)


def encode_text_features(device: str, queries: Sequence[str], clip_model_name: str, clip_pretrained: str):
    import torch

    queries = list(queries)
    try:
        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        tokens = tokenizer(queries).to(device)
    except ImportError:
        import clip

        openai_clip_name = "ViT-L/14" if clip_model_name == "ViT-L-14" else clip_model_name
        clip_model, _ = clip.load(openai_clip_name, device=device)
        tokens = clip.tokenize(queries).to(device)

    clip_model.eval()
    with torch.no_grad():
        features = clip_model.encode_text(tokens).float()
    if features.shape[-1] != 768:
        raise ValueError(f"text encoder produced {features.shape[-1]} dims, expected 768")
    features = torch.nn.functional.normalize(features, dim=-1)
    return {query: features[index] for index, query in enumerate(queries)}


def build_trainer(config: dict[str, Any], checkpoint: str):
    import torch
    from plenoxels.main import init_trainer, load_data

    config = dict(config)
    config["openseg_cache_dir"] = None
    model_type = infer_model_type(config)
    data = load_data(
        model_type,
        validate_only=False,
        render_only=False,
        **config,
    )
    config.update(data)
    trainer = init_trainer(model_type, **config)
    try:
        checkpoint_data = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    except TypeError:
        checkpoint_data = torch.load(checkpoint, map_location=trainer.device)
    trainer.load_model(checkpoint_data, training_needed=False)
    return trainer


def infer_train_camera_names(dataset) -> list[str]:
    if hasattr(dataset, "camera_names"):
        return list(dataset.camera_names)

    videopaths = sorted(glob.glob(os.path.join(dataset.datadir, "cam*.mp4")))
    camera_names = [Path(path).stem for path in videopaths]
    split_ids = np.arange(1, len(camera_names))
    if "coffee_martini" in dataset.datadir:
        split_ids = np.setdiff1d(split_ids, 12)
    if getattr(dataset, "max_cameras", None) is not None:
        split_ids = split_ids[:dataset.max_cameras]
    return [camera_names[index] for index in split_ids]


def infer_num_frames_per_camera(dataset, camera_names: Sequence[str]) -> int:
    if hasattr(dataset, "num_frames_per_camera"):
        return int(dataset.num_frames_per_camera)
    if not camera_names:
        raise ValueError("cannot infer frames per camera without train camera names")
    if len(dataset.poses) % len(camera_names) != 0:
        raise ValueError(
            f"cannot infer frames per camera: {len(dataset.poses)} poses for "
            f"{len(camera_names)} train cameras"
        )
    return len(dataset.poses) // len(camera_names)


def find_train_image_id(dataset, camera: str, raw_frame: int) -> tuple[int, int]:
    if dataset.split != "train":
        raise ValueError("expected train split dataset")
    camera_names = infer_train_camera_names(dataset)
    if camera not in camera_names:
        raise ValueError(f"camera {camera!r} is not in train cameras: {camera_names}")
    camera_index = camera_names.index(camera)
    num_frames_per_camera = infer_num_frames_per_camera(dataset, camera_names)
    start = camera_index * num_frames_per_camera
    end = start + num_frames_per_camera
    frame_ids_by_image = getattr(dataset, "frame_ids_by_image", None)
    if frame_ids_by_image is None:
        if raw_frame < 0 or raw_frame >= num_frames_per_camera:
            raise ValueError(f"frame {raw_frame} is outside train frame range")
        return start + raw_frame, camera_index

    import torch

    frame_ids = frame_ids_by_image[start:end]
    matches = torch.nonzero(frame_ids == raw_frame, as_tuple=False).flatten()
    if matches.numel() == 0:
        raise ValueError(f"frame {raw_frame} is not available for {camera}: {frame_ids.tolist()}")
    return start + int(matches[0].item()), camera_index


def render_train_frame(trainer, camera: str, raw_frame: int, text_features, batch_size: int, use_amp: bool):
    import torch
    from plenoxels.datasets.ray_utils import create_meshgrid, get_rays, stack_camera_dirs

    dataset = trainer.train_dataset
    image_id, camera_index = find_train_image_id(dataset, camera, raw_frame)
    h, w = dataset.img_h, dataset.img_w
    x, y = create_meshgrid(height=h, width=w, dev="cpu", add_half=True, flat=True)
    camera_dirs = stack_camera_dirs(x, y, dataset.intrinsics, True)
    c2w = dataset.poses[image_id]
    rays_o, rays_d = get_rays(
        camera_dirs,
        c2w,
        ndc=dataset.is_ndc,
        ndc_near=1.0,
        intrinsics=dataset.intrinsics,
        normalize_rd=True,
    )
    timestamp = dataset.timestamps[image_id * h * w].reshape(1)
    near_far = dataset.per_cam_near_fars[camera_index:camera_index + 1].to(trainer.device)
    bg_color = torch.ones((1, 3), dtype=torch.float32, device=trainer.device)

    rgb_chunks = []
    score_chunks = {query: [] for query in text_features}
    was_training = trainer.model.training
    had_eval_semantic_flag = hasattr(trainer.model, "render_semantic_in_eval")
    old_eval_semantic_flag = getattr(trainer.model, "render_semantic_in_eval", False)
    trainer.model.render_semantic_in_eval = True
    trainer.model.eval()
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
                    raise RuntimeError("model output did not include semantic_features")
                semantic = torch.nn.functional.normalize(outputs["semantic_features"].float(), dim=-1)
                rgb_chunks.append(outputs["rgb"].detach().cpu())
                for query, text_feature in text_features.items():
                    score_chunks[query].append((semantic @ text_feature).detach().cpu())
    finally:
        trainer.model.train(was_training)
        if had_eval_semantic_flag:
            trainer.model.render_semantic_in_eval = old_eval_semantic_flag
        else:
            delattr(trainer.model, "render_semantic_in_eval")

    rgb = torch.cat(rgb_chunks, 0).reshape(h, w, 3).clamp(0, 1).numpy()
    scores = {
        query: torch.cat(chunks, 0).reshape(h, w).numpy()
        for query, chunks in score_chunks.items()
    }
    return rgb, scores


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera", default="cam01")
    parser.add_argument("--frames", default="0,16,32,48,63")
    parser.add_argument("--queries", default="human,hand,glass")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--alpha", type=float, default=0.48)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("override", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    queries = parse_csv(args.queries)
    frames = [int(item) for item in parse_csv(args.frames)]

    config = load_config(args.config_path, args.override)
    trainer = build_trainer(config, args.checkpoint)
    text_features = encode_text_features(
        str(trainer.device),
        queries=queries,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )

    for frame in frames:
        rgb, scores_by_query = render_train_frame(
            trainer=trainer,
            camera=args.camera,
            raw_frame=frame,
            text_features=text_features,
            batch_size=args.batch_size,
            use_amp=args.amp,
        )
        rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        frame_dir = output_dir / args.camera / f"frame{frame:03d}"
        write_png(frame_dir / "rgb.png", rgb_u8)
        for query, scores in scores_by_query.items():
            heatmap = normalize_scores(scores)
            heatmap_u8 = np.clip(np.round(heatmap * 255.0), 0, 255).astype(np.uint8)
            heatmap_color = colorize_heatmap(heatmap, args.cmap)
            overlay = overlay_heatmap(rgb_u8, heatmap_color, args.alpha)
            side_by_side = np.concatenate(
                [
                    add_title(rgb_u8, "RGB"),
                    add_title(heatmap_color, f"query: {query}"),
                    add_title(overlay, f"overlay: {query}"),
                ],
                axis=1,
            )
            safe_query = query.replace(" ", "_").replace("/", "_")
            write_png(frame_dir / f"{safe_query}_heatmap.png", heatmap_color)
            write_png(frame_dir / f"{safe_query}_overlay.png", overlay)
            write_png(frame_dir / f"{safe_query}_side_by_side.png", side_by_side)
            np.save(frame_dir / f"{safe_query}_scores.npy", scores.astype(np.float32))
        print(f"rendered {args.camera} frame {frame:03d}: {', '.join(queries)}")

    print(f"Saved sample query renders to {output_dir}")


if __name__ == "__main__":
    main()
