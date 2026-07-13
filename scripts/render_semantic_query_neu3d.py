"""Render a human-query heatmap from the K-Planes OpenSeg semantic branch."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


QUERY = "human"


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32, copy=False)
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score <= min_score:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - min_score) / (max_score - min_score)


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.clip(heatmap.astype(np.float32, copy=False), 0.0, 1.0)
    color = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
    color[..., 0] = np.round(heatmap * 255.0).astype(np.uint8)
    color[..., 2] = np.round((1.0 - heatmap) * 255.0).astype(np.uint8)
    return color


def overlay_heatmap(rgb: np.ndarray, heatmap_color: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    heatmap_f = heatmap_color.astype(np.float32)
    out = rgb_f * (1.0 - alpha) + heatmap_f * alpha
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def write_png(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data).save(path)


def load_config(config_path: str, overrides: list[str]) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(os.path.basename(config_path), config_path)
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


def encode_human_text(device: str, clip_model_name: str, clip_pretrained: str):
    import torch

    try:
        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        tokens = tokenizer([QUERY]).to(device)
    except ImportError:
        import clip

        openai_clip_name = "ViT-L/14" if clip_model_name == "ViT-L-14" else clip_model_name
        clip_model, _ = clip.load(openai_clip_name, device=device)
        tokens = clip.tokenize([QUERY]).to(device)

    clip_model.eval()
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens).float()
    if text_features.shape[-1] != 768:
        raise ValueError(
            f"Text encoder produced {text_features.shape[-1]} dims, expected 768 for OpenSeg"
        )
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    return text_features[0]


def render_query_frame(trainer, frame_index: int, text_feature, batch_size: int, use_amp: bool):
    import torch

    dataset = trainer.test_dataset
    if frame_index < 0 or frame_index >= len(dataset):
        raise IndexError(f"frame_index {frame_index} is out of range for dataset length {len(dataset)}")

    data = dataset[frame_index]
    if isinstance(dataset.img_h, int):
        img_h, img_w = dataset.img_h, dataset.img_w
    else:
        img_h, img_w = dataset.img_h[frame_index], dataset.img_w[frame_index]

    rays_o = data["rays_o"]
    rays_d = data["rays_d"]
    timestamp = data["timestamps"]
    near_far = data["near_fars"].to(trainer.device)
    bg_color = data["bg_color"]
    if isinstance(bg_color, torch.Tensor):
        bg_color = bg_color.to(trainer.device)

    rgb_chunks = []
    score_chunks = []
    was_training = trainer.model.training
    trainer.model.train()
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for start in range(0, rays_o.shape[0], batch_size):
                end = min(start + batch_size, rays_o.shape[0])
                rays_o_b = rays_o[start:end].to(trainer.device)
                rays_d_b = rays_d[start:end].to(trainer.device)
                if timestamp.numel() == 1:
                    timestamps_b = timestamp.expand(rays_o_b.shape[0]).to(trainer.device)
                else:
                    timestamps_b = timestamp[start:end].to(trainer.device)

                outputs = trainer.model(
                    rays_o_b,
                    rays_d_b,
                    timestamps=timestamps_b,
                    bg_color=bg_color,
                    near_far=near_far,
                )
                if "semantic_features" not in outputs:
                    raise RuntimeError(
                        "model did not emit semantic_features; check semantic_enabled and checkpoint/config"
                    )

                semantic = torch.nn.functional.normalize(
                    outputs["semantic_features"].float(), dim=-1
                )
                scores = semantic @ text_feature
                rgb_chunks.append(outputs["rgb"].detach().cpu())
                score_chunks.append(scores.detach().cpu())
    finally:
        trainer.model.train(was_training)

    rgb = torch.cat(rgb_chunks, 0).reshape(img_h, img_w, 3).clamp(0, 1)
    scores = torch.cat(score_chunks, 0).reshape(img_h, img_w)
    return rgb.numpy(), scores.numpy()


def build_trainer(config: dict[str, Any], checkpoint: str):
    import torch
    from plenoxels.main import init_trainer, load_data

    model_type = infer_model_type(config)
    data = load_data(
        model_type,
        validate_only=True,
        render_only=False,
        **config,
    )
    config.update(data)
    trainer = init_trainer(model_type, **config)
    try:
        checkpoint_data = torch.load(
            checkpoint,
            map_location=trainer.device,
            weights_only=False,
        )
    except TypeError:
        checkpoint_data = torch.load(checkpoint, map_location=trainer.device)
    trainer.load_model(checkpoint_data, training_needed=False)
    return trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("override", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = load_config(args.config_path, args.override)
    trainer = build_trainer(config, args.checkpoint)
    text_feature = encode_human_text(
        str(trainer.device),
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
    rgb, scores = render_query_frame(
        trainer=trainer,
        frame_index=args.frame_index,
        text_feature=text_feature,
        batch_size=args.batch_size,
        use_amp=args.amp,
    )

    rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    heatmap = normalize_scores(scores)
    heatmap_u8 = np.clip(np.round(heatmap * 255.0), 0, 255).astype(np.uint8)
    heatmap_color = colorize_heatmap(heatmap)
    overlay = overlay_heatmap(rgb_u8, heatmap_color)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_png(output_dir / "rgb.png", rgb_u8)
    write_png(output_dir / "human_heatmap_gray.png", heatmap_u8)
    write_png(output_dir / "human_heatmap_color.png", heatmap_color)
    write_png(output_dir / "human_overlay.png", overlay)
    np.save(output_dir / "human_scores.npy", scores.astype(np.float32))
    print(f"Saved human query render outputs to {output_dir}")


if __name__ == "__main__":
    main()
