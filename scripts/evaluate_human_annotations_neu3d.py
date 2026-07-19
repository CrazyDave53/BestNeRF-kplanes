"""Evaluate K-Planes semantic query heatmaps against human annotation masks."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_semantic_query_neu3d import (
    colorize_heatmap,
    infer_model_type,
    load_config,
    normalize_scores,
    overlay_heatmap,
    write_png,
)


SUMMARY_METRIC_KEYS = {
    "ap",
    "best_iou",
    "best_threshold",
    "mask_pixels",
    "psnr",
    "ssim",
}


def binary_mask_from_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = image[..., 0]
    return image > 0


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def average_precision(scores: np.ndarray, target: np.ndarray) -> float:
    scores = scores.reshape(-1).astype(np.float64)
    target = target.reshape(-1).astype(bool)
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_target = target[order]
    tp = np.cumsum(sorted_target)
    ranks = np.arange(1, sorted_target.size + 1)
    precision_at_k = tp / ranks
    return float(precision_at_k[sorted_target].sum() / positives)


def threshold_metrics(scores: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    target = target.astype(bool)
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    union = float(np.logical_or(pred, target).sum())
    if union == 0.0:
        iou = 1.0
    else:
        iou = tp / union
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def best_iou(
    scores: np.ndarray,
    target: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    best_value = -1.0
    best_threshold = 0.0
    for threshold in thresholds:
        value = threshold_metrics(scores, target, float(threshold))["iou"]
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return float(best_value), float(best_threshold)


def compute_mask_metrics(
    scores: np.ndarray,
    target: np.ndarray,
    fixed_threshold: float = 0.5,
) -> dict[str, float]:
    if scores.shape != target.shape:
        raise ValueError(f"score/mask shape mismatch: {scores.shape} vs {target.shape}")
    scores = np.clip(scores.astype(np.float32, copy=False), 0.0, 1.0)
    target = target.astype(bool)
    fixed = threshold_metrics(scores, target, fixed_threshold)
    best_value, best_threshold = best_iou(scores, target)
    suffix = f"{fixed_threshold:.2f}"
    return {
        "ap": average_precision(scores, target),
        f"iou_at_{suffix}": fixed["iou"],
        f"precision_at_{suffix}": fixed["precision"],
        f"recall_at_{suffix}": fixed["recall"],
        f"f1_at_{suffix}": fixed["f1"],
        "best_iou": best_value,
        "best_threshold": best_threshold,
        "mask_pixels": float(target.sum()),
    }


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(np.float32, copy=False)
    target = target.astype(np.float32, copy=False)
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(np.float32, copy=False)
    target = target.astype(np.float32, copy=False)
    try:
        from skimage.metrics import structural_similarity

        return float(
            structural_similarity(
                target,
                pred,
                channel_axis=-1,
                data_range=1.0,
            )
        )
    except ImportError:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        values = []
        for channel in range(pred.shape[-1]):
            pred_c = pred[..., channel]
            target_c = target[..., channel]
            mu_pred = float(pred_c.mean())
            mu_target = float(target_c.mean())
            var_pred = float(pred_c.var())
            var_target = float(target_c.var())
            cov = float(((pred_c - mu_pred) * (target_c - mu_target)).mean())
            numerator = (2 * mu_pred * mu_target + c1) * (2 * cov + c2)
            denominator = (mu_pred ** 2 + mu_target ** 2 + c1) * (var_pred + var_target + c2)
            values.append(safe_divide(numerator, denominator))
        return float(np.mean(values))


def load_annotation_rows(path: Path, statuses: set[str]) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get("status", "") in statuses]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_metric_rows(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    out = []
    for group_key in sorted(groups):
        group_rows = groups[group_key]
        summary = {key: value for key, value in zip(keys, group_key)}
        summary["count"] = len(group_rows)
        numeric_keys = [
            key for key, value in group_rows[0].items()
            if (
                (key in SUMMARY_METRIC_KEYS or key.startswith(("iou_at_", "precision_at_", "recall_at_", "f1_at_")))
                and isinstance(value, (int, float, np.floating))
            )
        ]
        for key in numeric_keys:
            values = np.asarray([row[key] for row in group_rows], dtype=np.float64)
            summary[key] = float(np.nanmean(values))
        out.append(summary)
    return out


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
        raise ValueError(f"Text encoder produced {features.shape[-1]} dims, expected 768")
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
        checkpoint_data = torch.load(
            checkpoint,
            map_location=trainer.device,
            weights_only=False,
        )
    except TypeError:
        checkpoint_data = torch.load(checkpoint, map_location=trainer.device)
    trainer.load_model(checkpoint_data, training_needed=False)
    return trainer


def find_train_image_id(dataset, camera: str, raw_frame: int) -> tuple[int, int]:
    if dataset.split != "train":
        raise ValueError("expected a train split dataset")
    if camera not in dataset.camera_names:
        raise ValueError(f"camera {camera!r} is not in train dataset cameras: {dataset.camera_names}")
    camera_index = dataset.camera_names.index(camera)
    start = camera_index * dataset.num_frames_per_camera
    end = start + dataset.num_frames_per_camera
    if dataset.frame_ids_by_image is None:
        if raw_frame < 0 or raw_frame >= dataset.num_frames_per_camera:
            raise ValueError(f"frame {raw_frame} is outside train frame range")
        return start + raw_frame, camera_index

    import torch

    frame_ids = dataset.frame_ids_by_image[start:end]
    matches = torch.nonzero(frame_ids == raw_frame, as_tuple=False).flatten()
    if matches.numel() == 0:
        available = frame_ids.tolist()
        raise ValueError(
            f"frame {raw_frame} is not available for {camera}; available frames: {available}"
        )
    return start + int(matches[0].item()), camera_index


def render_train_query_frame(
    trainer,
    camera: str,
    raw_frame: int,
    text_feature,
    batch_size: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    pixel_start = image_id * h * w
    pixel_end = pixel_start + h * w
    gt_rgb = (dataset.imgs[pixel_start:pixel_end].float() / 255.0).reshape(h, w, 3).numpy()
    timestamp = dataset.timestamps[pixel_start].reshape(1)
    near_far = dataset.per_cam_near_fars[camera_index:camera_index + 1].to(trainer.device)
    bg_color = torch.ones((1, 3), dtype=torch.float32, device=trainer.device)

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
                semantic = torch.nn.functional.normalize(
                    outputs["semantic_features"].float(), dim=-1
                )
                scores = semantic @ text_feature
                rgb_chunks.append(outputs["rgb"].detach().cpu())
                score_chunks.append(scores.detach().cpu())
    finally:
        trainer.model.train(was_training)

    rgb = torch.cat(rgb_chunks, 0).reshape(h, w, 3).clamp(0, 1).numpy()
    scores = torch.cat(score_chunks, 0).reshape(h, w).numpy()
    return rgb, scores, gt_rgb


def resize_scores_to_mask(scores: np.ndarray, mask_shape: tuple[int, int]) -> np.ndarray:
    if scores.shape == mask_shape:
        return scores
    image = Image.fromarray(scores.astype(np.float32), mode="F")
    resized = image.resize((mask_shape[1], mask_shape[0]), resample=Image.BILINEAR)
    return np.array(resized, dtype=np.float32)


def evaluate_annotations(
    annotation_dir: Path,
    rows: Sequence[dict[str, str]],
    trainer,
    text_features: dict[str, Any],
    output_dir: Path,
    batch_size: int,
    use_amp: bool,
    fixed_threshold: float,
    save_images: bool,
) -> list[dict[str, Any]]:
    rendered: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    metric_rows = []
    for row in rows:
        camera = row["camera"]
        frame = int(row["frame"])
        query = row["query"]
        key = (camera, frame, query)
        if key not in rendered:
            rendered[key] = render_train_query_frame(
                trainer=trainer,
                camera=camera,
                raw_frame=frame,
                text_feature=text_features[query],
                batch_size=batch_size,
                use_amp=use_amp,
            )
        rgb, raw_scores, gt_rgb = rendered[key]
        mask = binary_mask_from_image(np.array(Image.open(annotation_dir / row["mask_path"])))
        heatmap = normalize_scores(resize_scores_to_mask(raw_scores, mask.shape))
        mask_metrics = compute_mask_metrics(
            heatmap,
            mask,
            fixed_threshold=fixed_threshold,
        )
        psnr = compute_psnr(rgb, gt_rgb)
        ssim = compute_ssim(rgb, gt_rgb)

        out_row = {
            "query": query,
            "camera": camera,
            "frame": frame,
            "status": row["status"],
            "mask_path": row["mask_path"],
            **mask_metrics,
            "psnr": psnr,
            "ssim": ssim,
        }
        metric_rows.append(out_row)

        if save_images:
            stem = f"{camera}_frame{frame:03d}_{query}"
            row_dir = output_dir / "renders"
            rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
            heatmap_u8 = np.clip(np.round(heatmap * 255.0), 0, 255).astype(np.uint8)
            heatmap_color = colorize_heatmap(heatmap)
            write_png(row_dir / f"{stem}_rgb.png", rgb_u8)
            write_png(row_dir / f"{stem}_heatmap.png", heatmap_u8)
            write_png(row_dir / f"{stem}_overlay.png", overlay_heatmap(rgb_u8, heatmap_color))
            np.save(row_dir / f"{stem}_scores.npy", raw_scores.astype(np.float32))
        print(
            f"{camera} frame {frame:03d} {query}: "
            f"AP={out_row['ap']:.4f} IoU@{fixed_threshold:.2f}="
            f"{out_row[f'iou_at_{fixed_threshold:.2f}']:.4f} "
            f"PSNR={psnr:.2f} SSIM={ssim:.4f}"
        )
    return metric_rows


def parse_statuses(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--statuses", default="accepted,corrected")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("override", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.annotation_dir / "manifest.csv"
    rows = load_annotation_rows(manifest_path, statuses=parse_statuses(args.statuses))
    if not rows:
        raise RuntimeError(f"no annotation rows selected from {manifest_path}")

    config = load_config(args.config_path, args.override)
    trainer = build_trainer(config, args.checkpoint)
    queries = sorted({row["query"] for row in rows})
    text_features = encode_text_features(
        str(trainer.device),
        queries=queries,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )

    metric_rows = evaluate_annotations(
        annotation_dir=args.annotation_dir,
        rows=rows,
        trainer=trainer,
        text_features=text_features,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_amp=args.amp,
        fixed_threshold=args.fixed_threshold,
        save_images=not args.no_save_images,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_annotation_metrics.csv", metric_rows)
    write_csv(args.output_dir / "metrics_by_query.csv", group_metric_rows(metric_rows, keys=["query"]))
    write_csv(args.output_dir / "metrics_overall.csv", group_metric_rows(metric_rows, keys=[]))
    print(f"Wrote metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
