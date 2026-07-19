"""Generate model mask proposals for human-annotation manifest rows.

This script follows the Grounded-SAM2 image pipeline:

1. GroundingDINO predicts text-conditioned boxes.
2. SAM2 predicts masks from those boxes.
3. Masks are merged, saved as binary proposals, and visualized as overlays.

The proposals are not final annotations. A human should review each overlay and
copy or edit accepted masks into `masks/<query>/`, then update `manifest.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


def format_grounding_prompt(query: str) -> str:
    prompt = query.strip().lower()
    if not prompt:
        raise ValueError("query must not be empty")
    if not prompt.endswith("."):
        prompt += "."
    return prompt


def load_manifest_rows(path: Path, statuses: set[str] | None = None) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if statuses is None:
        return rows
    return [row for row in rows if row.get("status", "") in statuses]


def merge_masks(
    masks: np.ndarray,
    mode: str,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    if masks.size == 0 or masks.shape[0] == 0:
        if image_shape is None:
            raise ValueError("image_shape is required when merging an empty mask set")
        return np.zeros(image_shape, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"expected masks with shape [N,H,W], got {masks.shape}")
    if mode == "union":
        return masks.any(axis=0)
    if mode == "largest":
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        return masks[int(areas.argmax())]
    raise ValueError("--mask-merge must be one of: union, largest")


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(path)


def create_overlay(
    image: Image.Image,
    mask: np.ndarray,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[str] | None = None,
    alpha: float = 0.45,
) -> Image.Image:
    base = image.convert("RGB")
    overlay = np.array(base).astype(np.float32)
    mask = np.asarray(mask, dtype=bool)
    color = np.array([255, 40, 40], dtype=np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(result)
    labels = labels or [""] * len(boxes)
    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 80), width=2)
        if label:
            draw.text((x0 + 2, y0 + 2), label, fill=(0, 255, 80))
    return result


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def normalize_sam2_config(
    sam2_config: str,
    grounded_sam2_root: Path | None,
) -> str:
    is_posix_absolute = sam2_config.startswith("/")
    config_path = Path(sam2_config)
    if not config_path.is_absolute() and not is_posix_absolute:
        return sam2_config.replace("\\", "/")
    if grounded_sam2_root is None:
        raise ValueError(
            "absolute --sam2-config requires --grounded-sam2-root so it can "
            "be converted to the Hydra config name expected by SAM2"
        )
    if is_posix_absolute:
        root_text = str(grounded_sam2_root).replace("\\", "/").rstrip("/")
        config_text = sam2_config.replace("\\", "/")
        prefix = root_text + "/"
        if not config_text.startswith(prefix):
            raise ValueError(
                "absolute --sam2-config must be inside --grounded-sam2-root; "
                "SAM2 expects a config name like configs/sam2.1/sam2.1_hiera_l.yaml"
            )
        return config_text[len(prefix):]
    try:
        relative = config_path.resolve().relative_to(grounded_sam2_root.resolve())
    except ValueError as exc:
        raise ValueError(
            "absolute --sam2-config must be inside --grounded-sam2-root; "
            "SAM2 expects a config name like configs/sam2.1/sam2.1_hiera_l.yaml"
        ) from exc
    return relative.as_posix()


def load_grounded_sam2(
    grounded_sam2_root: Path | None,
    sam2_config: str,
    sam2_checkpoint: Path,
    grounding_model_id: str,
    device: str,
):
    if grounded_sam2_root is not None:
        sys.path.insert(0, str(grounded_sam2_root))

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    sam2_config_name = normalize_sam2_config(sam2_config, grounded_sam2_root)
    sam2_model = build_sam2(sam2_config_name, str(sam2_checkpoint), device=device)
    image_predictor = SAM2ImagePredictor(sam2_model)
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        grounding_model_id
    ).to(device)
    grounding_model.eval()
    return torch, processor, grounding_model, image_predictor


def predict_masks_for_image(
    image: Image.Image,
    query: str,
    torch_module,
    processor,
    grounding_model,
    image_predictor,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, list[list[float]], list[float], list[str]]:
    prompt = format_grounding_prompt(query)
    image_rgb = image.convert("RGB")
    inputs = processor(images=image_rgb, text=prompt, return_tensors="pt").to(device)

    with torch_module.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_rgb.size[::-1]],
    )[0]

    boxes_tensor = results["boxes"]
    boxes = boxes_tensor.detach().cpu().numpy()
    scores = results["scores"].detach().cpu().numpy().tolist()
    labels = [str(label) for label in results["labels"]]
    if boxes.shape[0] == 0:
        height, width = image_rgb.size[1], image_rgb.size[0]
        return np.zeros((0, height, width), dtype=bool), [], scores, labels

    image_predictor.set_image(np.array(image_rgb))
    masks, _, _ = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
    )
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    elif masks.ndim == 2:
        masks = masks[None]
    return masks.astype(bool), boxes.tolist(), scores, labels


def process_rows(
    annotation_dir: Path,
    rows: Iterable[dict[str, str]],
    torch_module,
    processor,
    grounding_model,
    image_predictor,
    device: str,
    box_threshold: float,
    text_threshold: float,
    mask_merge: str,
    overwrite: bool,
) -> int:
    count = 0
    for row in rows:
        image_path = annotation_dir / row["image_path"]
        proposal_path = annotation_dir / row["proposal_path"]
        overlay_path = annotation_dir / row["overlay_path"]
        metadata_path = proposal_path.with_suffix(".json")
        if proposal_path.exists() and overlay_path.exists() and not overwrite:
            continue

        image = Image.open(image_path).convert("RGB")
        masks, boxes, scores, labels = predict_masks_for_image(
            image=image,
            query=row["query"],
            torch_module=torch_module,
            processor=processor,
            grounding_model=grounding_model,
            image_predictor=image_predictor,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        merged = merge_masks(
            masks,
            mode=mask_merge,
            image_shape=(image.size[1], image.size[0]),
        )

        save_binary_mask(proposal_path, merged)
        overlay = create_overlay(image, merged, boxes=boxes, labels=labels)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path)
        save_json(metadata_path, {
            "query": row["query"],
            "prompt": format_grounding_prompt(row["query"]),
            "image_path": row["image_path"],
            "proposal_path": row["proposal_path"],
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "mask_merge": mask_merge,
        })
        count += 1
        print(f"Wrote proposal: {proposal_path}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--grounded-sam2-root", type=Path, default=None)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.30)
    parser.add_argument("--mask-merge", choices=("union", "largest"), default="union")
    parser.add_argument("--statuses", default="pending")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.annotation_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    statuses = set(item.strip() for item in args.statuses.split(",") if item.strip())
    rows = load_manifest_rows(manifest_path, statuses=statuses or None)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        print("No manifest rows selected.")
        return

    torch_module, processor, grounding_model, image_predictor = load_grounded_sam2(
        grounded_sam2_root=args.grounded_sam2_root,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        grounding_model_id=args.grounding_model_id,
        device=args.device,
    )

    written = process_rows(
        annotation_dir=args.annotation_dir,
        rows=rows,
        torch_module=torch_module,
        processor=processor,
        grounding_model=grounding_model,
        image_predictor=image_predictor,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        mask_merge=args.mask_merge,
        overwrite=args.overwrite,
    )
    print(f"Wrote {written} proposal masks.")


if __name__ == "__main__":
    main()
