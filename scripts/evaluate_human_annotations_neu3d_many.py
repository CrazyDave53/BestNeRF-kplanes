#!/usr/bin/env python3
"""Evaluate many K-Planes semantic checkpoints while loading Neu3D data once."""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_human_annotations_neu3d import (
    apply_eval_config_to_trainer,
    build_trainer,
    encode_text_features,
    evaluate_annotations,
    group_metric_rows,
    load_annotation_rows,
    load_checkpoint_into_trainer,
    load_config,
    parse_statuses,
    parse_thresholds,
    write_csv,
    write_metric_outputs,
)


@dataclass(frozen=True)
class CheckpointSpec:
    expname: str
    checkpoint: Path
    config_path: Path


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"refusing to extract unsafe tar member: {member.name}")
        tar.extractall(destination)


def slugify(value: str) -> str:
    value = value.lower().replace("_", "-")
    return re.sub(r"[^a-z0-9-]+", "", value)


def extract_archives(search_roots: Sequence[Path], extract_root: Path) -> list[Path]:
    extracted_roots = []
    for root in search_roots:
        for archive in sorted(root.rglob("*.tar.gz")):
            destination = extract_root / slugify(archive.stem.removesuffix(".tar"))
            marker = destination / ".extracted"
            if not marker.exists():
                print(f"Extracting {archive} -> {destination}")
                safe_extract_tar(archive, destination)
                marker.write_text(str(archive))
            extracted_roots.append(destination)
    return extracted_roots


def discover_checkpoint_specs(
    search_roots: Sequence[Path],
    checkpoint_globs: Sequence[str],
    include_pattern: str,
    exclude_pattern: str,
) -> list[CheckpointSpec]:
    checkpoints: set[Path] = set()
    for pattern in checkpoint_globs:
        for path in glob.glob(pattern, recursive=True):
            checkpoint = Path(path)
            if checkpoint.name == "model.pth" and checkpoint.is_file():
                checkpoints.add(checkpoint)
    for root in search_roots:
        if root.exists():
            checkpoints.update(root.rglob("model.pth"))

    specs = []
    for checkpoint in sorted(checkpoints, key=lambda path: str(path)):
        expname = checkpoint.parent.name
        if include_pattern and include_pattern not in expname:
            continue
        if exclude_pattern and exclude_pattern in expname:
            continue
        config_path = checkpoint.parent / "config.py"
        if not config_path.is_file():
            continue
        specs.append(
            CheckpointSpec(
                expname=expname,
                checkpoint=checkpoint,
                config_path=config_path,
            )
        )
    return sorted(specs, key=lambda spec: spec.expname)


def add_experiment(rows: Sequence[dict[str, Any]], experiment: str) -> list[dict[str, Any]]:
    return [{"experiment": experiment, **row} for row in rows]


def write_combined_outputs(output_dir: Path, rows_by_experiment: dict[str, list[dict[str, Any]]]) -> None:
    all_grouped_rows = []
    all_per_annotation_rows = []
    for experiment, metric_rows in rows_by_experiment.items():
        all_per_annotation_rows.extend(add_experiment(metric_rows, experiment))
        grouped = group_metric_rows(metric_rows, keys=["method", "query"])
        all_grouped_rows.extend(add_experiment(grouped, experiment))

    write_csv(output_dir / "all_per_annotation_metrics.csv", all_per_annotation_rows)
    write_csv(output_dir / "all_metrics_by_method_query.csv", all_grouped_rows)

    student_human = [
        row for row in all_grouped_rows
        if row.get("method") == "student_kplanes" and row.get("query") == "human"
    ]
    if student_human:
        student_human.sort(
            key=lambda row: (
                float(row.get("ap", float("nan"))),
                float(row.get("best_iou", float("nan"))),
            ),
            reverse=True,
        )
        write_csv(output_dir / "student_human_summary.csv", student_human)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-glob", action="append", default=[])
    parser.add_argument("--extract-archives-from", type=Path, action="append", default=[])
    parser.add_argument("--extract-root", type=Path, default=Path("/kaggle/working/extracted_semantic_ablation_logs"))
    parser.add_argument("--teacher-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-pattern", default="cm_semantic_64f_ds16")
    parser.add_argument("--exclude-pattern", default="smoke")
    parser.add_argument("--statuses", default="accepted,corrected")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--fixed-thresholds", default=None)
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.annotation_dir / "manifest.csv"
    rows = load_annotation_rows(manifest_path, statuses=parse_statuses(args.statuses))
    if not rows:
        raise RuntimeError(f"no annotation rows selected from {manifest_path}")

    search_roots = list(args.checkpoint_root)
    if args.extract_archives_from:
        search_roots.extend(extract_archives(args.extract_archives_from, args.extract_root))
    specs = discover_checkpoint_specs(
        search_roots=search_roots,
        checkpoint_globs=args.checkpoint_glob,
        include_pattern=args.include_pattern,
        exclude_pattern=args.exclude_pattern,
    )
    if not specs:
        raise RuntimeError("no checkpoints found for multi-evaluation")

    fixed_thresholds = (
        parse_thresholds(args.fixed_thresholds)
        if args.fixed_thresholds is not None
        else [args.fixed_threshold]
    )
    fixed_threshold = fixed_thresholds[0]

    first_config = load_config(str(specs[0].config_path), [])
    trainer = build_trainer(first_config, str(specs[0].checkpoint))
    queries = sorted({row["query"] for row in rows})
    text_features = encode_text_features(
        str(trainer.device),
        queries=queries,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )

    rows_by_experiment: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        print(f"==== evaluating {spec.expname} ====")
        config = load_config(str(spec.config_path), [])
        apply_eval_config_to_trainer(trainer, config)
        load_checkpoint_into_trainer(trainer, str(spec.checkpoint))
        metric_rows = evaluate_annotations(
            annotation_dir=args.annotation_dir,
            rows=rows,
            trainer=trainer,
            text_features=text_features,
            teacher_cache_dir=args.teacher_cache_dir,
            output_dir=args.output_dir / spec.expname,
            batch_size=args.batch_size,
            use_amp=args.amp,
            fixed_threshold=fixed_threshold,
            fixed_thresholds=fixed_thresholds,
            save_images=not args.no_save_images,
        )
        write_metric_outputs(args.output_dir / spec.expname, metric_rows)
        rows_by_experiment[spec.expname] = metric_rows

    write_combined_outputs(args.output_dir, rows_by_experiment)
    print(f"Wrote combined metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
