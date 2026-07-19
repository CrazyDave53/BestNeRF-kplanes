import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_human_annotations_neu3d import (
    average_precision,
    best_iou,
    binary_mask_from_image,
    compute_mask_metrics,
    compute_psnr,
    compute_ssim,
    compute_teacher_scores,
    display_heatmap_for_rgb,
    group_metric_rows,
    load_annotation_rows,
    make_metric_row,
    resolve_teacher_cache_root,
)


class EvaluateHumanAnnotationsNeu3DTest(unittest.TestCase):
    def test_binary_mask_from_image_treats_nonzero_as_foreground(self):
        mask = binary_mask_from_image(np.array([[0, 1], [255, 0]], dtype=np.uint8))

        np.testing.assert_array_equal(mask, np.array([[False, True], [True, False]]))

    def test_average_precision_ranks_positive_pixels(self):
        target = np.array([True, False, True, False])
        scores = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)

        self.assertAlmostEqual(average_precision(scores, target), (1.0 + 2.0 / 3.0) / 2.0)

    def test_compute_mask_metrics_reports_threshold_and_best_iou(self):
        target = np.array([[True, False], [True, False]])
        scores = np.array([[0.9, 0.6], [0.7, 0.1]], dtype=np.float32)

        metrics = compute_mask_metrics(scores, target, fixed_threshold=0.5)

        self.assertAlmostEqual(metrics["iou_at_0.50"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["precision_at_0.50"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["recall_at_0.50"], 1.0)
        self.assertAlmostEqual(metrics["f1_at_0.50"], 0.8)
        self.assertAlmostEqual(metrics["best_iou"], 1.0)
        self.assertAlmostEqual(metrics["best_threshold"], 0.61)

    def test_best_iou_handles_empty_masks(self):
        scores = np.zeros((2, 2), dtype=np.float32)
        target = np.zeros((2, 2), dtype=bool)

        value, threshold = best_iou(scores, target, thresholds=[0.5])

        self.assertEqual(value, 1.0)
        self.assertEqual(threshold, 0.5)

    def test_compute_psnr_returns_infinite_for_identical_images(self):
        image = np.zeros((2, 2, 3), dtype=np.float32)

        self.assertTrue(np.isinf(compute_psnr(image, image)))

    def test_compute_ssim_returns_one_for_identical_images(self):
        image = np.full((16, 16, 3), 0.5, dtype=np.float32)

        self.assertAlmostEqual(compute_ssim(image, image), 1.0)

    def test_display_heatmap_for_rgb_resizes_mask_sized_heatmap_to_rgb_shape(self):
        heatmap = np.zeros((4, 6), dtype=np.float32)
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)

        resized = display_heatmap_for_rgb(heatmap, rgb)

        self.assertEqual(resized.shape, (2, 3))

    def test_load_annotation_rows_keeps_accepted_and_corrected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.csv"
            with manifest.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["image_path", "proposal_path", "mask_path", "overlay_path",
                                "query", "camera", "frame", "status", "source", "reviewer", "notes"],
                )
                writer.writeheader()
                for status in ("pending", "accepted", "corrected", "rejected"):
                    writer.writerow({
                        "image_path": "images/a.png",
                        "proposal_path": "proposals/human/a.png",
                        "mask_path": "masks/human/a.png",
                        "overlay_path": "overlays/human/a.png",
                        "query": status,
                        "camera": "cam01",
                        "frame": "0",
                        "status": status,
                        "source": "grounded_sam2",
                        "reviewer": "",
                        "notes": "",
                    })

            rows = load_annotation_rows(manifest, statuses={"accepted", "corrected"})

        self.assertEqual([row["query"] for row in rows], ["accepted", "corrected"])

    def test_group_metric_rows_averages_numeric_values_by_query(self):
        rows = [
            {"method": "student_kplanes", "query": "human", "frame": 0, "ap": 0.5, "psnr": 20.0},
            {"method": "student_kplanes", "query": "human", "frame": 8, "ap": 1.0, "psnr": 30.0},
            {"method": "teacher_openseg", "query": "human", "frame": 0, "ap": 0.25, "psnr": np.nan},
        ]

        grouped = group_metric_rows(rows, keys=["method", "query"])
        by_method = {row["method"]: row for row in grouped}

        self.assertEqual(by_method["teacher_openseg"]["count"], 1)
        self.assertNotIn("frame", by_method["teacher_openseg"])
        self.assertAlmostEqual(by_method["teacher_openseg"]["ap"], 0.25)
        self.assertTrue(np.isnan(by_method["teacher_openseg"]["psnr"]))
        self.assertEqual(by_method["student_kplanes"]["count"], 2)
        self.assertAlmostEqual(by_method["student_kplanes"]["ap"], 0.75)
        self.assertAlmostEqual(by_method["student_kplanes"]["psnr"], 25.0)

    def test_compute_teacher_scores_uses_normalized_dot_product(self):
        shard = np.zeros((1, 2, 2, 3), dtype=np.float16)
        shard[0, 0, 0] = [1, 0, 0]
        shard[0, 0, 1] = [0, 1, 0]
        shard[0, 1, 0] = [0, 0, 2]
        shard[0, 1, 1] = [1, 1, 0]

        scores = compute_teacher_scores(
            shard=shard,
            frame_id=0,
            text_feature=np.array([1, 0, 0], dtype=np.float32),
        )

        np.testing.assert_allclose(scores, [[1.0, 0.0], [0.0, 2 ** -0.5]], atol=1e-4)

    def test_resolve_teacher_cache_root_finds_nested_kaggle_dataset_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "datasets" / "lenguyenminhchau" / "coffee-martini-openseg-ds16-64f"
            nested.mkdir(parents=True)
            np.save(nested / "cam01.npy", np.zeros((1, 1, 1, 3), dtype=np.float16))

            resolved = resolve_teacher_cache_root(root, camera="cam01")

        self.assertEqual(resolved, nested)

    def test_make_metric_row_records_method_name(self):
        row = {"query": "human", "camera": "cam01", "frame": "0", "status": "accepted",
               "mask_path": "masks/human/cam01_frame000.png"}
        mask = np.array([[True, False], [False, False]])
        scores = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)

        out = make_metric_row(
            source_row=row,
            method="teacher_openseg",
            heatmap=scores,
            mask=mask,
            fixed_threshold=0.5,
            psnr=np.nan,
            ssim=np.nan,
        )

        self.assertEqual(out["method"], "teacher_openseg")
        self.assertEqual(out["query"], "human")
        self.assertAlmostEqual(out["ap"], 1.0)


if __name__ == "__main__":
    unittest.main()
