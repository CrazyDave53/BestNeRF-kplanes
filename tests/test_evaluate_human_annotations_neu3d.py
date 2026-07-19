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
    group_metric_rows,
    load_annotation_rows,
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
            {"query": "human", "frame": 0, "ap": 0.5, "psnr": 20.0},
            {"query": "human", "frame": 8, "ap": 1.0, "psnr": 30.0},
            {"query": "hand", "frame": 0, "ap": 0.25, "psnr": 10.0},
        ]

        grouped = group_metric_rows(rows, keys=["query"])

        self.assertEqual(grouped[0]["query"], "hand")
        self.assertEqual(grouped[0]["count"], 1)
        self.assertNotIn("frame", grouped[0])
        self.assertAlmostEqual(grouped[1]["ap"], 0.75)
        self.assertAlmostEqual(grouped[1]["psnr"], 25.0)


if __name__ == "__main__":
    unittest.main()
