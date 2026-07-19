import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.propose_human_masks_grounded_sam2 import (
    create_overlay,
    format_grounding_prompt,
    load_manifest_rows,
    merge_masks,
    save_binary_mask,
)


class ProposeHumanMasksGroundedSam2Test(unittest.TestCase):
    def test_format_grounding_prompt_lowercases_and_adds_period(self):
        self.assertEqual(format_grounding_prompt(" Human "), "human.")
        self.assertEqual(format_grounding_prompt("left hand."), "left hand.")

    def test_merge_masks_unions_or_selects_largest_mask(self):
        masks = np.array([
            [[True, False], [False, False]],
            [[False, True], [True, True]],
        ])

        np.testing.assert_array_equal(
            merge_masks(masks, mode="union"),
            np.array([[True, True], [True, True]]),
        )
        np.testing.assert_array_equal(
            merge_masks(masks, mode="largest"),
            np.array([[False, True], [True, True]]),
        )

    def test_merge_masks_returns_empty_mask_for_no_detections(self):
        merged = merge_masks(np.zeros((0, 3, 4), dtype=bool), mode="union", image_shape=(3, 4))

        self.assertEqual(merged.shape, (3, 4))
        self.assertFalse(merged.any())

    def test_save_binary_mask_writes_zero_and_255_png(self):
        mask = np.array([[False, True], [True, False]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mask.png"
            save_binary_mask(path, mask)
            loaded = np.array(Image.open(path))

        self.assertEqual(loaded.dtype, np.uint8)
        self.assertEqual(set(np.unique(loaded).tolist()), {0, 255})

    def test_create_overlay_tints_foreground_pixels(self):
        image = Image.new("RGB", (2, 1), color=(100, 100, 100))
        mask = np.array([[False, True]])

        overlay = np.array(create_overlay(image, mask, boxes=[]))

        self.assertEqual(overlay[0, 0].tolist(), [100, 100, 100])
        self.assertNotEqual(overlay[0, 1].tolist(), [100, 100, 100])

    def test_load_manifest_rows_filters_statuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["image_path", "proposal_path", "mask_path", "overlay_path",
                                "query", "camera", "frame", "status", "source", "reviewer", "notes"],
                )
                writer.writeheader()
                writer.writerow({
                    "image_path": "images/a.png",
                    "proposal_path": "proposals/human/a.png",
                    "mask_path": "masks/human/a.png",
                    "overlay_path": "overlays/human/a.png",
                    "query": "human",
                    "camera": "cam01",
                    "frame": "0",
                    "status": "pending",
                    "source": "grounded_sam2",
                    "reviewer": "",
                    "notes": "",
                })
                writer.writerow({
                    "image_path": "images/b.png",
                    "proposal_path": "proposals/human/b.png",
                    "mask_path": "masks/human/b.png",
                    "overlay_path": "overlays/human/b.png",
                    "query": "human",
                    "camera": "cam01",
                    "frame": "8",
                    "status": "accepted",
                    "source": "grounded_sam2",
                    "reviewer": "",
                    "notes": "",
                })

            rows = load_manifest_rows(path, statuses={"pending"})

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_path"], "images/a.png")


if __name__ == "__main__":
    unittest.main()
