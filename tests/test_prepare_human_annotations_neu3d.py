import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.prepare_human_annotations_neu3d import (
    MANIFEST_FIELDS,
    build_manifest_rows,
    is_binary_mask_array,
    parse_csv_values,
    parse_frame_list,
    write_manifest,
)


class PrepareHumanAnnotationsNeu3DTest(unittest.TestCase):
    def test_parse_frame_list_sorts_deduplicates_and_rejects_negative_ids(self):
        self.assertEqual(parse_frame_list("16,0,8,8"), [0, 8, 16])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_frame_list("0,-1")

    def test_parse_csv_values_strips_blanks_and_rejects_empty_lists(self):
        self.assertEqual(parse_csv_values("human, hand,,glass"), ["human", "hand", "glass"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_csv_values(" , ")

    def test_build_manifest_rows_uses_stable_relative_paths(self):
        rows = build_manifest_rows(
            cameras=["cam00", "cam08"],
            frames=[0, 8],
            queries=["human", "hand"],
            source="grounded_sam2",
        )

        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["image_path"], "images/cam00_frame000.png")
        self.assertEqual(rows[0]["proposal_path"], "proposals/human/cam00_frame000.png")
        self.assertEqual(rows[0]["mask_path"], "masks/human/cam00_frame000.png")
        self.assertEqual(rows[0]["overlay_path"], "overlays/human/cam00_frame000_overlay.png")
        self.assertEqual(rows[0]["query"], "human")
        self.assertEqual(rows[0]["camera"], "cam00")
        self.assertEqual(rows[0]["frame"], "0")
        self.assertEqual(rows[0]["status"], "pending")
        self.assertEqual(rows[0]["source"], "grounded_sam2")

    def test_write_manifest_preserves_expected_header(self):
        rows = build_manifest_rows(
            cameras=["cam00"],
            frames=[0],
            queries=["human"],
            source="grounded_sam2",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.csv"
            write_manifest(manifest_path, rows)

            with manifest_path.open(newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, MANIFEST_FIELDS)
                written_rows = list(reader)

        self.assertEqual(written_rows, rows)

    def test_is_binary_mask_array_accepts_bool_zero_one_and_255_masks(self):
        self.assertTrue(is_binary_mask_array(np.array([[False, True]])))
        self.assertTrue(is_binary_mask_array(np.array([[0, 1]], dtype=np.uint8)))
        self.assertTrue(is_binary_mask_array(np.array([[0, 255]], dtype=np.uint8)))
        self.assertFalse(is_binary_mask_array(np.array([[0, 127, 255]], dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
