import csv
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_human_annotations_neu3d_many import (
    CheckpointSpec,
    discover_checkpoint_specs,
    extract_archives,
    write_combined_outputs,
)


class EvaluateHumanAnnotationsNeu3DManyTest(unittest.TestCase):
    def _write_checkpoint_dir(self, root: Path, expname: str) -> Path:
        exp_dir = root / expname
        exp_dir.mkdir(parents=True)
        (exp_dir / "model.pth").write_bytes(b"checkpoint")
        (exp_dir / "config.py").write_text("config = {}\n")
        return exp_dir

    def test_discover_checkpoint_specs_filters_smoke_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_checkpoint_dir(root, "cm_semantic_64f_ds16_topk8")
            self._write_checkpoint_dir(root, "cm_semantic_smoke_topk8")
            self._write_checkpoint_dir(root, "other_experiment")

            specs = discover_checkpoint_specs(
                search_roots=[root],
                checkpoint_globs=[],
                include_pattern="cm_semantic_64f_ds16",
                exclude_pattern="smoke",
            )

        self.assertEqual([spec.expname for spec in specs], ["cm_semantic_64f_ds16_topk8"])

    def test_extract_archives_returns_extracted_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_source = root / "source"
            exp_dir = self._write_checkpoint_dir(archive_source, "cm_semantic_64f_ds16_topk16")
            archive_path = root / "cm-sem-ablation-topk16.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(exp_dir, arcname=exp_dir.name)

            extracted = extract_archives([root], root / "extract")
            specs = discover_checkpoint_specs(
                search_roots=extracted,
                checkpoint_globs=[],
                include_pattern="cm_semantic_64f_ds16",
                exclude_pattern="smoke",
            )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].expname, "cm_semantic_64f_ds16_topk16")

    def test_write_combined_outputs_writes_ranked_student_human_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            rows_by_experiment = {
                "exp_b": [
                    {
                        "method": "student_kplanes",
                        "query": "human",
                        "camera": "cam01",
                        "frame": 0,
                        "status": "accepted",
                        "mask_path": "mask.png",
                        "ap": 0.9,
                        "best_iou": 0.8,
                        "best_threshold": 0.5,
                        "mask_pixels": 1.0,
                        "psnr": 30.0,
                        "ssim": 0.9,
                    }
                ],
                "exp_a": [
                    {
                        "method": "student_kplanes",
                        "query": "human",
                        "camera": "cam01",
                        "frame": 0,
                        "status": "accepted",
                        "mask_path": "mask.png",
                        "ap": 0.95,
                        "best_iou": 0.7,
                        "best_threshold": 0.5,
                        "mask_pixels": 1.0,
                        "psnr": 29.0,
                        "ssim": 0.8,
                    }
                ],
            }

            write_combined_outputs(output_dir, rows_by_experiment)
            with (output_dir / "student_human_summary.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual([row["experiment"] for row in rows], ["exp_a", "exp_b"])


if __name__ == "__main__":
    unittest.main()
