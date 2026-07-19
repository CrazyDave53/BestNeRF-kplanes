import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class SemanticAblationQueueTest(unittest.TestCase):
    def _load_config(self, path: Path, env: dict[str, str]) -> dict:
        with mock.patch.dict(os.environ, env, clear=False):
            spec = importlib.util.spec_from_file_location("ablation_config", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module.config

    def test_full_ablation_config_reads_environment(self):
        config = self._load_config(
            ROOT / "plenoxels/configs/local/dynerf_cm_semantic_64f_ds16_ablation.py",
            {
                "SEM_ABLATION_NAME": "topk16_smooth010",
                "SEM_RENDER_MODE": "topk_weighted",
                "SEM_TOPK": "16",
                "SEM_SMOOTH_L1_WEIGHT": "0.1",
            },
        )

        self.assertEqual(config["expname"], "cm_semantic_64f_ds16_topk16_smooth010")
        self.assertEqual(config["semantic_render_mode"], "topk_weighted")
        self.assertEqual(config["semantic_topk"], 16)
        self.assertEqual(config["semantic_smooth_l1_weight"], 0.1)

    def test_smoke_ablation_config_reads_environment(self):
        config = self._load_config(
            ROOT / "plenoxels/configs/local/dynerf_cm_semantic_smoke_ablation.py",
            {
                "SEM_ABLATION_NAME": "smooth005",
                "SEM_RENDER_MODE": "full_weighted",
                "SEM_TOPK": "24",
                "SEM_SMOOTH_L1_WEIGHT": "0.05",
            },
        )

        self.assertEqual(config["expname"], "cm_semantic_smoke_smooth005")
        self.assertEqual(config["num_steps"], 20)
        self.assertEqual(config["semantic_render_mode"], "full_weighted")
        self.assertEqual(config["semantic_smooth_l1_weight"], 0.05)

    def test_smoke_queue_contains_expected_ablation_grid(self):
        script = (ROOT / "scripts/kaggle_smoke_semantic_ablation_queue.sh").read_text()

        for name in [
            "smooth005",
            "smooth010",
            "smooth020",
            "topk1",
            "topk8",
            "topk16",
            "topk24",
            "topk32",
            "topk48",
            "topk8_smooth010",
            "topk16_smooth010",
            "topk24_smooth010",
            "topk32_smooth010",
            "topk48_smooth010",
            "topk24_smooth005",
            "topk24_smooth020",
        ]:
            self.assertIn(name, script)

        self.assertIn("dynerf_cm_semantic_smoke_ablation.py", script)

    def test_full_queue_uploads_after_each_checkpoint(self):
        script = (ROOT / "scripts/kaggle_run_semantic_ablation_queue.sh").read_text()

        self.assertIn("kaggle datasets create", script)
        self.assertIn("kaggle datasets version", script)
        self.assertIn("verify_checkpoint", script)
        self.assertIn("upload_dataset", script)
        self.assertIn("dynerf_cm_semantic_64f_ds16_ablation.py", script)


if __name__ == "__main__":
    unittest.main()
