import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class SemanticCoarseToFineQueueTest(unittest.TestCase):
    def _load_config(self, path: Path, env: dict[str, str]) -> dict:
        with mock.patch.dict(os.environ, env, clear=False):
            spec = importlib.util.spec_from_file_location("c2f_config", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module.config

    def test_coarse_config_reads_environment_and_uses_single_semantic_scale(self):
        config = self._load_config(
            ROOT / "plenoxels/configs/local/dynerf_cm_semantic_64f_ds16_c2f_coarse.py",
            {
                "SEM_C2F_NAME": "topk8",
                "SEM_RENDER_MODE": "topk_weighted",
                "SEM_TOPK": "8",
                "SEM_SMOOTH_L1_WEIGHT": "0.0",
                "SEM_C2F_COARSE_STEPS": "3000",
                "LOG_ROOT": "/kaggle/temp/c2f_logs",
            },
        )

        self.assertEqual(config["expname"], "cm_semantic_64f_ds16_c2f_topk8_coarse")
        self.assertEqual(config["logdir"], "/kaggle/temp/c2f_logs")
        self.assertEqual(config["num_steps"], 3000)
        self.assertEqual(config["save_every"], 3000)
        self.assertEqual(config["semantic_multiscale_res"], [1])
        self.assertEqual(config["semantic_render_mode"], "topk_weighted")
        self.assertEqual(config["semantic_topk"], 8)
        self.assertEqual(config["semantic_smooth_l1_weight"], 0.0)

    def test_fine_config_adds_fine_semantic_scale_and_skips_optimizer_reload(self):
        config = self._load_config(
            ROOT / "plenoxels/configs/local/dynerf_cm_semantic_64f_ds16_c2f_fine.py",
            {
                "SEM_C2F_NAME": "topk16_smooth010",
                "SEM_RENDER_MODE": "topk_weighted",
                "SEM_TOPK": "16",
                "SEM_SMOOTH_L1_WEIGHT": "0.1",
                "SEM_C2F_FINE_STEPS": "10000",
                "LOG_ROOT": "/kaggle/temp/c2f_logs",
            },
        )

        self.assertEqual(config["expname"], "cm_semantic_64f_ds16_c2f_topk16_smooth010")
        self.assertEqual(config["logdir"], "/kaggle/temp/c2f_logs")
        self.assertEqual(config["num_steps"], 10000)
        self.assertEqual(config["save_every"], 10000)
        self.assertEqual(config["semantic_multiscale_res"], [1, 2])
        self.assertEqual(config["semantic_render_mode"], "topk_weighted")
        self.assertEqual(config["semantic_topk"], 16)
        self.assertEqual(config["semantic_smooth_l1_weight"], 0.1)
        self.assertFalse(config["load_optimizer"])
        self.assertFalse(config["load_scheduler"])

    def test_smoke_queue_runs_expected_c2f_variants(self):
        script = (ROOT / "scripts/kaggle_smoke_semantic_c2f_queue.sh").read_text()

        for name in ["topk8", "topk16", "topk24", "topk16_smooth010"]:
            self.assertIn(name, script)

        self.assertIn("dynerf_cm_semantic_64f_ds16_c2f_coarse.py", script)
        self.assertIn("dynerf_cm_semantic_64f_ds16_c2f_fine.py", script)
        self.assertIn("/kaggle/temp/semantic_c2f_smoke_logs", script)
        self.assertIn("SEM_C2F_COARSE_STEPS=20", script)
        self.assertIn("SEM_C2F_FINE_STEPS=40", script)

    def test_full_queue_resumes_fine_stage_from_coarse_checkpoint_and_uploads(self):
        script = (ROOT / "scripts/kaggle_run_semantic_c2f_queue.sh").read_text()

        self.assertIn("--log-dir", script)
        self.assertIn("cm_semantic_64f_ds16_c2f_${name}_coarse", script)
        self.assertIn("SEM_C2F_COARSE_STEPS=3000", script)
        self.assertIn("SEM_C2F_FINE_STEPS=10000", script)
        self.assertIn("verify_checkpoint", script)
        self.assertIn("upload_dataset", script)
        self.assertIn("kaggle datasets create", script)
        self.assertIn("kaggle datasets version", script)


if __name__ == "__main__":
    unittest.main()
