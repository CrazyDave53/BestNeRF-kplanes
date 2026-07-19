import os

from plenoxels.configs.local.dynerf_cm_semantic_64f_ds16 import config


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


config = dict(config)

ablation_name = os.environ.get("SEM_ABLATION_NAME", "ablation")
config["expname"] = f"cm_semantic_64f_ds16_{ablation_name}"
config["semantic_render_mode"] = os.environ.get("SEM_RENDER_MODE", "full_weighted")
config["semantic_topk"] = _env_int("SEM_TOPK", 24)
config["semantic_smooth_l1_weight"] = _env_float("SEM_SMOOTH_L1_WEIGHT", 0.0)
