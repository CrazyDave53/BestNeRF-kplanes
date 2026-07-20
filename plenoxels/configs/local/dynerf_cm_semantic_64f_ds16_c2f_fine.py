import os

from plenoxels.configs.local.dynerf_cm_semantic_64f_ds16 import config


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


config = dict(config)

c2f_name = os.environ.get("SEM_C2F_NAME", "topk8")
config["logdir"] = os.environ.get("LOG_ROOT", config["logdir"])
config["expname"] = f"cm_semantic_64f_ds16_c2f_{c2f_name}"
config["num_steps"] = _env_int("SEM_C2F_FINE_STEPS", 10000)
config["save_every"] = config["num_steps"]
config["semantic_multiscale_res"] = [1, 2]
config["semantic_render_mode"] = os.environ.get("SEM_RENDER_MODE", "topk_weighted")
config["semantic_topk"] = _env_int("SEM_TOPK", 8)
config["semantic_smooth_l1_weight"] = _env_float("SEM_SMOOTH_L1_WEIGHT", 0.0)
config["load_optimizer"] = False
config["load_scheduler"] = False
