from plenoxels.configs.local.dynerf_cm_semantic_64f_ds16 import config

config = dict(config)

config["expname"] = "cm_semantic_64f_ds16_topk24"
config["semantic_render_mode"] = "topk_weighted"
config["semantic_topk"] = 24
