from plenoxels.configs.local.dynerf_cm_semantic_64f_ds16 import config

config = dict(config)

config["expname"] = "cm_semantic_64f_ds16_topk24_smoothl1"
config["semantic_render_mode"] = "topk_weighted"
config["semantic_topk"] = 24
config["semantic_smooth_l1_weight"] = 0.1
