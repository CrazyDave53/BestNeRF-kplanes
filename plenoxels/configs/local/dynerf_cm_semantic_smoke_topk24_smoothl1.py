from plenoxels.configs.local.dynerf_cm_semantic_smoke import config

config = dict(config)

config["expname"] = "cm_semantic_smoke_topk24_smoothl1"
config["semantic_render_mode"] = "topk_weighted"
config["semantic_topk"] = 24
config["semantic_smooth_l1_weight"] = 0.1
