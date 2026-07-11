from plenoxels.configs.local.dynerf_cm_rgb_real_64f_10k import config

config = dict(config)

config["expname"] = "cm_semantic_64f_ds16"
config["data_downsample"] = 4
config["batch_size"] = 1024
config["num_steps"] = 10000
config["save_every"] = 10000
config["valid_every"] = -1

config["semantic_enabled"] = True
config["train_rgb"] = True
config["train_semantic"] = True
config["freeze_rgb_for_semantic"] = False
config["semantic_detach_geometry"] = True
config["semantic_feature_dim"] = 768
config["semantic_loss_weight"] = 0.1
config["openseg_cache_dir"] = "/kaggle/input/coffee-martini-openseg-ds16-64f"

config["semantic_grid_config"] = [{
    "grid_dimensions": 2,
    "input_coordinate_dim": 4,
    "output_coordinate_dim": 8,
    "resolution": [64, 64, 64, 64],
}]
config["semantic_multiscale_res"] = [1, 2]
config["semantic_linear_decoder_layers"] = 1
