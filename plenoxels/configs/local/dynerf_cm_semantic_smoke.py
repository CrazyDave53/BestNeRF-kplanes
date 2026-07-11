from plenoxels.configs.local.dynerf_cm_baseline_smoke import config

config = dict(config)

config["expname"] = "cm_semantic_smoke"
config["data_downsample"] = 4
config["batch_size"] = 64
config["num_steps"] = 20
config["save_every"] = 20
config["max_train_cameras"] = 1
config["max_train_tsteps"] = 4
config["max_test_cameras"] = 1
config["max_test_tsteps"] = 2

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
    "resolution": [16, 16, 16, 16],
}]
config["semantic_multiscale_res"] = [1]
config["semantic_linear_decoder_layers"] = 1
