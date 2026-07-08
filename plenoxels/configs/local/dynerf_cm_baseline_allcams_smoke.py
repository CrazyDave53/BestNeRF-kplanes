from plenoxels.configs.local.dynerf_cm_baseline_smoke import config

config = dict(config)
config["expname"] = "cm_baseline_smoke_allcams"
config["max_train_cameras"] = None
config["max_train_tsteps"] = 4
config["max_test_cameras"] = 1
config["max_test_tsteps"] = 2
config["num_steps"] = 20
config["batch_size"] = 128
config["save_every"] = 20
config["valid_every"] = -1
config["save_outputs"] = False

