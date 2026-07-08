from plenoxels.configs.final.DyNeRF.dynerf_hybrid import config

config = dict(config)

config["expname"] = "cm_rgb_full"
config["logdir"] = "./logs/baseline"
config["device"] = "cuda:0"

config["data_dirs"] = ["data/neu3d/coffee_martini"]
config["data_downsample"] = 2
config["contract"] = False
config["ndc"] = True
config["ndc_far"] = 2.6
config["near_scaling"] = 0.9
config["scene_bbox"] = [[-3.0, -1.8, -1.2], [3.0, 1.8, 1.2]]

config["isg"] = False
config["isg_step"] = -1
config["ist_step"] = -1
config["keyframes"] = False

config["max_train_cameras"] = None
config["max_train_tsteps"] = None
config["max_test_cameras"] = 1
config["max_test_tsteps"] = 8

config["num_steps"] = 30001
config["batch_size"] = 4096
config["save_every"] = 10000
config["valid_every"] = -1
config["save_outputs"] = False
config["train_fp16"] = True

config["optim_type"] = "adam"
config["scheduler_type"] = "warmup_cosine"
config["lr"] = 0.01

