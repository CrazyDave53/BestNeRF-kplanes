config = {
    "expname": "cm_baseline_smoke",
    "logdir": "./logs/baseline",
    "device": "cuda:0",

    # Coffee Martini / DyNeRF-style data.
    "data_downsample": 2,
    "data_dirs": ["data/neu3d/coffee_martini"],
    "contract": False,
    "ndc": True,
    "ndc_far": 2.6,
    "near_scaling": 0.9,
    "isg": False,
    "isg_step": -1,
    "ist_step": -1,
    "keyframes": False,
    "scene_bbox": [[-3.0, -1.8, -1.2], [3.0, 1.8, 1.2]],

    # Very short smoke run.
    "num_steps": 20,
    "batch_size": 128,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr": 0.01,

    # Keep regularization close to the intended Coffee Martini run.
    "distortion_loss_weight": 0.001,
    "histogram_loss_weight": 1.0,
    "l1_time_planes": 0.0001,
    "l1_time_planes_proposal_net": 0.0001,
    "plane_tv_weight": 0.0002,
    "plane_tv_weight_proposal_net": 0.0002,
    "time_smoothness_weight": 0.001,
    "time_smoothness_weight_proposal_net": 1e-5,

    "save_every": 20,
    "valid_every": -1,
    "save_outputs": False,
    "train_fp16": True,

    "max_train_cameras": 1,
    "max_train_tsteps": 4,
    "max_test_cameras": 1,
    "max_test_tsteps": 2,

    # Raymarching settings.
    "single_jitter": False,
    "num_samples": 16,
    "num_proposal_samples": [32, 16],
    "num_proposal_iterations": 2,
    "use_same_proposal_network": False,
    "use_proposal_weight_anneal": True,
    "proposal_net_args_list": [
        {"num_input_coords": 4, "num_output_coords": 8, "resolution": [16, 16, 16, 16]},
        {"num_input_coords": 4, "num_output_coords": 8, "resolution": [16, 16, 16, 16]},
    ],

    # RGB K-Planes settings from the intended M1.2/M4 setup.
    "concat_features_across_scales": True,
    "density_activation": "trunc_exp",
    "linear_decoder": True,
    "linear_decoder_layers": 1,
    "multiscale_res": [1, 2],
    "grid_config": [{
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 8,
        "resolution": [16, 16, 16, 16],
    }],
}
