from pathlib import Path
import importlib
import json

LR_ARGS_OVERWRITE = ['feature_lr', 'opacity_lr', 'scaling_lr', 'delta_position_lr']
BACKGROUND_ARGS_OVERWRITE = ['static_background', 'no_background', 'render_background_probability', 'pretrained_background']
LOSS_ARGS_OVERWRITE = ['isometry_loss_weight', 'chamfer_loss_weight',
                       'photometric_loss_weight', 'tracking_loss_weight', 'segmentation_loss_weight',
                       'velocity_smoothing_loss_weight', 'depthmap_loss_weight', 'scaling_loss_weight',
                       'instance_isometry_loss_weight', 'lpips_loss_weight', 'tracking_depth_loss_weight']
ALGORITHM_ARGS_OVERWRITE = ['number_of_gaussians', 'frame_transport_dropout', 'supervision_downscale', 'prune_points',
                           'freeze_previous_in_motion_estimation',  'freeze_frames_of_origin', 'downsample_reducescale']
ISOMETRY_ARGS_OVERWRITE = ['isometry_knn', 'isometry_knn_radius', 'isometry_per_segment', 'isometry_use_l2',
                           'isometry_weight_background_factor']
CHAMFER_ARGS_OVERWRITE = ['chamfer_agg_group_ratio']
TRACKING_ARGS_OVERWRITE = ['tracking_knn', 'tracking_radius',  'tracking_loss_per_segment', 'tracking_window']
RIGIDITY_ARGS_OVERWRITE = ['instance_isometry_numpairs']

# get argument overwrite for each part of model
MODEL_ARGS_OVERWRITE = LR_ARGS_OVERWRITE + BACKGROUND_ARGS_OVERWRITE + LOSS_ARGS_OVERWRITE + ALGORITHM_ARGS_OVERWRITE +\
                 ISOMETRY_ARGS_OVERWRITE + CHAMFER_ARGS_OVERWRITE + TRACKING_ARGS_OVERWRITE + RIGIDITY_ARGS_OVERWRITE
DATA_ARGS_OVERWRITE = ['depth_remove_outliers', 'outlier_std_ratio']
PIPELINE_ARGS_OVERWRITE = []
DATAPARSER_ARGS_OVERWRITE = ['downscale_factor']


def load_model_config(args):
    mod = importlib.import_module(args.model_config)
    train_config = mod.train_config

    # override defaults with any input arguments specified
    train_config.data = Path(args.data_dir)
    train_config.pipeline.datamanager.data = Path(args.data_dir)
    train_config.load_dir = Path(args.load_dir) if args.load_dir else None
    train_config.output_dir = Path(args.output)

    # compute max number of iterations based on specified stages
    with open(str(train_config.data / 'dataset.json'), 'r') as f:
        dataset = json.load(f)
        num_frames = dataset['num_exemplars']
    stages = train_config.pipeline.stages
    total_iters = 0
    for stage in stages:
        total_iters += stage.get('steps', 0) * num_frames
    print("Setting max iters to %s" % total_iters)
    train_config.max_num_iterations = total_iters

    # Copy model config parameters from input args
    for argument in MODEL_ARGS_OVERWRITE:
        if hasattr(args, argument) and getattr(args, argument) is not None:
            print("Overwriting Model Argument %s with %s!" % (argument, getattr(args, argument)))
            setattr(train_config.pipeline.model, argument, getattr(args, argument))

    # Copy pipeline config parameters
    for argument in PIPELINE_ARGS_OVERWRITE:
        if hasattr(args, argument) and getattr(args, argument) is not None:
            print("Overwriting Pipeline Argument %s with %s!" % (argument, getattr(args, argument)))
            setattr(train_config.pipeline, argument, getattr(args, argument))

    return train_config


def load_data_args(train_config, args):
    for argument in DATA_ARGS_OVERWRITE:
        if hasattr(args, argument) and getattr(args, argument) is not None:
            print("Overwriting Data Argument %s with %s!" % (argument, getattr(args, argument)))
            setattr(train_config.pipeline.datamanager, argument, getattr(args, argument))

    for argument in DATAPARSER_ARGS_OVERWRITE:
        if hasattr(args, argument) and getattr(args, argument) is not None:
            print("Overwriting DataParser Argument %s with %s!" % (argument, getattr(args, argument)))
            setattr(train_config.pipeline.datamanager.dataparser, argument, getattr(args, argument))

    return train_config