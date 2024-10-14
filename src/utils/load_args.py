'''
argparse options.
'''
import argparse
import yaml
import json


def get_argparse_input():

    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Input/Output Parameters
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True,
                        help='Model config as a python path. Eg "configs.model_configs.xxx')
    parser.add_argument('--output', type=str, default="./out")
    parser.add_argument('--only_render', type=str, default='False')
    parser.add_argument('--only_interactive_visualize', type=str, default='False')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--_wandb', type=str)  # for preemptive jobs

    # ---------------------------- Model Parameters --------------------------------
    # learning rates
    parser.add_argument('--delta_position_lr', type=float)

    # training schedule
    parser.add_argument('--number_of_gaussians', type=int)
    parser.add_argument('--frame_transport_dropout', type=float)

    # background parameters
    parser.add_argument('--static_background', type=str)
    parser.add_argument('--no_background', type=str)
    parser.add_argument('--render_background_probability', type=str)

    # loss weights
    parser.add_argument('--isometry_loss_weight', type=float)
    parser.add_argument('--chamfer_loss_weight', type=float)
    parser.add_argument('--photometric_loss_weight', type=float)
    parser.add_argument('--tracking_loss_weight', type=float)
    parser.add_argument('--segmentation_loss_weight', type=float)
    parser.add_argument('--velocity_smoothing_loss_weight', type=float)
    parser.add_argument('--depthmap_loss_weight', type=float)
    parser.add_argument('--instance_isometry_loss_weight', type=float)
    parser.add_argument('--scaling_loss_weight', type=float)
    parser.add_argument('--lpips_loss_weight', type=float)

    # training algorithm hyperparameters
    parser.add_argument('--supervision_downscale', type=int)
    parser.add_argument('--freeze_previous_in_motion_estimation', type=str)
    parser.add_argument('--freeze_frames_of_origin', type=str)

    # merge downsampling hyperparameters
    parser.add_argument('--prune_points', type=str)
    parser.add_argument('--downsample_reducescale', type=float)

    # isometry properties
    parser.add_argument('--isometry_knn', type=int)
    parser.add_argument('--isometry_knn_radius', type=float)
    parser.add_argument('--isometry_per_segment', type=str)

    # chamfer properties
    parser.add_argument('--chamfer_agg_group_ratio', type=float)

    # tracking loss properties
    parser.add_argument('--tracking_knn', type=int)
    parser.add_argument('--tracking_radius', type=int)
    parser.add_argument('--tracking_loss_per_segment', type=str)
    parser.add_argument('--tracking_window', type=int)

    # global rigidity loss properties
    parser.add_argument('--instance_isometry_numpairs', type=int)

    # ---------------------------- Data Parameters --------------------------------
    parser.add_argument('--depth_remove_outliers', type=str)
    # --------------------------------------------------------------------------------

    # --------------------------- Data Parser Parameters ---------------------------
    parser.add_argument('--downscale_factor', type=int)

    args, unknown_args = parser.parse_known_args()

    # overwrite boolean strings
    args.depth_remove_outliers = args.depth_remove_outliers in ["True", "true", "yes", "1"] or (args.depth_remove_outliers is None)  # defaults to True
    args.prune_points = args.prune_points in ["True", "true", "yes", "1"] or (args.prune_points is None)  # defaults to True
    args.freeze_previous_in_motion_estimation = args.freeze_previous_in_motion_estimation in ["True", "true", "yes", "1"] or (args.freeze_previous_in_motion_estimation is None)  # defaults to True
    args.static_background = args.static_background in ["True", "true", "yes", "1"]  # defaults to False
    args.no_background = (args.no_background in ["True", "true", "yes", "1"])  # defaults to False
    args.tracking_loss_per_segment = (args.tracking_loss_per_segment in ["True", "true", "yes", "1"]) or (args.tracking_loss_per_segment is None)  # defaults to True
    args.freeze_frames_of_origin = (args.freeze_frames_of_origin in ["True", "true", "yes", "1"]) or (args.freeze_frames_of_origin is None)  # defaults to True
    args.only_render = args.only_render in ["True", "true", "yes", "1"]  # defaults to False
    args.only_interactive_visualize = args.only_interactive_visualize in ["True", "true", "yes", "1"]  # defaults to False
    args.isometry_per_segment = (args.isometry_per_segment in ["True", "true", "yes", "1"]) or (args.isometry_per_segment is None)  # defaults to True

    # convert some params to float --> set them as str in argparse because wandb passes in "None" as string
    args.render_background_probability = float(args.render_background_probability) if (args.render_background_probability != "None" and args.render_background_probability is not None) else 0.5

    # format _wandb preempting arg
    if args._wandb is None:
        args._wandb = {}
    else:
        args._wandb = args._wandb.replace("'", '"')
        args._wandb = args._wandb.replace("False", 'false')
        args._wandb = args._wandb.replace("True", 'true')
        args._wandb = json.loads(args._wandb)
    if args.load_dir == "None":
        args.load_dir = ''

    return args
