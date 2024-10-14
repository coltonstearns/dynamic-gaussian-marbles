
import os
import shutil
from datetime import datetime
import torch
import os.path as osp
import wandb
import time

from src.utils.load_args import get_argparse_input
from src.visualization.render import render_trainval, render_video_trajectory
from src.utils.load_model import load_model_config, load_data_args

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def train(args):
    wandb.init(project="DynamicGaussianMarbles", config=args)

    # Set up output directory
    if "sweep" in args.output.lower():
        args.output = os.path.join(args.output, wandb.run.id)

    # check if this unique run ID's directory exists. If it does, this means we're respawning a previous run!
    # if osp.exists(args.output) and "sweep" in args.output.lower() and not args.load_dir:
    #     setup_wandb_preemptive(args)
    #     print("Preempting Sweep Run! Run ID %s" % wandb.run.id)

    # set up training config
    os.makedirs(args.output, exist_ok=True)
    train_config = load_model_config(args)
    train_config = load_data_args(train_config, args)
    data_dir = train_config.pipeline.datamanager.data

    # initialize nerfstudio config
    train_config.set_timestamp()
    args.output = train_config.get_base_dir()
    train_config.print_to_terminal()
    train_config.save_config()

    # build nerfstudio training engine
    trainer = train_config.setup()
    nerfstudio_log_dir = (trainer.base_dir / trainer.config.viewer.relative_log_filename).parent
    if not os.path.exists(nerfstudio_log_dir):
        os.makedirs(nerfstudio_log_dir)
    trainer.setup()

    # if we loaded a model, update the number of optimization iterations
    trainer.config.max_num_iterations = trainer.config.max_num_iterations - trainer._start_step

    # Run trainer
    if not args.only_render and not args.only_interactive_visualize:
        trainer.train()
    print('Finished Training!')

    # render training images
    timestamp = "_".join(str(datetime.now()).split(" "))[:-7]
    trainer.pipeline.eval()

    # check if in interactive mode
    if args.only_interactive_visualize:
        print("Entering Interactive Visualization.")
        while True:
            time.sleep(1)

    # compute correspondences in internal representation
    trainer.pipeline.model.compute_correspondences()
    render_trainval(trainer, args.output, timestamp)

    # calculate final metrics
    final_val_losses = trainer.pipeline.get_average_eval_image_metrics()
    final_val_losses = {"FINAL_VAL_" + k: final_val_losses[k] for k in final_val_losses}
    wandb.log(final_val_losses)

    # run dycheck tracking eval
    dycheck_tracking_score = trainer.pipeline.run_dycheck_tracking_eval(outdir=osp.join(args.output, "render"))
    wandb.log(dycheck_tracking_score)

    # Render novel view synthesis trajectory
    trainer.pipeline.eval()
    if os.path.exists(os.path.join(str(data_dir), 'camera_paths')):
        render_config_dir = os.path.join(str(data_dir), 'camera_paths')
    else:
        render_config_dir = None

    # render all camera trajectories
    outnames, trajectory_files = [], []
    list_render_config_dir = [] if render_config_dir is None else os.listdir(render_config_dir)
    for trajectory_file in list_render_config_dir:
        outname = trajectory_file.replace('.json', '.mp4')
        trajectory_files.append(osp.join(render_config_dir, trajectory_file))
        outnames.append(outname)
    render_video_trajectory(trainer.pipeline, osp.join(args.output, "render"), outnames, trajectory_files)

    # empty cuda cache
    torch.cuda.empty_cache()


def setup_wandb_preemptive(args):
    print("Preempting Sweep Run! Run ID %s" % wandb.run.id)
    load_dir = args.output
    child_directories = os.listdir(load_dir)
    while 'nerfstudio_models' not in child_directories:
        if len(child_directories) == 0:
            raise RuntimeError("Trying to preempt from directory %s, but couldn't find models directory" % args.output)
        if len(child_directories) > 1:  # indicates we're at nerfstudio date/time folder
            checkpoints = []
            for child in child_directories:
                checkpoint_dir = os.path.join(load_dir, child, 'nerfstudio_models')
                if os.path.exists(checkpoint_dir):
                    checkpoints.append(sorted(os.listdir(checkpoint_dir))[-1])
                else:
                    checkpoints.append('')
            latest_idx = sorted(range(len(checkpoints)), key=checkpoints.__getitem__)[-1]
            child_directories = [child_directories[latest_idx]]
        load_dir = os.path.join(load_dir, child_directories[0])
        child_directories = os.listdir(load_dir)
    load_dir = os.path.join(load_dir, 'nerfstudio_models')
    print("Loading preemptive run from %s" % load_dir)
    args.load_dir = load_dir


if __name__ == '__main__':
    args = get_argparse_input()
    train(args)
