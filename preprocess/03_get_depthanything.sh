#!/bin/bash

# store directory info
directory=$1
condapath=$2
curdir=$(pwd)
directory=$(realpath $directory)
source $condapath/bin/activate
conda activate dynamic-gaussian-marbles

# run depth estimation
cd external/Depth-Anything-V2/metric_depth
if [ ! -f checkpoints/depth_anything_v2_metric_hypersim_vitl.pth ]; then
    echo "Need to download HyperSim DepthAnythingV2 checkpoint."
    mkdir checkpoints
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true -O checkpoints/depth_anything_v2_metric_hypersim_vitl.pth
else
    echo "HyperSim DepthAnythingV2 checkpoint already downloaded!"
fi
python preprocess_depth_anything.py --img-path $directory/rgb/1x --outdir $directory/depth/1x --encoder vitl --dataset hypersim
cd $curdir