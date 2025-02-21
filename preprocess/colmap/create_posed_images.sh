#!/bin/bash

# prepare bash environment
directory=$1
condapath=$2
directory=$(realpath $directory)
curdir=$(pwd)
source $condapath/bin/activate
conda activate dgmarbles
echo "Creating colmap processing directory within parent" $directory

# first, run COLMAP on images
bash 00_run_colmap.sh $directory $condapath

# extract camera poses and updated instrinsics from COLMAP
python 01_format_colmap_cameras.py --outdir "${directory}/camera-colmap" --colmap_database_path "${directory}/colmap/database.db" --colmap_sparse_dir "${directory}/colmap/sparse"

# get COLMAP sparse point cloud
# now part of colmap process!
# python 02_get_colmap_pointcloud.py --input_path "${directory}/colmap/sparse/0/points3D.bin" --output_dir "${directory}"

# update our depth maps to align with COLMAP intrinsics and extrinsics
PYTHONPATH=../.. python 03_align_depthmaps_with_colmap.py --input_colmap_pc "${directory}/points3D.npy" --input_depth_dir "${directory}/depth/1x" --input_camera_dir "${directory}/camera-colmap" \
   --input_segmentation_dir "${directory}/segmentation/1x" --output_depth_dir "${directory}/depth-colmap/1x" --path_fnmatch "*.npy"

# finally, recreate eval renders
cd ..
python 05_generate_eval_render_trajectories.py --outname novelview.json --data_dir $directory --camera_dir "${directory}/camera-colmap" --fps 24 --novelview_frame 20 40 60 --novelview_radius 0.5
