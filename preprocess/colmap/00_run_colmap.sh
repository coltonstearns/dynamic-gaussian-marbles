#!/bin/bash

# prepare bash environment
directory=$1
condapath=$2
directory=$(realpath $directory)
curdir=$(pwd)
source $condapath/bin/activate
conda activate dgmarbles
echo "Creating colmap processing directory within parent" $directory

# format rgb images for colmap
mkdir "${directory}/colmap"
mkdir "${directory}/colmap/images"
python helper_convert_images.py --png_image_dir ${directory}/rgb/1x --jpg_output_dir "${directory}/colmap/images"

# format masks for colmap
#mkdir "${directory}/colmap/masks"
#python helper_convert_masks.py --segmentations_folder_path ${directory}/segmentation/1x --outdir "${directory}/colmap/masks"

## run colmap
cd "${directory}/colmap"
colmap feature_extractor --database_path ./database.db --image_path ./images  --ImageReader.mask_path ./masks --SiftExtraction.use_gpu=1
colmap exhaustive_matcher --database_path ./database.db  # or alternatively any other matcher
mkdir sparse
colmap mapper --database_path ./database.db --image_path ./images --output_path ./sparse random_seed

