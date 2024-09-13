#!/bin/bash

video_path=$1
directory=$2
directory=$(realpath $directory)
video_path=$(realpath $video_path)
curdir=$(pwd)
echo "Creating data directory" $directory "for video:" $video_path

# create rgb and sam2 directories
mkdir $directory
mkdir "${directory}/rgb"
mkdir "${directory}/rgb/1x"
mkdir "${directory}/sam2"

# get images using ffmpeg
ffmpeg -i $video_path -r 10 "${directory}/rgb/1x/%04d.png"
ffmpeg -framerate 24 -pattern_type glob -i "${directory}/rgb/1x/*.png" -c:v libx264 -pix_fmt yuv420p "${directory}/video.mp4"
# make framerate 24 because SAM2 takes this be default!
