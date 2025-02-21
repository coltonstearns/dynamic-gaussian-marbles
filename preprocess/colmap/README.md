> The below instructions assume a Linux-based operating system with a bash shell and ffmpeg installed.  
<img src="../../media/linux-icon.png" alt="drawing" width="80"/>
<img src="../../media/ffmpeg-icon.png" alt="drawing" width="120"/>


## Step 0: Preprocess Directory without Colmap
**Please follow README in the parent directory to first preprocess the data directory WITHOUT colmap!**

This results in a data folder `./data/real-world/<video-name>` with the following structure:
```aiignore
scene_datadir
├── camera
│   └── C_XXXXX.json
│   └── ...
├── camera_paths
│   ├── novelview_20KF.json
│   └── ...
├── rgb
│   └── 1x
│       ├── C_XXXXX.png
│       └── ...
├── depth
│   └── 1x
│       ├── C_XXXXX.npy
│       └── ...
├── segmentation
│   └── 1x
│       ├── C_XXXXX.npy
│       └── ...
├── tracks
│   └── 1x
│       ├── track_XXXXX.npy
│       └── ...
└── splits
│   ├── train.json
│   └── val.json
│── scene.json
└── dataset.json 
```

## Step 1: Estimating Camera Poses
We provide a script that automatically runs colmap on the video non-foreground regions and formats the resulting camera poses. 
Before running this, please make sure you have [colmap](https://colmap.github.io/) successfully installed on your ubuntu machine, verified
by running `colmap -h`.

Pass the data directory and base conda environment to `create_posed_images.sh`:
```aiignore
bash create_posed_images.sh ../data/real-world/<video-name> ~/anaconda3
```

> Note: COLMAP is very slow, and often fails. This script prints COLMAP's status to stoud, but it can
> sometimes be unclear if COLMAP fails to find the correct solution. One sanity check is to 
> look in the `./data/real-world/<video-name>/colmap/sparse` directory. If there are more subdirectories than
> `'0'`, it is likely COLMAP failed to solve all camera poses.

> Note: We update the depth maps to be in the same scale as the COLMAP camera poses. If COLMAP succeeds but only estimates
> a highly sparse point cloud, our depthmap-alignment may fail.

## Step 2: Using Camera Poses
If Step 1 finishes successfully, the data directory will have two **additional** folders:
```aiignore
scene_datadir
├── ...
├── ...
├── camera-colmap
└── depth-colmap
```
To use the camera poses, simply rename these directories
to `camera` and `depth` respectively (and place the original camera and depth directories under a new name). And 
that's it! The code natively handles camera pose!