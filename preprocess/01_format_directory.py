import json
import os
import argparse
import numpy as np
import re
import cv2

SCENE_INFO = \
    {
        "center": [-0.5, 1.0, -1.75], "far": 0.6,
        "near": 0.03, "scale": 0.1
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="./data",
        help="directory to base output directory",
    )
    parser.add_argument(
        "--image_dir",
        default="",
        help="path to images to create cameras for",
    )
    parser.add_argument(
        "--fov_estimate",
        default=80,
        help="general ballpark estimate of the field of view of the camera",
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.outdir, 'camera')):
        os.makedirs(os.path.join(args.outdir, 'camera'))

    static_orientation = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    static_position = [0.0, 0.0, 0.0]
    dataset_info = {'count': 0, 'ids': [], 'num_exemplars': 0, 'train_ids': []}
    times = []
    for time, im_path in enumerate(sorted(os.listdir(args.image_dir))):
        image = cv2.imread(os.path.join(args.image_dir, im_path))
        h, w = image.shape[:2]
        focal_length = 0.5 * max(h, w) / np.tan(0.5 * args.fov_estimate * np.pi / 180)

        # create dycheck camera
        camera = {
            "focal_length": focal_length,
            "image_size": [w, h],
            "orientation": static_orientation,
            "pixel_aspect_ratio": 1.0,
            "position": static_position,
            "principal_point": [w/2, h/2],
            "radial_distortion": [0.0, 0.0, 0.0],
            "skew": 0.0,
            "tangential_distortion": [0.0, 0.0]
        }

        cam_path = im_path.replace('.png', '.json')
        with open(os.path.join(args.outdir, 'camera', cam_path), "w") as f:
            json.dump(camera, f)

        # update dataset info -- all files are used in training in this in-the-wild setting
        dataset_info['count'] += 1
        dataset_info['ids'].append(cam_path.replace('.json', ''))
        dataset_info['num_exemplars'] += 1
        dataset_info['train_ids'].append(cam_path.replace('.json', ''))
        times.append(time)

    with open(os.path.join(args.outdir, 'dataset.json'), "w") as f:
        json.dump(dataset_info, f)

    # create generic scene info
    with open(os.path.join(args.outdir, 'scene.json'), "w") as f:
        json.dump(SCENE_INFO, f)

    # create scene train split
    split_dir = os.path.join(args.outdir, 'splits')
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    train_split = {'camera_ids': [0 for i in range(dataset_info['num_exemplars'])],
                   'time_ids': times,
                   'frame_names': dataset_info['train_ids']}
    with open(os.path.join(split_dir, 'train.json'), "w") as f:
        json.dump(train_split, f)

    # create scene val split - this is simply a place holder of the first training camera
    val_split = {'camera_ids': [0], 'time_ids': [0], 'frame_names': dataset_info['train_ids'][:1]}
    with open(os.path.join(split_dir, 'val.json'), "w") as f:
        json.dump(val_split, f)




