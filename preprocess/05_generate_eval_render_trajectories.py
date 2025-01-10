import argparse
import json
import os
import copy
import numpy as np
from pathlib import Path
import math


def load_training_cameras(camera_dir, scene_scale):
    fnames = sorted(os.listdir(camera_dir))
    cameras = []
    for fname in fnames:
        # load camera json
        with open(os.path.join(camera_dir, fname)) as f:
            camera_data = json.load(f)

        # extract data
        w, h = camera_data['image_size']
        orientation = np.array(camera_data['orientation']).reshape(3,3)  # len 9 list
        position = np.array(camera_data['position'])  # len 3 list
        fov = 2 * math.atan(h / (2 * camera_data['focal_length']))
        fov = fov * 180 / np.pi


        # format position and orientation
        orientation = orientation.T
        position *= scene_scale
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = orientation
        camera_to_world[:3, 3] = position
        camera_to_world[0:3, 1:3] *= -1
        camera_to_world[2, :] *= -1  # invert world z
        camera_to_world = camera_to_world[[0, 2, 1, 3], :]  # switch y and z

        # create template camera
        camera = {'camera_to_world': camera_to_world.flatten().tolist(), 'fov': fov, 'aspect': w/h}
        cameras.append(camera)

    return cameras


def get_nvs_extrinsic_motion(camera, fps, novelview_radius):
    nvs_extrinsics = []

    # start with zoom in for 2 seconds
    zoom_nframes = fps
    orig_extrinsics = np.array(camera['camera_to_world']).reshape((4, 4)).copy()
    zoomed_in_extrinsics = orig_extrinsics.copy()
    zoomed_in_extrinsics[1, 3] += novelview_radius
    nvs_extrinsics += linear_interpolate_camera(orig_extrinsics, zoomed_in_extrinsics, zoom_nframes)
    nvs_extrinsics += linear_interpolate_camera(zoomed_in_extrinsics, orig_extrinsics, zoom_nframes)

    # then zoom out for 2 seconds
    zoomed_out_extrinsics = np.array(camera['camera_to_world']).reshape((4, 4)).copy()
    zoomed_out_extrinsics[1, 3] -= novelview_radius
    nvs_extrinsics += linear_interpolate_camera(orig_extrinsics, zoomed_out_extrinsics, zoom_nframes)
    nvs_extrinsics += linear_interpolate_camera(zoomed_out_extrinsics, orig_extrinsics, zoom_nframes)

    # then move camera right quickly
    move_right_extrinsics = np.array(camera['camera_to_world']).reshape((4, 4)).copy()
    move_right_extrinsics[0, 3] += novelview_radius
    nvs_extrinsics += linear_interpolate_camera(orig_extrinsics, move_right_extrinsics, fps // 2)

    # next, spiral camera
    spiral_nframes = fps * 2
    for j in range(spiral_nframes):
        angle = 2 * np.pi * (j / spiral_nframes)
        spiral_extrinsics = np.array(camera['camera_to_world']).reshape((4, 4)).copy()
        spiral_extrinsics[0, 3] += novelview_radius * np.cos(angle)
        spiral_extrinsics[2, 3] += novelview_radius * np.sin(angle)
        nvs_extrinsics.append(spiral_extrinsics)

    # then move camera back to center
    nvs_extrinsics += linear_interpolate_camera(move_right_extrinsics, orig_extrinsics, fps // 2)

    return nvs_extrinsics




def linear_interpolate_camera(cam1, cam2, nframes):
    out = []
    for i in range(nframes):
        interp_cam = cam1 * (nframes - i) / nframes + cam2 * (i / nframes)
        out.append(interp_cam)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outname",
        default="./data",
        help="directory to base output directory",
    )
    parser.add_argument(
        "--fps",
        default=24,
        type=int,
    )
    parser.add_argument(
        "--data_dir",
        default='./data',
    )
    parser.add_argument(
        "--novelview_frame",
        default=0,
        type=int,
        nargs='+'
    )
    parser.add_argument(
        "--novelview_radius",
        default=0.15,
        type=float,
    )
    parser.add_argument('--nvidia_dataset', action='store_true')

    args = parser.parse_args()

    # create ouput direction
    os.makedirs(os.path.join(args.data_dir, 'camera_paths'), exist_ok=True)

    # get camera info
    camera_dir = os.path.join(args.data_dir, 'camera')
    camera_fname = os.listdir(camera_dir)[0]
    with open(os.path.join(camera_dir, camera_fname)) as f:
        camera_data = json.load(f)
    w, h = camera_data['image_size']
    orientation = np.array(camera_data['orientation']).reshape(3,3)  # len 9 list
    position = np.array(camera_data['position'])  # len 3 list
    fov = 2 * math.atan(h / (2 * camera_data['focal_length']))
    fov = fov * 180 / np.pi

    # get scene info and cameras
    with open(os.path.join(args.data_dir, "scene.json"), 'r') as f:
        scene_dict = json.load(f)
    scale = scene_dict["scale"]
    cameras = load_training_cameras(camera_dir, scale)

    # load list of training images
    image_dir = os.path.join(args.data_dir, "rgb", "1x")
    rgbs = sorted(os.listdir(image_dir))
    rgbs = [rgb for rgb in rgbs if rgb.startswith('0')]

    # ==============================================================
    # create static novel view synthesis cameras
    all_cameras = {'camera_type': 'perspective', 'render_height': h, 'render_width': w,
                   'fps': args.fps, 'crop': None}
    for keyframe in args.novelview_frame:
        camera_trajectory = []
        assert keyframe < len(rgbs), f"Novel view frame {keyframe} is greater than number of frames {len(rgbs)}"
        for i in range(len(rgbs)):
            # append training view camera
            time = i / len(rgbs)
            camera = copy.deepcopy(cameras[i])
            camera['render_time'] = time
            camera_trajectory.append(camera)

            # if we're at our novel view timeframe, create zoom in/out as well as spiral effect
            if i == keyframe:
                nvs_extrinsics = get_nvs_extrinsic_motion(camera, args.fps, args.novelview_radius)

                # append novelview cameras to camera path
                for extrinsics in nvs_extrinsics:
                    nvs_camera = copy.deepcopy(camera)
                    nvs_camera['camera_to_world'] = extrinsics.flatten().tolist()
                    nvs_camera['render_time'] = time
                    camera_trajectory.append(nvs_camera)

        all_cameras['camera_path'] = camera_trajectory
        seconds = len(camera_trajectory) / args.fps
        all_cameras['seconds'] = seconds

        # write to file
        outname = args.outname.replace('.json', '') + '_%sKF' % keyframe
        outname = os.path.join(args.data_dir, 'camera_paths', outname)
        with open(outname + '.json', 'w') as f:
            json.dump(all_cameras, f, indent=4)


    # ==============================================================
    # create spiral camera trajectory
    all_cameras = {'camera_type': 'perspective', 'render_height': h, 'render_width': w,
                   'fps': args.fps, 'crop': None}
    camera_trajectory = []
    assert keyframe < len(rgbs), f"Novel view frame {keyframe} is greater than number of frames {len(rgbs)}"
    spiral_nframes = len(rgbs) // 3
    for i in range(len(rgbs)):
        # load training view camera
        time = i / len(rgbs)
        camera = copy.deepcopy(cameras[i])
        camera['render_time'] = time

        # augment by spiral offset
        angle = 2 * np.pi * (i / spiral_nframes)
        spiral_extrinsics = np.array(camera['camera_to_world']).reshape((4, 4)).copy()
        spiral_extrinsics[0, 3] += args.novelview_radius * np.cos(angle)
        spiral_extrinsics[2, 3] += args.novelview_radius * np.sin(angle)
        camera['camera_to_world'] = spiral_extrinsics.flatten().tolist()
        camera_trajectory.append(camera)

    all_cameras['camera_path'] = camera_trajectory
    seconds = len(camera_trajectory) / args.fps
    all_cameras['seconds'] = seconds

    # write to file
    outname = args.outname.replace('.json', '') + '_spiral'
    outname = os.path.join(args.data_dir, 'camera_paths', outname)
    with open(outname + '.json', 'w') as f:
        json.dump(all_cameras, f, indent=4)


