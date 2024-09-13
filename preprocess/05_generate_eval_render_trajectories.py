import argparse
import json
import os
import copy
import numpy as np
from pathlib import Path
import math

CAMERA_LOC = np.array([[1, 0, 0, 1.25],
                       [0, 0, -1, -4.375],
                       [0, 1, 0, -2.5],
                       [0, 0, 0, 1]])

CAMERA_LOC_NVIDIA = np.array([[1, 0, 0, -0.51],
                              [0, 0, -1, -2.14],
                              [0, 1, 0, -0.02],
                              [0, 0, 0, 1]])



# y is in/out, x is left/right, z is up/down

CAMERA_TEMPLATE = {'camera_to_world': [1, 0, 0, 1.25,
                                       0, 0, -1, -4.375,
                                       0, 1, 0, -2.5,
                                       0, 0, 0, 1],
                    'fov': 50,
                   'aspect': 1.6461187214611872,
                   'render_time': 0}

CAMERA_TEMPLATE_NVIDIA = {'camera_to_world': [1, 0, 0, -0.51,
                                       0, 0, -1, -2.14,
                                       0, 1, 0, -0.02,
                                       0, 0, 0, 1],
                    'fov': 50,
                   'aspect': 1.6461187214611872,
                   'render_time': 0}


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

    # get scene info
    with open(os.path.join(args.data_dir, "scene.json"), 'r') as f:
        scene_dict = json.load(f)
    center = np.array(scene_dict["center"], dtype=np.float32)
    scale = scene_dict["scale"]
    near = scene_dict["near"]
    far = scene_dict["far"]

    # format position and orientation
    orientation = orientation.T
    position -= center
    bbox_size, dataparse_scalefactor = 1.5, 4.0
    position *= bbox_size / 4 / far * dataparse_scalefactor
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = orientation
    camera_to_world[:3, 3] = position
    camera_to_world[0:3, 1:3] *= -1
    camera_to_world[2, :] *= -1  # invert world z
    camera_to_world = camera_to_world[[0, 2, 1, 3], :]  # switch y and z

    # create template camera
    camera_template = {'camera_to_world': camera_to_world.flatten().tolist(), 'fov': fov, 'aspect': w/h}
    camera_loc_template = camera_to_world

    # create nerfstudio camera dict
    all_cameras = {'camera_type': 'perspective', 'render_height': h, 'render_width': w,
                   'fps': args.fps, 'crop': None}
    # camera_template = copy.deepcopy(CAMERA_TEMPLATE_NVIDIA) if args.nvidia_dataset else copy.deepcopy(CAMERA_TEMPLATE)
    # camera_template['aspect'] = args.width / args.height
    # camera_template['fov'] = args.fov

    image_dir = os.path.join(args.data_dir, "rgb", "1x")
    rgbs = sorted(os.listdir(image_dir))
    rgbs = [rgb for rgb in rgbs if rgb.startswith('0')]
    for novel_frame in args.novelview_frame:
        camera_path = []
        assert novel_frame < len(rgbs), f"novelview_frame {novel_frame} is greater than number of frames {len(rgbs)}"

        for i in range(len(rgbs)):
            # append training view camera
            time = i / len(rgbs)
            camera = copy.deepcopy(camera_template)
            camera['render_time'] = time
            camera_path.append(camera)

            # if we're at our novel view timeframe, create zoom in/out as well as spiral effect
            if i == novel_frame:
                novelview_cameras = []

                # start with zoom in for 2 seconds
                zoom_nframes = args.fps
                camera_loc = camera_loc_template.copy()
                zoomed_in_camera = camera_loc_template.copy()
                zoomed_in_camera[1, 3] += args.novelview_radius
                novelview_cameras += linear_interpolate_camera(camera_loc, zoomed_in_camera, zoom_nframes)
                novelview_cameras += linear_interpolate_camera(zoomed_in_camera, camera_loc, zoom_nframes)

                # then zoom out for 2 seconds
                zoomed_out_camera = camera_loc_template.copy()
                zoomed_out_camera[1, 3] -= args.novelview_radius
                novelview_cameras += linear_interpolate_camera(camera_loc, zoomed_out_camera, zoom_nframes)
                novelview_cameras += linear_interpolate_camera(zoomed_out_camera, camera_loc, zoom_nframes)

                # then move camera right quickly
                move_right_camera = camera_loc_template.copy()
                move_right_camera[0, 3] += args.novelview_radius
                novelview_cameras += linear_interpolate_camera(camera_loc, move_right_camera, args.fps//2)

                # next, spiral camera
                spiral_nframes = args.fps * 2
                for j in range(spiral_nframes):
                    angle = 2 * np.pi * (j / spiral_nframes)
                    spiral_camera = camera_loc_template.copy()
                    spiral_camera[0, 3] += args.novelview_radius * np.cos(angle)
                    spiral_camera[2, 3] += args.novelview_radius * np.sin(angle)
                    novelview_cameras.append(spiral_camera)

                # then move camera back to center
                novelview_cameras += linear_interpolate_camera(move_right_camera, camera_loc, args.fps//2)

                # append novelview cameras to camera path
                for novelview_camera in novelview_cameras:
                    camera = copy.deepcopy(camera_template)
                    camera['camera_to_world'] = novelview_camera.flatten().tolist()
                    camera['render_time'] = time
                    camera_path.append(camera)

        all_cameras['camera_path'] = camera_path
        seconds = len(camera_path) / args.fps
        all_cameras['seconds'] = seconds

        # write to file
        outname = args.outname.replace('.json', '') + '_%sKF' % novel_frame
        outname = os.path.join(args.data_dir, 'camera_paths', outname)
        with open(outname + '.json', 'w') as f:
            json.dump(all_cameras, f, indent=4)


