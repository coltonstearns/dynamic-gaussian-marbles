import argparse
import cv2
import numpy as np
import os
import torch
import json

from PIL import Image
import depth_pro
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./')
    parser.add_argument('--data_dir', type=str, default='./')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists(os.path.join("checkpoints", "depth_pro.pt")):
        print("Downloading Depth Pro Weights!")
        subprocess.run("wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints", shell=True)

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.cuda().eval()

    # get rgb filenames
    filenames = os.listdir(args.img_path)
    filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
    filenames.sort()
    os.makedirs(args.outdir, exist_ok=True)

    depth_medians, depth_stds = [], []
    for k, filename in enumerate(filenames):
        print(f'Progress {k + 1}/{len(filenames)}: {filename}')
        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(filename)
        image = transform(image)

        # Run inference.
        prediction = model.infer(image.cuda(), f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        depth = depth.cpu().numpy().astype(np.float32)
        depth_medians.append(np.median(depth))
        depth_stds.append(np.std(depth))
        filename = os.path.basename(filename)
        outpath = os.path.join(args.outdir, filename.replace('png', 'npy'))
        np.save(outpath, depth)

    # overwrite scene info to account for appropriate depth scaling
    # we want our scene to have median roughly at 5 units, and most of the scene within [0,10]
    avg_median = np.mean(depth_medians)
    scaling = 4.0 / avg_median  # put median at 4 units away
    far_plane = 20 / scaling
    near_plane = 0.05 / scaling
    with open(os.path.join(args.data_dir, "scene.json"), "w") as f:
        scene = { "far": far_plane, "near": near_plane, "scale": scaling}
        json.dump(scene, f)
