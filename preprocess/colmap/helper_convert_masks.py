import numpy as np
import cv2
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        default="./data",
        help="directory to output masks",
    )
    parser.add_argument(
        "--segmentations_folder_path",
        default="",
        help="path to folder containing ordered segmentation maps",
    )

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    seg_fnames = os.listdir(args.segmentations_folder_path)
    for seg_fname in seg_fnames:
        newname = seg_fname.replace('npy', 'jpg') + '.png'
        segmentation = np.load(os.path.join(args.segmentations_folder_path, seg_fname))
        mask = (segmentation == 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(args.outdir, newname), mask)

