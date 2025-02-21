import os
import argparse
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--png_image_dir",
        default="./data",
        help="directory containing png images",
    )
    parser.add_argument(
        "--jpg_output_dir",
        default="",
        help="path to write jpg images to",
    )

    args = parser.parse_args()

    png_imnames = sorted(os.listdir(args.png_image_dir))
    png_imnames = [name for name in png_imnames if name.endswith(".png")]
    for name in png_imnames:
        im = cv2.imread(os.path.join(args.png_image_dir, name))
        cv2.imwrite(os.path.join(args.jpg_output_dir, name.replace('.png', '.jpg')), im)
