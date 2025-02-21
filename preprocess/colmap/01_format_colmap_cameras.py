import sys
import numpy as np
import argparse
import os
import json
import sqlite3
from helper_colmap_utils import read_cameras_binary, read_images_binary
import struct
import plotly.graph_objects as go
import open3d as o3d


def generate_plotly_pc_fig(pc, clrs, title, size=[-3, 3], fig=None, row=None, col=None, markersize=None):
    """
    Visualize a point cloud in plotly
    pc: numpy array of size Nx3
    clrs: numpy array or list of size N x 3
    """
    clrs = [f'rgb({int(clrs[i][0])}, {int(clrs[i][1])}, {int(clrs[i][2])})' for i in range(clrs.shape[0])]

    fig = go.Figure(data =[go.Scatter3d(x = pc[:, 0],
                                        y = pc[:, 1],
                                        z = pc[:, 2],
                                        mode ='markers',
                                        marker=dict(color=clrs, size=2 if markersize is None else markersize, opacity=1, line=dict(width=0)))])
    fig.update_layout(title_text=title,
                      scene=dict(
                          xaxis=dict(range=size,
                                 # backgroundcolor="rgba(0, 0, 0,0)",
                                 # gridcolor="white",
                                 # showbackground=True,
                                 # zerolinecolor="white",
                                 # visible=False
                                     ),
                          yaxis=dict(range=size,
                                     # backgroundcolor="rgba(0, 0, 0,0)",
                                     # gridcolor="white",
                                     # showbackground=True,
                                     # zerolinecolor="white",
                                     # visible=False
                                     ),
                          zaxis=dict(range=size,
                                     # backgroundcolor="rgba(0, 0, 0,0)",
                                     # gridcolor="white",
                                     # showbackground=True,
                                     # zerolinecolor="white",
                                     # visible=False
                                     )
                      ),
                      scene_aspectmode='cube',
                      )

    return fig


def read_points_3d_binary(path, outdir):
    points, all_corrrespondences = [], []
    with open(path, 'rb') as file:
        num_points_3d = struct.unpack('<Q', file.read(8))[0]
        for _ in range(num_points_3d):
            point_3d_id = struct.unpack('<Q', file.read(8))[0]
            x = struct.unpack('<d', file.read(8))[0]
            y = struct.unpack('<d', file.read(8))[0]
            z = struct.unpack('<d', file.read(8))[0]
            r = struct.unpack('<B', file.read(1))[0]
            g = struct.unpack('<B', file.read(1))[0]
            b = struct.unpack('<B', file.read(1))[0]
            error = struct.unpack('<d', file.read(8))[0]
            point = [x, y, z, r, g, b, error]
            points.append(point)

            correspondences = []
            track_length = struct.unpack('<Q', file.read(8))[0]
            for _ in range(track_length):
                image_id = struct.unpack('<I', file.read(4))[0]
                point_2d_idx = struct.unpack('<I', file.read(4))[0]
                correspondences.append((image_id, point_2d_idx))
            all_corrrespondences.append(correspondences)

    # move from OpenCV to OpenGL
    points = np.array(points)
    # todo: is this the correct coordinate system!?
    points[:, 1:3] *= -1  # invert y and z for y-down, z-forward to y-up, z-backward

    np.save(os.path.join(outdir, 'points3D.npy'), points)
    # np.save(os.path.join(outdir, 'correspondences.npy'), all_corrrespondences)

    # visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points)[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.array(points)[:, 3:6])
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=32, std_ratio=0.5)
    pc_viz = np.array(pcd.points)
    clrs_viz = np.array(pcd.colors)
    size = np.max(np.std(pc_viz, axis=0) * 3)
    pc_viz -= np.mean(pc_viz, axis=0)
    fig = generate_plotly_pc_fig(pc_viz, clrs_viz, "3D Points", size=[-size, size])
    # fig.show()
    fig.write_html(os.path.join(outdir, 'points3D.html'))


def get_colmap_db_mappings(db_path):
    # Connect to the database
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Execute the query to retrieve data from the 'images' table
    query = "SELECT * FROM images ORDER BY image_id;"
    cur.execute(query)

    # Fetch all rows (data) from the result
    rows = cur.fetchall()

    # Print the data (you can customize this part as needed)
    mappings = {}
    for row in rows:
        image_name, image_id = row[1], row[0]
        mappings[image_name] = image_id

    # Close the database connection
    con.close()

    return mappings


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        default="./cameras",
        help="directory to output colmap images.txt file",
    )
    parser.add_argument(
        "--colmap_database_path",
        default="",
        help="path to colmap db file",
    )
    parser.add_argument(
        "--colmap_sparse_dir",
        default="",
        help="path to colmap folder with binary output files",
    )
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # colmap goes through many sparse
    subdirs = os.listdir(args.colmap_sparse_dir)  # often '0', '1', '2', etc. When colmap doesn't perfectly solve it
    num_images = 0
    colmap_cameras, colmap_images, colmap_subdir = None, None, None
    for subdir in subdirs:
        cdir = os.path.join(args.colmap_sparse_dir, subdir)
        cams = read_cameras_binary(os.path.join(cdir, 'cameras.bin'))
        imgs = read_images_binary(os.path.join(cdir, 'images.bin'))
        if len(imgs) > num_images:
            colmap_cameras, colmap_images, colmap_subdir = cams, imgs, subdir
            num_images = len(imgs)

    # get colmap db mappings
    db_mappings = get_colmap_db_mappings(args.colmap_database_path)
    db_mappings = {v: k.replace('.jpg', '.png') for k, v in db_mappings.items()}  # now maps from colmap ID to image name

    # iterate through cameras and read info
    print(db_mappings)
    print(colmap_cameras)
    first_key = sorted(list(colmap_cameras.keys()))[0]
    colmap_camera = colmap_cameras[first_key]  # we only use one camera intrinsics!
    for colmap_id in colmap_images.keys():
        # sanity check on intrinsics
        print("Difference in intrinsics between this camera and first camera:")
        print(colmap_cameras[colmap_id].params - colmap_camera.params)

        # get colmap rotation and translation
        colmap_image = colmap_images[colmap_id]
        qvec, tvec = colmap_image.qvec, colmap_image.tvec
        imname = colmap_image.name.replace('.jpg', '')

        # convert into colmap extrinsics matrix
        colmap_rotation = qvec2rotmat(qvec)
        colmap_w2c = np.eye(4)
        colmap_w2c[:3, :3] = colmap_rotation
        colmap_w2c[:3, 3] = tvec

        # convert from w2c to c2w
        colmap_c2w = np.linalg.inv(colmap_w2c)

        # convert from colmap coords to nerfstudio coords
        nstudio_c2w = colmap_c2w.copy()
        nstudio_c2w[0:3, 1:3] *= -1
        nstudio_c2w[[2, 1], :] = nstudio_c2w[[1, 2], :]
        nstudio_c2w[2, :] *= -1

        # convert from nerfstudio to our weird coordinate system
        our_c2w = nstudio_c2w.copy()
        our_c2w = our_c2w[[2, 0, 1], :]
        our_c2w[2, :] *= -1
        our_c2w = our_c2w[[1, 0, 2], :]
        our_c2w[0:3, 1:3] *= -1
        our_rotation = our_c2w[:3, :3].copy()
        our_translation = our_c2w[:3, 3].copy()
        our_rotation = our_rotation.T


        # finally, log this camera pose
        our_camera = {
            "focal_length": colmap_cameras[colmap_id].params[0],
            "image_size": [colmap_camera.width, colmap_camera.height],
            "orientation": our_rotation.tolist(),
            "pixel_aspect_ratio": 1.0,
            "position": our_translation.tolist(),
            "principal_point": [colmap_camera.params[1], colmap_camera.params[2]],
            "radial_distortion": [0.0, 0.0, 0.0],
            "skew": 0.0,
            "tangential_distortion": [0.0, 0.0]
        }
        with open(os.path.join(args.outdir, imname + '.json'), 'w') as f:
            json.dump(our_camera, f)

    # format COLMAP sparse point cloud
    pc_path = os.path.join(args.colmap_sparse_dir, colmap_subdir, 'points3D.bin')
    read_points_3d_binary(pc_path, os.path.join(args.outdir, '..'))
