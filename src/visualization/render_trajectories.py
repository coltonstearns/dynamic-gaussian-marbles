import numpy as np
import open3d as o3d
import torch
import time
# from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import open3d.visualization.rendering as rendering

# from helpers import setup_camera, quat_mult
# from external import build_rotation
# from colormap import colormap
from copy import deepcopy
from matplotlib import pyplot as plt

colormap = np.array([
    # 0     ,         0,         0,
    0.5020,         0,         0,
         0,    0.5020,         0,
    0.5020,    0.5020,         0,
         0,         0,    0.5020,
    0.5020,         0,    0.5020,
         0,    0.5020,    0.5020,
    # 0.5020,    0.5020,    0.5020,
    0.2510,         0,         0,
    0.7529,         0,         0,
    0.2510,    0.5020,         0,
    0.7529,    0.5020,         0,
    0.2510,         0,    0.5020,
    0.7529,         0,    0.5020,
    0.2510,    0.5020,    0.5020,
    0.7529,    0.5020,    0.5020,
         0,    0.2510,         0,
    0.5020,    0.2510,         0,
         0,    0.7529,         0,
    0.5020,    0.7529,         0,
         0,    0.2510,    0.5020,
    0.5020,    0.2510,    0.5020,
         0,    0.7529,    0.5020,
    0.5020,    0.7529,    0.5020,
    0.2510,    0.2510,         0,
    0.7529,    0.2510,         0,
    0.2510,    0.7529,         0,
    0.7529,    0.7529,         0,
    0.2510,    0.2510,    0.5020,
    0.7529,    0.2510,    0.5020,
    0.2510,    0.7529,    0.5020,
    0.7529,    0.7529,    0.5020,
         0,         0,    0.2510,
    0.5020,         0,    0.2510,
         0,    0.5020,    0.2510,
    0.5020,    0.5020,    0.2510,
         0,         0,    0.7529,
    0.5020,         0,    0.7529,
         0,    0.5020,    0.7529,
    0.5020,    0.5020,    0.7529,
    0.2510,         0,    0.2510,
    0.7529,         0,    0.2510,
    0.2510,    0.5020,    0.2510,
    0.7529,    0.5020,    0.2510,
    0.2510,         0,    0.7529,
    0.7529,         0,    0.7529,
    0.2510,    0.5020,    0.7529,
    0.7529,    0.5020,    0.7529,
         0,    0.2510,    0.2510,
    0.5020,    0.2510,    0.2510,
         0,    0.7529,    0.2510,
    0.5020,    0.7529,    0.2510,
         0,    0.2510,    0.7529,
    0.5020,    0.2510,    0.7529,
         0,    0.7529,    0.7529,
    0.5020,    0.7529,    0.7529,
    # 0.2510,    0.2510,    0.2510,
    0.7529,    0.2510,    0.2510,
    0.2510,    0.7529,    0.2510,
    0.7529,    0.7529,    0.2510,
    0.2510,    0.2510,    0.7529,
    0.7529,    0.2510,    0.7529,
    0.2510,    0.7529,    0.7529,
    # 0.7529,    0.7529,    0.7529,
    0.1255,         0,         0,
    0.6275,         0,         0,
    0.1255,    0.5020,         0,
    0.6275,    0.5020,         0,
    0.1255,         0,    0.5020,
    0.6275,         0,    0.5020,
    0.1255,    0.5020,    0.5020,
    0.6275,    0.5020,    0.5020,
    0.3765,         0,         0,
    0.8784,         0,         0,
    0.3765,    0.5020,         0,
    0.8784,    0.5020,         0,
    0.3765,         0,    0.5020,
    0.8784,         0,    0.5020,
    0.3765,    0.5020,    0.5020,
    0.8784,    0.5020,    0.5020,
    0.1255,    0.2510,         0,
    0.6275,    0.2510,         0,
    0.1255,    0.7529,         0,
    0.6275,    0.7529,         0,
    0.1255,    0.2510,    0.5020,
    0.6275,    0.2510,    0.5020,
    0.1255,    0.7529,    0.5020,
    0.6275,    0.7529,    0.5020,
    0.3765,    0.2510,         0,
    0.8784,    0.2510,         0,
    0.3765,    0.7529,         0,
    0.8784,    0.7529,         0,
    0.3765,    0.2510,    0.5020,
    0.8784,    0.2510,    0.5020,
    0.3765,    0.7529,    0.5020,
    0.8784,    0.7529,    0.5020,
    0.1255,         0,    0.2510,
    0.6275,         0,    0.2510,
    0.1255,    0.5020,    0.2510,
    0.6275,    0.5020,    0.2510,
    0.1255,         0,    0.7529,
    0.6275,         0,    0.7529,
    0.1255,    0.5020,    0.7529,
    0.6275,    0.5020,    0.7529,
    0.3765,         0,    0.2510,
    0.8784,         0,    0.2510,
    0.3765,    0.5020,    0.2510,
    0.8784,    0.5020,    0.2510,
    0.3765,         0,    0.7529,
    0.8784,         0,    0.7529,
    0.3765,    0.5020,    0.7529,
    0.8784,    0.5020,    0.7529,
    0.1255,    0.2510,    0.2510,
    0.6275,    0.2510,    0.2510,
    0.1255,    0.7529,    0.2510,
    0.6275,    0.7529,    0.2510,
    0.1255,    0.2510,    0.7529,
    0.6275,    0.2510,    0.7529,
    0.1255,    0.7529,    0.7529,
    0.6275,    0.7529,    0.7529,
    0.3765,    0.2510,    0.2510,
    0.8784,    0.2510,    0.2510,
    0.3765,    0.7529,    0.2510,
    0.8784,    0.7529,    0.2510,
    0.3765,    0.2510,    0.7529,
    0.8784,    0.2510,    0.7529,
    0.3765,    0.7529,    0.7529,
    0.8784,    0.7529,    0.7529,
         0,    0.1255,         0,
    0.5020,    0.1255,         0,
         0,    0.6275,         0,
    0.5020,    0.6275,         0,
         0,    0.1255,    0.5020,
    0.5020,    0.1255,    0.5020,
         0,    0.6275,    0.5020,
    0.5020,    0.6275,    0.5020,
    0.2510,    0.1255,         0,
    0.7529,    0.1255,         0,
    0.2510,    0.6275,         0,
    0.7529,    0.6275,         0,
    0.2510,    0.1255,    0.5020,
    0.7529,    0.1255,    0.5020,
    0.2510,    0.6275,    0.5020,
    0.7529,    0.6275,    0.5020,
         0,    0.3765,         0,
    0.5020,    0.3765,         0,
         0,    0.8784,         0,
    0.5020,    0.8784,         0,
         0,    0.3765,    0.5020,
    0.5020,    0.3765,    0.5020,
         0,    0.8784,    0.5020,
    0.5020,    0.8784,    0.5020,
    0.2510,    0.3765,         0,
    0.7529,    0.3765,         0,
    0.2510,    0.8784,         0,
    0.7529,    0.8784,         0,
    0.2510,    0.3765,    0.5020,
    0.7529,    0.3765,    0.5020,
    0.2510,    0.8784,    0.5020,
    0.7529,    0.8784,    0.5020,
         0,    0.1255,    0.2510,
    0.5020,    0.1255,    0.2510,
         0,    0.6275,    0.2510,
    0.5020,    0.6275,    0.2510,
         0,    0.1255,    0.7529,
    0.5020,    0.1255,    0.7529,
         0,    0.6275,    0.7529,
    0.5020,    0.6275,    0.7529,
    0.2510,    0.1255,    0.2510,
    0.7529,    0.1255,    0.2510,
    0.2510,    0.6275,    0.2510,
    0.7529,    0.6275,    0.2510,
    0.2510,    0.1255,    0.7529,
    0.7529,    0.1255,    0.7529,
    0.2510,    0.6275,    0.7529,
    0.7529,    0.6275,    0.7529,
         0,    0.3765,    0.2510,
    0.5020,    0.3765,    0.2510,
         0,    0.8784,    0.2510,
    0.5020,    0.8784,    0.2510,
         0,    0.3765,    0.7529,
    0.5020,    0.3765,    0.7529,
         0,    0.8784,    0.7529,
    0.5020,    0.8784,    0.7529,
    0.2510,    0.3765,    0.2510,
    0.7529,    0.3765,    0.2510,
    0.2510,    0.8784,    0.2510,
    0.7529,    0.8784,    0.2510,
    0.2510,    0.3765,    0.7529,
    0.7529,    0.3765,    0.7529,
    0.2510,    0.8784,    0.7529,
    0.7529,    0.8784,    0.7529,
    0.1255,    0.1255,         0,
    0.6275,    0.1255,         0,
    0.1255,    0.6275,         0,
    0.6275,    0.6275,         0,
    0.1255,    0.1255,    0.5020,
    0.6275,    0.1255,    0.5020,
    0.1255,    0.6275,    0.5020,
    0.6275,    0.6275,    0.5020,
    0.3765,    0.1255,         0,
    0.8784,    0.1255,         0,
    0.3765,    0.6275,         0,
    0.8784,    0.6275,         0,
    0.3765,    0.1255,    0.5020,
    0.8784,    0.1255,    0.5020,
    0.3765,    0.6275,    0.5020,
    0.8784,    0.6275,    0.5020,
    0.1255,    0.3765,         0,
    0.6275,    0.3765,         0,
    0.1255,    0.8784,         0,
    0.6275,    0.8784,         0,
    0.1255,    0.3765,    0.5020,
    0.6275,    0.3765,    0.5020,
    0.1255,    0.8784,    0.5020,
    0.6275,    0.8784,    0.5020,
    0.3765,    0.3765,         0,
    0.8784,    0.3765,         0,
    0.3765,    0.8784,         0,
    0.8784,    0.8784,         0,
    0.3765,    0.3765,    0.5020,
    0.8784,    0.3765,    0.5020,
    0.3765,    0.8784,    0.5020,
    0.8784,    0.8784,    0.5020,
    0.1255,    0.1255,    0.2510,
    0.6275,    0.1255,    0.2510,
    0.1255,    0.6275,    0.2510,
    0.6275,    0.6275,    0.2510,
    0.1255,    0.1255,    0.7529,
    0.6275,    0.1255,    0.7529,
    0.1255,    0.6275,    0.7529,
    0.6275,    0.6275,    0.7529,
    0.3765,    0.1255,    0.2510,
    0.8784,    0.1255,    0.2510,
    0.3765,    0.6275,    0.2510,
    0.8784,    0.6275,    0.2510,
    0.3765,    0.1255,    0.7529,
    0.8784,    0.1255,    0.7529,
    0.3765,    0.6275,    0.7529,
    0.8784,    0.6275,    0.7529,
    0.1255,    0.3765,    0.2510,
    0.6275,    0.3765,    0.2510,
    0.1255,    0.8784,    0.2510,
    0.6275,    0.8784,    0.2510,
    0.1255,    0.3765,    0.7529,
    0.6275,    0.3765,    0.7529,
    0.1255,    0.8784,    0.7529,
    0.6275,    0.8784,    0.7529,
    0.3765,    0.3765,    0.2510,
    0.8784,    0.3765,    0.2510,
    0.3765,    0.8784,    0.2510,
    0.8784,    0.8784,    0.2510,
    0.3765,    0.3765,    0.7529,
    0.8784,    0.3765,    0.7529,
    0.3765,    0.8784,    0.7529,
    0.8784,    0.8784,    0.7529,
    # 1.0,       1.0,       1.0,
]).reshape(-1, 3)


def visualize_snapshot(means3d, colors, w2c, k, height, width, rgb=None):
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=int(w * view_scale), height=int(h * view_scale), visible=False)
    means3d = means3d.cpu().data.numpy()
    colors = colors.cpu().data.numpy()
    colors = colormap[np.arange(colors.shape[0]) % colormap.shape[0]]

    w2c = w2c.cpu().data.numpy()
    k = k
    pcd = o3d.geometry.PointCloud()

    numlines = 100
    randselect = np.random.randint(0, means3d.shape[0], size=(numlines,))
    means3d = means3d[randselect]
    colors = colors[randselect]

    # create point cloud object
    # pcd.points = o3d.utility.Vector3dVector(means3d)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # create lineset properties
    pts_trans = np.concatenate([means3d + np.array([[0.2*i, 0, 0,]]) for i in range(5)], axis=0)
    colors_trans = colormap[np.arange(numlines) % colormap.shape[0]]
    colors_trans = np.concatenate([colors_trans for i in range(5)], axis=0)
    pt_indices = np.arange(pts_trans.shape[0])
    lines = np.stack((pt_indices, pt_indices - numlines), -1)[numlines:]
    colors_lineset = colors_trans[numlines:]

    # create lineset object
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts_trans, np.float64))
    lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors_lineset, np.float64))
    lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(lines, np.int32))

    # initialize renderer
    render = rendering.OffscreenRenderer(width=width, height=height)

    # get pc trajectory
    pc_traj = o3d.geometry.PointCloud()
    pc_traj.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts_trans, np.float64))
    pc_traj.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors_trans, np.float64))
    pc_material = rendering.MaterialRecord()
    pc_material.base_color = [1.0, 1.0, 1.0, 1.0]  # setting to white lets our pointcloud colors override it
    pc_material.shader = "defaultLit"
    pc_material.point_size = 5
    render.scene.add_geometry("pc", pc_traj, pc_material)

    lineset_material = rendering.MaterialRecord()
    lineset_material.shader = "unlitLine"
    lineset_material.line_width = 3.0
    lineset_material.base_color = [1.0, 1.0, 1.0, 1.0]
    render.scene.add_geometry("lineset", lineset, lineset_material)

    # add camera and render
    render.setup_camera(k, w2c.T, width, height)
    # render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
    #                                  75000)
    # render.scene.scene.enable_sun_light(True)
    render.scene.show_axes(True)
    img = render.render_to_image()
    print("Saving image at test.png")
    o3d.io.write_image("test.png", img, 9)


def visualize_trajectories(pc_seq, w2c, k, height, width):
    # build lineset internal rep
    numlines = pc_seq[0].size(0)
    pcs = np.concatenate([pc.cpu().numpy() for i, pc in enumerate(pc_seq)], axis=0)
    colors = colormap[np.arange(numlines) % colormap.shape[0]]
    colors_pc = np.concatenate([colors for i in range(len(pc_seq))], axis=0)
    colors_lines = colors_pc[numlines:]

    pt_indices = np.arange(pcs.shape[0])
    lines = np.stack((pt_indices, pt_indices - numlines), -1)[numlines:]

    # instantiate opencv renderer
    render = rendering.OffscreenRenderer(width=width, height=height)

    # instantiate pc
    pc_traj = o3d.geometry.PointCloud()
    pc_traj.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pcs, np.float64))
    pc_traj.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors_pc, np.float64))
    pc_material = rendering.MaterialRecord()
    pc_material.base_color = [1.0, 1.0, 1.0, 1.0]  # setting to white lets our pointcloud colors override it
    pc_material.shader = "defaultLit"
    pc_material.point_size = 3
    render.scene.add_geometry("pc", pc_traj, pc_material)

    # instantiate lineset object
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pcs, np.float64))
    lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors_lines, np.float64))
    lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(lines, np.int32))
    lineset_material = rendering.MaterialRecord()
    lineset_material.shader = "unlitLine"
    lineset_material.line_width = 1.0
    lineset_material.base_color = [1.0, 1.0, 1.0, 1.0]
    render.scene.add_geometry("lineset", lineset, lineset_material)

    # add camera and render
    render.setup_camera(k, w2c.T, width, height)
    render.scene.show_axes(False)
    img = render.render_to_image()
    return torch.from_numpy(np.array(img)).to(pc_seq[0].device)
