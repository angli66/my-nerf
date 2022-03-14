from load_bottles import load_bottles_data
from model import get_model
from run_nerf import run_one_iter_of_nerf

import numpy as np
import torch
import matplotlib.pyplot as plt

import os as ops

def generate_prediction():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Hyperparameters
    chunk_size = 1024 * 32
    t_n = 1.0 # Near bound
    t_f = 5.0 # Far bound
    N_c = 64 # Number of coarse samples along a ray
    N_f = 128 # Number of fine samples along a ray

    # Load dataset.
    data_dir = "./bottles"
    f_cd, train_set, val_set, test_poses = \
        load_bottles_data(data_dir, opencv_format=False, low_res=False)
    focal, camera_dis = f_cd
    _, train_imgs = train_set
    _, _ = val_set
    img_size = train_imgs.shape[1]

    # Set up initial ray origin (init_o) and ray directions (init_ds). These are the
    # same across samples, we just rotate them based on the orientation of the camera.
    # See Section 4.
    xs = torch.arange(img_size) - img_size / 2
    ys = torch.arange(img_size) - img_size / 2
    (xs, ys) = torch.meshgrid(xs, -ys, indexing="xy")
    pixel_coords = torch.stack([xs, ys, -focal * torch.ones_like(xs)], dim=-1)
    # We want the zs to be negative ones, so we divide everything by the focal length
    # (which is in pixel units).
    camera_coords = pixel_coords / focal
    init_ds = camera_coords.to(device)
    init_o = torch.Tensor(np.array([0, 0, camera_dis])).to(device)

    # Initialize bins used to sample depths along a ray. See Equation (2) in Section 4.
    t_i_c_gap = (t_f - t_n) / N_c
    t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)

    # Load coarse and fine MLPs.
    F_c = get_model(device)
    F_f = get_model(device)
    F_c.load_state_dict(torch.load("log/model/latest_coarse.pt"))
    F_f.load_state_dict(torch.load("log/model/latest_fine.pt"))

     # Generate final test views.
    print("Generating predictions for test poses...")
    if not ops.path.isdir('prediction'):
        ops.makedirs('prediction')
    test_poses = torch.Tensor(test_poses)
    for i, test_pose in enumerate(test_poses):
        print(f"Generating prediction {i}...")
        test_R = test_pose[:3, :3].to(device)
        test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
        test_os = (test_R @ init_o).expand(test_ds.shape)

        # Generate one row of the map at a time to avoid running out of memory.
        C_rs_f = []
        depth_map = []
        for row in range(img_size):
            with torch.no_grad():
                (_, C_rs_f_row, depth_map_row) = run_one_iter_of_nerf(
                    test_ds[row:row+1, :],
                    N_c,
                    t_i_c_bin_edges,
                    t_i_c_gap,
                    test_os[row:row+1, :],
                    chunk_size,
                    F_c,
                    N_f,
                    t_f,
                    F_f,
                )
                C_rs_f.append(C_rs_f_row)
                depth_map.append(depth_map_row)
        C_rs_f = torch.vstack(C_rs_f)
        C_rs_f = torch.clamp(C_rs_f, 0, 1)
        depth_map = torch.vstack(depth_map)

        plt.imsave(f"prediction/rgb_map_{i}.png", C_rs_f.detach().cpu().numpy())
        plt.imsave(f"prediction/depth_map_{i}.png", depth_map.detach().cpu().numpy())

if __name__ == '__main__':
    generate_prediction()