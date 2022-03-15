# This code is mainly referencing to:
# https://github.com/airalcorn2/pytorch-nerf

from load_bottles import load_bottles_data
from model import get_model
from sample import get_coarse_query_points, get_fine_query_points
from render import render_radiance_volume

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import random
import os as ops

def run_one_iter_of_nerf(
    ds, N_c, t_i_c_bin_edges, t_i_c_gap, os, chunk_size, F_c, N_f, t_f, F_f
):
    (r_ts_c, t_is_c) = get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os)
    (C_rs_c, w_is_c, _) = render_radiance_volume(r_ts_c, ds, chunk_size, F_c, t_is_c)

    (r_ts_f, t_is_f) = get_fine_query_points(w_is_c, N_f, t_is_c, t_f, os, ds)
    (C_rs_f, _, depth_map) = render_radiance_volume(r_ts_f, ds, chunk_size, F_f, t_is_f)

    return (C_rs_c, C_rs_f, depth_map)


def main():
    # Set seed and device.
    seed = 291
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    
    # Hyperparameters
    # Training
    iter_start = 300000
    num_iters = 600000
    eval_every = 30000
    chunk_size = 1024 * 64 # Number of query points passed through the MLP at a time
    batch_img_size = 32
    n_batch_pix = batch_img_size**2 # Number of training rays per iteration
    lr = 1e-3
    lrate_decay = 1000
    decay_steps = lrate_decay * 1000
    decay_rate = 0.1

    # Volume rendering
    t_n = 1.0 # Near bound
    t_f = 5.0 # Far bound
    N_c = 64 # Number of coarse samples along a ray
    N_f = 128 # Number of fine samples along a ray
    
    # Load dataset.
    data_dir = "./bottles"
    f_cd, train_set, val_set, test_poses = \
        load_bottles_data(data_dir, opencv_format=False, low_res=False)
    focal, camera_dis = f_cd
    train_poses, train_imgs = train_set
    val_poses_copy, val_imgs_copy = val_set
    img_size = train_imgs.shape[1]

    # Rearrange train set and validation set
    val_to_use_idx = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    val_poses, val_imgs = val_poses_copy[val_to_use_idx], val_imgs_copy[val_to_use_idx]
    val_poses_copy = np.delete(val_poses_copy, val_to_use_idx, axis=0)
    val_imgs_copy = np.delete(val_imgs_copy, val_to_use_idx, axis=0)
    train_poses = np.concatenate((train_poses, val_poses_copy), axis=0)
    train_imgs = np.concatenate((train_imgs, val_imgs_copy), axis=0)

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

    # Initialize coarse and fine MLPs.
    F_c = get_model(device)
    F_f = get_model(device)

    ######################## Load previous process ########################
    F_c.load_state_dict(torch.load("log/model/latest_coarse.pt"))
    F_f.load_state_dict(torch.load("log/model/latest_fine.pt"))

    # Initialize optimizer. See Section 5.3.
    optimizer = optim.Adam(list(F_c.parameters()) + list(F_f.parameters()), lr=lr)
    criterion = nn.MSELoss()

    # Start training model.
    # Preparation
    images = torch.Tensor(train_imgs)
    poses = torch.Tensor(train_poses)
    val_imgs = torch.Tensor(val_imgs)
    val_poses = torch.Tensor(val_poses)
    n_pix = img_size**2
    pixel_ps = torch.full((n_pix,), 1 / n_pix).to(device)

    # Log list
    iternums = []
    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []

    F_c.train()
    F_f.train()
    print("Start training...")
    if not ops.path.isdir('log'):
        ops.makedirs('log')
    if not ops.path.isdir('log/evaluation'):
        ops.makedirs('log/evaluation')
    if not ops.path.isdir('log/model'):
        ops.makedirs('log/model')
    if not ops.path.isdir('result'):
        ops.makedirs('result')
    for i in range(iter_start, num_iters):
        if i % 1000 == 0:
            print(f"progress: {i}/{num_iters}")
        # Sample image and associated pose.
        target_img_idx = np.random.randint(images.shape[0])
        target_pose = poses[target_img_idx].to(device)
        R = target_pose[:3, :3]

        # Get rotated ray origins (os) and ray directions (ds). See Section 4.
        ds = torch.einsum("ij,hwj->hwi", R, init_ds)
        os = (R @ init_o).expand(ds.shape)

        # Sample a batch of rays.
        pix_idxs = pixel_ps.multinomial(n_batch_pix, False)
        pix_idx_rows = torch.div(pix_idxs, img_size, rounding_mode='floor')
        pix_idx_cols = pix_idxs % img_size
        ds_batch = ds[pix_idx_rows, pix_idx_cols].reshape(
            batch_img_size, batch_img_size, -1
        )
        os_batch = os[pix_idx_rows, pix_idx_cols].reshape(
            batch_img_size, batch_img_size, -1
        )

        # Run NeRF.
        (C_rs_c, C_rs_f, _) = run_one_iter_of_nerf(
            ds_batch,
            N_c,
            t_i_c_bin_edges,
            t_i_c_gap,
            os_batch,
            chunk_size,
            F_c,
            N_f,
            t_f,
            F_f,
        )
        target_img = images[target_img_idx].to(device)
        target_img_batch = target_img[pix_idx_rows, pix_idx_cols].reshape(C_rs_f.shape)
        # Calculate the mean squared error for both the coarse and fine MLP models and
        # update the weights. See Equation (6) in Section 5.3.
        loss = criterion(C_rs_c, target_img_batch) + criterion(C_rs_f, target_img_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Exponentially decay learning rate. See Section 5.3 and:
        # https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/.
        for g in optimizer.param_groups:
            g["lr"] = lr * decay_rate ** (i / decay_steps)

        # Evaluation
        if i % eval_every == 0:
            # Choose two random samples from train set and validation set.
            eval_imgs = []
            eval_poses = []
            eval_idx = [random.randint(0, 189), random.randint(0, 9)]
            eval_imgs.append(torch.Tensor(train_imgs[eval_idx[0]]))
            eval_imgs.append(torch.Tensor(val_imgs[eval_idx[1]]))
            eval_poses.append(torch.Tensor(train_poses[eval_idx[0]]))
            eval_poses.append(torch.Tensor(val_poses[eval_idx[1]]))
            C_rs_fs = []
            F_c.eval()
            F_f.eval()
            for eval_pose in eval_poses:
                eval_R = eval_pose[:3, :3].to(device)
                eval_ds = torch.einsum("ij,hwj->hwi", eval_R, init_ds)
                eval_os = (eval_R @ init_o).expand(eval_ds.shape)

                # Generate one row of the map at a time to avoid running out of memory.
                C_rs_f = []
                for row in range(img_size):
                    with torch.no_grad():
                        (_, C_rs_f_row, _) = run_one_iter_of_nerf(
                            eval_ds[row:row+1, :],
                            N_c,
                            t_i_c_bin_edges,
                            t_i_c_gap,
                            eval_os[row:row+1, :],
                            chunk_size,
                            F_c,
                            N_f,
                            t_f,
                            F_f,
                        )
                        C_rs_f.append(C_rs_f_row)
                C_rs_f = torch.vstack(C_rs_f)
                C_rs_f = torch.clamp(C_rs_f, 0, 1)
                C_rs_fs.append(C_rs_f)

            iternums.append(i)
            train_loss = criterion(C_rs_fs[0], eval_imgs[0].to(device))
            val_loss = criterion(C_rs_fs[1], eval_imgs[1].to(device))
            train_psnr = -10.0 * torch.log10(train_loss)
            val_psnr = -10.0 * torch.log10(val_loss)
            print(f"Evaluated. train loss: {train_loss.item()}, train psnr: {train_psnr.item()}, validation loss: {val_loss.item()}, validation psnr: {val_psnr.item()}")
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            train_psnrs.append(train_psnr.item())
            val_psnrs.append(val_psnr.item())

            # Visualization
            plt.figure(figsize=(16, 9), constrained_layout=True)
            plt.suptitle(f"Iteration {iter_start} to {i}")
            plt.subplot(221)
            plt.title(f"Train Image {eval_idx[0]} Ground Truth")
            plt.imshow(eval_imgs[0].detach().cpu().numpy())
            plt.subplot(222)
            plt.title(f"Train Image {eval_idx[0]} Predicted RGB Map")
            plt.imshow(C_rs_fs[0].detach().cpu().numpy())
            plt.subplot(223)
            plt.title(f"Validation Image {eval_idx[1]} Ground Truth")
            plt.imshow(eval_imgs[1].detach().cpu().numpy())
            plt.subplot(224)
            plt.title(f"Validation Image {eval_idx[1]} Predicted RGB Map")
            plt.imshow(C_rs_fs[1].detach().cpu().numpy())
            plt.savefig(f"log/evaluation/iter_{i}_evaluation.png")
            plt.close()

            plt.figure()
            plt.title("Loss against Iteration")
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.plot(iternums, train_losses, label = 'train')
            plt.plot(iternums, val_losses, label = 'val')
            plt.savefig("log/loss.png")
            plt.close()

            plt.figure()
            plt.title("PSNR against Iteration")
            plt.xlabel('Iteration')
            plt.ylabel('PSNR')
            plt.plot(iternums, train_psnrs, label = 'train')
            plt.plot(iternums, val_psnrs, label = 'val')
            plt.savefig("log/psnr.png")
            plt.close()

            F_c.train()
            F_f.train()

            # Save model
            torch.save(F_c.state_dict(), "log/model/latest_coarse.pt")
            torch.save(F_f.state_dict(), "log/model/latest_fine.pt")

    print("Training complete!")

    # Generate final test views.
    print("Generating predictions for test poses...")
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

        plt.imsave(f"result/final_rgb_map_{i}.png", C_rs_f.detach().cpu().numpy())
        plt.imsave(f"result/final_depth_map_{i}.png", depth_map.detach().cpu().numpy())

if __name__ == "__main__":
    main()