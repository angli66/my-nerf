# This code is mainly referencing to:
# https://github.com/airalcorn2/pytorch-nerf

from load_bottles import load_bottles_data
from model import get_model
from sample import get_coarse_query_points, get_fine_query_points
from render import render_radiance_volume

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    
    # Hyperparameters
    # Training
    num_iters = 1000 # 300000
    eval_every = 100
    chunk_size = 1024 * 32 # Number of query points passed through the MLP at a time
    batch_img_size = 32 # Number of training rays per iteration
    n_batch_pix = batch_img_size**2
    lr = 5e-4
    lrate_decay = 250
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
        load_bottles_data(data_dir, opencv_format=False, low_res=True)
    focal, camera_dis = f_cd
    train_poses, train_imgs = train_set
    val_poses, val_imgs = val_set
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

    # Initialize coarse and fine MLPs.
    F_c = get_model(device)
    F_f = get_model(device)

    # Initialize optimizer. See Section 5.3.
    optimizer = optim.Adam(list(F_c.parameters()) + list(F_f.parameters()), lr=lr)
    criterion = nn.MSELoss()

    # Initialize bins used to sample depths along a ray. See Equation (2) in Section 4.
    t_i_c_gap = (t_f - t_n) / N_c
    t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)

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
    val_losses = []
    psnrs_1 = []
    psnrs_2 = []

    F_c.train()
    F_f.train()
    print("Start training...")
    for i in range(num_iters):
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

        if i % eval_every == 0:
            # Choose two random samples from validation set to use for evaluation
            eval_idx = torch.randint(0, 100, (2,))
            eval_imgs = val_imgs[eval_idx]
            eval_poses = val_poses[eval_idx]
            C_rs_fs = []
            F_c.eval()
            F_f.eval()
            for eval_pose in eval_poses:
                eval_R = eval_pose[:3, :3].to(device)
                eval_ds = torch.einsum("ij,hwj->hwi", eval_R, init_ds)
                eval_os = (eval_R @ init_o).expand(eval_ds.shape)

                # # Generate one row of the map at a time to avoid out of memory
                # C_rs_f = []
                # depth_map = []
                # for i in range(len(eval_pose)):
                with torch.no_grad():
                    (_, C_rs_f, _) = run_one_iter_of_nerf(
                        eval_ds,
                        N_c,
                        t_i_c_bin_edges,
                        t_i_c_gap,
                        eval_os,
                        chunk_size,
                        F_c,
                        N_f,
                        t_f,
                        F_f,
                    )

                C_rs_fs.append(C_rs_f)

            eval_imgs = eval_imgs.to(device)
            iternums.append(i)
            loss_1 = criterion(C_rs_fs[0], eval_imgs[0])
            loss_2 = criterion(C_rs_fs[1], eval_imgs[1])
            val_loss = loss_1.item() + loss_2.item()
            val_losses.append(val_loss)
            print(f"progress: {i}/{num_iters}, validation loss: {val_loss / 2}")
            psnr_1 = -10.0 * torch.log10(loss_1)
            psnr_2 = -10.0 * torch.log10(loss_2)
            psnrs_1.append(psnr_1.item())
            psnrs_2.append(psnr_2.item())

            # Visualization
            plt.figure(figsize=(16, 9), constrained_layout=True)
            plt.suptitle(f"Iteration 1 to {i}")
            plt.subplot(231)
            plt.title(f"Validation Image {eval_idx[0]} Ground Truth")
            plt.imshow(eval_imgs[0].detach().cpu().numpy())
            plt.subplot(232)
            plt.title(f"Validation Image {eval_idx[0]} Predicted RGB Map")
            plt.imshow(C_rs_fs[0].detach().cpu().numpy())
            plt.subplot(233)
            plt.title(f"Validation Image {eval_idx[0]} PSNR")
            plt.plot(iternums, psnrs_1)
            plt.subplot(234)
            plt.title(f"Validation Image {eval_idx[1]} Ground Truth")
            plt.imshow(eval_imgs[1].detach().cpu().numpy())
            plt.subplot(235)
            plt.title(f"Validation Image {eval_idx[1]} Predicted RGB Map")
            plt.imshow(C_rs_fs[1].detach().cpu().numpy())
            plt.subplot(236)
            plt.title(f"Validation Image {eval_idx[1]} PSNR")
            plt.plot(iternums, psnrs_2)
            plt.savefig(f"log/evaluation/iter_{i}_evaluation.png")
            plt.close()

            plt.figure()
            plt.plot(iternums, val_losses)
            plt.savefig("log/validation_loss.png")
            plt.close()

            F_c.train()
            F_f.train()

            # Save model
            torch.save(F_c.state_dict(), "log/model/latest_coarse.pt")
            torch.save(F_f.state_dict(), "log/model/latest_fine.pt")
            if (len(val_losses) == 0 or val_loss <= min(val_losses)):
                torch.save(F_c.state_dict(), "log/model/best_coarse.pt")
                torch.save(F_f.state_dict(), "log/model/best_fine.pt")

    print("Training complete!")

    # Generate test views.
    print("Generating predictions for test poses...")
    test_poses = torch.Tensor(test_poses)
    for i, test_pose in enumerate(test_poses):
        print(f"Generating prediction {i}...")
        test_R = test_pose[:3, :3].to(device)
        test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
        test_os = (test_R @ init_o).expand(test_ds.shape)
        with torch.no_grad():
            (_, C_rs_f, depth_map) = run_one_iter_of_nerf(
                test_ds,
                N_c,
                t_i_c_bin_edges,
                t_i_c_gap,
                test_os,
                chunk_size,
                F_c,
                N_f,
                t_f,
                F_f,
            )
        plt.imsave(f"result/rgb_map_{i}.png", C_rs_f.detach().cpu().numpy())
        plt.imsave(f"result/depth_map_{i}.png", depth_map.detach().cpu().numpy())

if __name__ == "__main__":
    main()