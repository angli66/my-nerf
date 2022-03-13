import numpy as np
import cv2
import os
import imageio

def _get_poses_from_dir(poses_dir, opencv_format):
    poses = []
    for pose_dir in poses_dir:
        with open(pose_dir, 'rb') as f:
            pose = np.loadtxt(f)
            if opencv_format: # Flip 2nd and 3rd column if in opencv format
                pose[:, 1:3] *= -1
            poses.append(pose)
    poses = np.array(poses).astype(np.float32)

    return poses

def _get_imgs_from_dir(imgs_dir, low_res):
    imgs = []
    for img_dir in imgs_dir:
        with open(img_dir, 'rb') as f:
            img = imageio.imread(f)
            imgs.append(img)
    imgs = (np.array(imgs) / 255.).astype(np.float32)

    if low_res:
        imgs_low_res = np.zeros((imgs.shape[0], 100, 100, 4))
        for i, img in enumerate(imgs):
            imgs_low_res[i] = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        imgs = imgs_low_res

    # Transform RGBA into RGB for white background image
    imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])

    return imgs

def load_bottles_data(data_dir, opencv_format=False, low_res=False):
    print("Loading bottles data set...")

    # Load intrinsic matrix
    with open(os.path.join(data_dir, 'intrinsics.txt'), 'r') as f:
        intrinsics = np.loadtxt(f)

    if low_res:
        print("Low resolution mode on")
        intrinsics[0:2, :] /= 8

    focal = intrinsics[0][0]
    camera_dis = 3.0

    # Load train set
    train_poses_dir = [os.path.join(data_dir, 'pose', f) for f in sorted(os.listdir(os.path.join(data_dir, 'pose'))) \
                        if f.startswith('0') and f.endswith('txt')]
    train_imgs_dir = [os.path.join(data_dir, 'rgb', f) for f in sorted(os.listdir(os.path.join(data_dir, 'rgb'))) \
                        if f.startswith('0') and f.endswith('png')]

    train_poses = _get_poses_from_dir(train_poses_dir, opencv_format)
    train_imgs = _get_imgs_from_dir(train_imgs_dir, low_res)

    # Load validation set
    val_poses_dir = [os.path.join(data_dir, 'pose', f) for f in sorted(os.listdir(os.path.join(data_dir, 'pose'))) \
                        if f.startswith('1') and f.endswith('txt')]
    val_imgs_dir = [os.path.join(data_dir, 'rgb', f) for f in sorted(os.listdir(os.path.join(data_dir, 'rgb'))) \
                        if f.startswith('1') and f.endswith('png')]

    val_poses = _get_poses_from_dir(val_poses_dir, opencv_format)
    val_imgs = _get_imgs_from_dir(val_imgs_dir, low_res)

    # Load test poses
    test_ids = ['0000', '0016', '0055', '0093', '0160']
    test_poses_dir = [os.path.join(data_dir, 'pose', f"2_test_{id}.txt") for id in test_ids]
    test_poses = _get_poses_from_dir(test_poses_dir, opencv_format)

    print("bottles data set loaded")

    return (focal, camera_dis), (train_poses, train_imgs), (val_poses, val_imgs), test_poses
