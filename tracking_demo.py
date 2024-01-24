import torch
from scene import Scene
import os
import sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.pose_utils import (
    get_camera_from_tensor,
    get_tensor_from_camera,
    quadmultiply,
)
from utils.loss_utils import l1_loss

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def compute_relative_world_to_camera(R2, t2, R1, t1):
    zero_row = torch.tensor([[0, 0, 0, 1]], dtype=R1.dtype, device=R1.device)
    E1_inv = torch.cat(
        [torch.transpose(R1, 0, 1), -torch.transpose(R1, 0, 1) @ t1.reshape(-1, 1)],
        dim=1,
    )
    E1_inv = torch.cat([E1_inv, zero_row], dim=0)
    E2 = torch.cat([R2, -R2 @ t2.reshape(-1, 1)], dim=1)
    E2 = torch.cat([E2, zero_row], dim=0)
    E_rel = E2 @ E1_inv
    return E_rel


def prune_gaussians(gaussians, opac_thres):
    # Prune Gaussians that have opacity below opac_thres
    mask = gaussians.get_opacity > opac_thres
    attributes = [
        "_xyz",
        "_scaling",
        "_rotation",
        "_opacity",
        "_features_dc",
        "_features_rest",
    ]
    for attr in attributes:
        a = getattr(gaussians, attr)
        setattr(
            gaussians,
            attr,
            a[torch.squeeze(mask)],
        )


def optimize_cam(
    opt,
    view,
    gaussians,
    pipeline,
    background,
    optimizer,
    camera_tensor_q,
    camera_tensor_T,
) -> float:
    # We can keep camera orientation fixed, and then track a rigid body transformation
    # of the Gaussians around the camera

    # Apply w2c transform to gaussians
    rel_w2c = get_camera_from_tensor(torch.cat([camera_tensor_q, camera_tensor_T]))
    gaussians_xyz = gaussians._xyz.clone().detach()
    gaussians_rot = gaussians._rotation.clone().detach()

    pts_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
    pts4 = torch.cat((gaussians_xyz, pts_ones), dim=1)
    gaussians_xyz_trans = (rel_w2c @ pts4.T).T[:, :3]
    gaussians_rot_trans = quadmultiply(camera_tensor_q, gaussians_rot)
    gaussians._xyz = gaussians_xyz_trans
    gaussians._rotation = gaussians_rot_trans

    # Render
    result = render(view, gaussians, pipeline, background, fix_camera=True)
    image = result["render"]

    # Loss
    gt_image = view.original_image.cuda()
    loss = l1_loss(image, gt_image)
    loss.backward(retain_graph=True)

    # Optimize
    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Restore untransformed points
        gaussians._xyz = gaussians_xyz
        gaussians._rotation = gaussians_rot

    return loss, image


def track(dataset, opt, pp, checkpoint_iter: int, const_velocity):
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=checkpoint_iter, shuffle=False)
    # prune_gaussians(gaussians, 0.5)

    gt_viewpoints_list = scene.getTrainCameras().copy()

    # Load GT poses
    gt_camera_tensor_list = []
    for viewpoint in gt_viewpoints_list:
        # Convert W2C transformation matrix to a quaternion + translation vector
        w2c = viewpoint.world_view_transform.transpose(0, 1)
        camera_tensor = get_tensor_from_camera(w2c)
        gt_camera_tensor_list.append(camera_tensor)

    # Create saved list of all camera poses
    camera_tensor_list = torch.zeros([len(gt_viewpoints_list), 7]).cuda()
    pos_np = np.zeros([len(gt_viewpoints_list), 7])
    pos_np_init = np.zeros([len(gt_viewpoints_list), 7])
    pos_np_gt = np.zeros([len(gt_viewpoints_list), 7])

    # Get the first camera tensor as the starting point
    camera_tensor = gt_camera_tensor_list[0]
    camera_tensor_T = camera_tensor[-3:].requires_grad_()
    camera_tensor_q = camera_tensor[:4].requires_grad_()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    progress_bar = tqdm(gt_viewpoints_list, desc="Tracking progress")

    # Save renders for reference
    render_path = os.path.join(dataset.model_path, "tracking_renders")
    makedirs(render_path, exist_ok=True)

    # Pick the next camera
    for idx, view in enumerate(progress_bar):
        # Stop prematurely at specified idx
        if idx == 6:
            break
        if idx not in [0, 5]:
            continue

        # # If using the constant velocity model
        # if const_velocity and idx - 2 >= 0:
        #     pre_w2c = get_camera_from_tensor(camera_tensor)
        #     delta = (
        #         pre_w2c @ get_camera_from_tensor(camera_tensor_list[idx - 2]).inverse()
        #     )
        #     camera_tensor = get_tensor_from_camera(delta @ pre_w2c)
        #     camera_tensor_T = camera_tensor[-3:].requires_grad_()
        #     camera_tensor_q = camera_tensor[:4].requires_grad_()

        pose_optimizer = torch.optim.Adam(
            [
                {"params": [camera_tensor_T], "lr": 0.001},
                {"params": [camera_tensor_q], "lr": 0.001},
            ]
        )

        with torch.no_grad():
            pos_np_init[idx, :] = (
                torch.cat([camera_tensor_q, camera_tensor_T]).cpu().detach().numpy()
            )

        # For some iterations
        for cam_iter in range(500):
            loss, rendering = optimize_cam(
                opt,
                view,
                gaussians,
                pp,
                background,
                pose_optimizer,
                camera_tensor_q,
                camera_tensor_T,
            )

            with torch.no_grad():
                if cam_iter == 0:
                    initial_loss = loss

        with torch.no_grad():
            # Save images
            torchvision.utils.save_image(
                torch.cat(
                    [
                        view.original_image[0:3, :, :],
                        rendering,
                        torch.abs(rendering - view.original_image[0:3, :, :]),
                    ],
                    dim=2,
                ),
                os.path.join(render_path, "{0:05d}".format(idx) + ".png"),
            )

            progress_bar.set_postfix({"Loss Diff": f"{initial_loss:.4f}->{loss:.4f}"})
            progress_bar.update()

            camera_tensor_list[idx] = (
                torch.cat([camera_tensor_q, camera_tensor_T]).clone().detach()
            )
            pos_np[idx, :] = camera_tensor_list[idx].cpu().detach().numpy()
            pos_np_gt[idx, :] = gt_camera_tensor_list[idx].cpu().detach().numpy()

    # If we ended early, delete zero rows
    non_zero_rows_mask = np.any(pos_np != 0, axis=1)
    pos_np = pos_np[non_zero_rows_mask]
    pos_np_init = pos_np_init[non_zero_rows_mask]
    pos_np_gt = pos_np_gt[non_zero_rows_mask]

    T0 = get_camera_from_tensor(torch.tensor(pos_np[0, :]).cuda())
    T5 = get_camera_from_tensor(torch.tensor(pos_np[1, :]).cuda())
    R0 = T0[:3, :3]
    t0 = T0[:3, 3]
    R5 = T5[:3, :3]
    t5 = T5[:3, 3]
    est_rel = compute_relative_world_to_camera(R5, t5, R0, t0)
    print(est_rel)

    T0 = get_camera_from_tensor(torch.tensor(pos_np_gt[0, :]).cuda())
    T5 = get_camera_from_tensor(torch.tensor(pos_np_gt[1, :]).cuda())
    R0 = T0[:3, :3]
    t0 = T0[:3, 3]
    R5 = T5[:3, :3]
    t5 = T5[:3, 3]
    gt_rel = compute_relative_world_to_camera(R5, t5, R0, t0)
    print(gt_rel)

    diff = compute_relative_world_to_camera(
        est_rel[:3, :3], est_rel[:3, 3], gt_rel[:3, :3], gt_rel[:3, 3]
    )
    print(diff)

    np.save(scene.model_path + "/tracking_traj", pos_np, allow_pickle=True)
    np.save(scene.model_path + "/tracking_traj_init", pos_np_init, allow_pickle=True)
    np.save(scene.model_path + "/tracking_traj_gt", pos_np_gt, allow_pickle=True)

    progress_bar.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")
    ax1.plot(pos_np[:, 4], pos_np[:, 5], pos_np[:, 6], label="optim")
    ax1.plot(pos_np_gt[:, 4], pos_np_gt[:, 5], pos_np_gt[:, 6], label="gt")
    ax1.legend()
    plt.show()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Tracking script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--const_velocity", action="store_true", default=False)
    parser.add_argument("--pose_noise", type=float, default=0.0)

    # args = parser.parse_args(sys.argv[1:])
    args = get_combined_args(parser)
    print("Tracking " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    track(
        model.extract(args),
        op.extract(args),
        pipeline.extract(args),
        args.iteration,
        args.const_velocity,
    )

    print("\nTracking complete.")
