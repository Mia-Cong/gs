import numpy as np
import torch
import torch.nn.functional as F


def propagate_imu(camm1):
    """
    Propagate camera pose based on IMU measurements
    """
    # TODO
    return camm1


def propagate_const_vel(camm1, camm2):
    """
    Propagate camera pose based on constant velocity model

    Args:
        camm1 (tensor): pose at idx-1
        camm2 (tensor): pose at idx-2
    Returns:
        cam (tensor): propagated pose
    """
    # pre_w2c = get_camera_from_tensor(camm1)
    # delta = pre_w2c @ get_camera_from_tensor(camm2).inverse()
    # cam = get_tensor_from_camera(delta @ pre_w2c)
    #
    # print(cam)

    prev_rot1 = F.normalize(camm1[:4].clone().unsqueeze(0)).detach()
    prev_rot2 = F.normalize(camm2[:4].clone().unsqueeze(0)).detach()
    new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2)).squeeze().detach()
    # Translation
    prev_tran1 = camm1[-3:].clone().detach()
    prev_tran2 = camm2[-3:].clone().detach()
    new_tran = prev_tran1 + (prev_tran1 - prev_tran2).detach()

    # print(torch.cat([new_rot, new_tran]))

    return torch.cat([new_rot, new_tran])


def quadmultiply(q1, q2):
    """
    Multiply two quaternions together using quaternion arithmetic
    """
    # Extract scalar and vector parts of the quaternions
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    # Calculate the quaternion product
    result_quaternion = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )

    return result_quaternion


def quad2rotation(q):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    # bs = quad.shape[0]
    # qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    # two_s = 2.0 / (quad * quad).sum(-1)
    # rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    # rot_mat[:, 0, 0] = 1 - two_s * (qj**2 + qk**2)
    # rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    # rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    # rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    # rot_mat[:, 1, 1] = 1 - two_s * (qi**2 + qk**2)
    # rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    # rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    # rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    # rot_mat[:, 2, 2] = 1 - two_s * (qi**2 + qj**2)
    # return rot_mat
    norm = torch.sqrt(
        q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    )
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3)).to(q.get_device())
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    # normalize quad
    quad = F.normalize(quad)
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    # Add homogenous row
    homogenous_row = torch.tensor([0, 0, 0, 1]).cuda()
    RT = torch.cat([RT, homogenous_row[None, None, :].repeat(N, 1, 1)], 1)
    if N == 1:
        RT = RT[0]

    return RT


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            gpu_id = RT.get_device()
            RT = RT.detach().cpu()
        RT = RT.numpy()
    from mathutils import Matrix

    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor
