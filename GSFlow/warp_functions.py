"""
Warp Functions for Gaussian Flow

This module provides warping utilities to transform Gaussian flow 
based on depth and camera geometry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BackprojectDepth(nn.Module):
    """
    Layer to transform a depth image into a point cloud.
    
    This module back-projects 2D depth values into 3D camera space
    using the inverse camera intrinsics.
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        """
        Back-project depth to 3D points.
        
        Args:
            depth (torch.Tensor): Depth map, shape [B, 1, H, W]
            inv_K (torch.Tensor): Inverse camera intrinsics, shape [B, 4, 4]
            
        Returns:
            torch.Tensor: 3D points in camera space, shape [B, 4, H*W]
        """
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """
    Layer which projects 3D points into a camera with intrinsics K and at position T.
    
    This module projects 3D points into 2D image space using camera 
    intrinsics and extrinsics.
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points (torch.Tensor): 3D points, shape [B, 4, H*W]
            K (torch.Tensor): Camera intrinsics, shape [B, 4, 4]
            T (torch.Tensor): Camera transformation, shape [B, 4, 4]
            
        Returns:
            tuple: (normalized_coords, pixel_coords)
                - normalized_coords: Normalized coordinates [-1, 1], shape [B, H, W, 2]
                - pixel_coords: Pixel coordinates, shape [B, H, W, 2]
        """
        P = torch.matmul(K, T)[:, :3, :]  # B 3 4
        cam_points = torch.matmul(P, points)  # B 3 H*W
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2:3, :] + self.eps)  # B 2 H*W
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)  # B 2 H W
        pix_coords = pix_coords.permute(0, 2, 3, 1)  # B H W 2
        
        # Normalize to [-1, 1] for grid_sample
        _pix_coords_ = torch.clone(pix_coords)
        _pix_coords_[..., 0] /= self.width - 1
        _pix_coords_[..., 1] /= self.height - 1
        _pix_coords_ = (_pix_coords_ - 0.5) * 2
        return _pix_coords_, pix_coords


def warping_gs_flow(depth_gt, gs_flow, camera_pose, next_camera_pose):
    """
    Warp Gaussian flow from one camera view to another using depth information.
    
    This function warps the Gaussian flow field to account for camera motion,
    allowing the flow to be properly aligned in the target camera frame.
    
    Args:
        depth_gt (torch.Tensor): Ground truth depth map, shape [(B), (1), H, W]
        gs_flow (torch.Tensor): Gaussian flow to warp, shape [2, H, W]
        camera_pose: Camera pose object with intrinsic and extrinsic matrices
        next_camera_pose: Next camera pose object with intrinsic and extrinsic matrices
        
    Returns:
        torch.Tensor: Warped Gaussian flow, shape [2, H, W]
    """
    H, W = depth_gt.shape[-2:]  # depth_gt: (B) (1) H W
    backprojdepth = BackprojectDepth(1, H, W).cuda()
    project3d = Project3D(1, H, W).cuda()
    
    inv_K1 = torch.linalg.inv(camera_pose.intrinsic.cuda())[None]  # B 4 4
    K2 = next_camera_pose.intrinsic.cuda()[None]  # B 4 4
    T12 = torch.matmul(torch.linalg.inv(next_camera_pose.extrinsic.cuda()), 
                     camera_pose.extrinsic.cuda())[None]  # B 4 4
    
    points_3d = backprojdepth(depth_gt, inv_K1)  # B 4 HW
    pixel_coords_norm, _ = project3d(points_3d, K2, T12)  # B H W 2
    gs_flow_warped = F.grid_sample(gs_flow.unsqueeze(0), pixel_coords_norm, 
                                   padding_mode="border", align_corners=True)  # B 2 H W
    return gs_flow_warped.squeeze(0)
