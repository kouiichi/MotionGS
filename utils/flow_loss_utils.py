"""
Flow Loss Utilities

This module provides loss functions combining optical flow supervision 
with traditional L1 loss for training 4D Gaussian Splatting models.
"""

import torch
import torch.nn.functional as F
from .loss_utils import l1_loss, ssim


def normalize_flow(flow, height, width):
    """
    Normalize flow values to [-1, 1] range based on image dimensions.
    
    Args:
        flow (torch.Tensor): Flow tensor, shape [2, H, W] or [B, 2, H, W]
        height (int): Image height
        width (int): Image width
        
    Returns:
        torch.Tensor: Normalized flow with same shape as input
    """
    normalized_flow = flow.clone()
    
    # Handle both [2, H, W] and [B, 2, H, W] shapes
    if len(flow.shape) == 3:
        normalized_flow[0] /= height
        normalized_flow[1] /= width
    else:
        normalized_flow[:, 0] /= height
        normalized_flow[:, 1] /= width
    
    normalized_flow = normalized_flow.clamp(-1, 1)
    return normalized_flow


def flow_loss(flow_pred, flow_gt, height, width):
    """
    Compute L1 loss between predicted and ground truth optical flow.
    
    Flow values are normalized by image dimensions before computing the loss.
    
    Args:
        flow_pred (torch.Tensor): Predicted flow, shape [2, H, W]
        flow_gt (torch.Tensor): Ground truth flow, shape [2, H, W]
        height (int): Image height for normalization
        width (int): Image width for normalization
        
    Returns:
        torch.Tensor: Scalar loss value
    """
    flow_pred_norm = normalize_flow(flow_pred, height, width)
    flow_gt_norm = normalize_flow(flow_gt, height, width)
    return l1_loss(flow_pred_norm, flow_gt_norm)


def combined_flow_l1_loss(image_pred, image_gt, flow_pred, flow_gt, height, width,
                          flow_weight=1.0, lambda_dssim=0.2, mask=None):
    """
    Combined loss function using both flow supervision and L1/SSIM image loss.
    
    This is the main loss function for training with optical flow supervision.
    It combines:
    1. Image reconstruction loss (L1 + SSIM)
    2. Optical flow loss (L1)
    
    Args:
        image_pred (torch.Tensor): Predicted rendered image, shape [3, H, W]
        image_gt (torch.Tensor): Ground truth image, shape [3, H, W]
        flow_pred (torch.Tensor): Predicted Gaussian flow, shape [2, H, W]
        flow_gt (torch.Tensor): Ground truth optical flow, shape [2, H, W]
        height (int): Image height
        width (int): Image width
        flow_weight (float): Weight for flow loss term (default: 1.0)
        lambda_dssim (float): Weight for SSIM loss (default: 0.2)
        mask (torch.Tensor, optional): Optional mask for loss computation
        
    Returns:
        dict: Dictionary containing:
            - 'total': Total combined loss
            - 'image_l1': L1 image loss component
            - 'image_ssim': SSIM loss component
            - 'flow': Flow loss component
    """
    # Image reconstruction loss
    if mask is not None:
        Ll1 = l1_loss(image_pred, image_gt, mask=mask)
        Lssim = 1.0 - ssim(image_pred, image_gt, mask=mask)
    else:
        Ll1 = l1_loss(image_pred, image_gt)
        Lssim = 1.0 - ssim(image_pred, image_gt)
    
    L_image = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * Lssim
    
    # Flow loss
    Lflow = flow_loss(flow_pred, flow_gt, height, width)
    
    # Combined loss
    L_total = L_image + flow_weight * Lflow
    
    return {
        'total': L_total,
        'image_l1': Ll1,
        'image_ssim': Lssim,
        'flow': Lflow
    }


def flow_supervised_loss(image_pred, image_gt, flow_pred, flow_gt, height, width,
                        flow_weight=1.0, use_l1_only=False, lambda_dssim=0.2):
    """
    Simplified flow-supervised loss function.
    
    This function provides a simpler interface for flow supervision, optionally
    using only L1 loss without SSIM.
    
    Args:
        image_pred (torch.Tensor): Predicted rendered image, shape [3, H, W]
        image_gt (torch.Tensor): Ground truth image, shape [3, H, W]
        flow_pred (torch.Tensor): Predicted Gaussian flow, shape [2, H, W]
        flow_gt (torch.Tensor): Ground truth optical flow, shape [2, H, W]
        height (int): Image height
        width (int): Image width
        flow_weight (float): Weight for flow loss term (default: 1.0)
        use_l1_only (bool): If True, only use L1 loss without SSIM (default: False)
        lambda_dssim (float): Weight for SSIM loss (default: 0.2)
        
    Returns:
        torch.Tensor: Combined loss value
    """
    if use_l1_only:
        L_image = l1_loss(image_pred, image_gt)
    else:
        Ll1 = l1_loss(image_pred, image_gt)
        Lssim = 1.0 - ssim(image_pred, image_gt)
        L_image = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * Lssim
    
    Lflow = flow_loss(flow_pred, flow_gt, height, width)
    
    return L_image + flow_weight * Lflow


def compute_flow_metrics(flow_pred, flow_gt, height, width):
    """
    Compute evaluation metrics for optical flow prediction.
    
    Args:
        flow_pred (torch.Tensor): Predicted flow, shape [2, H, W]
        flow_gt (torch.Tensor): Ground truth flow, shape [2, H, W]
        height (int): Image height
        width (int): Image width
        
    Returns:
        dict: Dictionary containing flow metrics:
            - 'epe': End-point error (average L2 distance)
            - 'l1': L1 error
            - 'angular': Angular error in degrees
    """
    flow_pred_norm = normalize_flow(flow_pred, height, width)
    flow_gt_norm = normalize_flow(flow_gt, height, width)
    
    # End-point error (EPE)
    epe = torch.norm(flow_pred_norm - flow_gt_norm, p=2, dim=0).mean()
    
    # L1 error
    l1_error = torch.abs(flow_pred_norm - flow_gt_norm).mean()
    
    # Angular error
    flow_pred_3d = torch.cat([flow_pred_norm, torch.ones_like(flow_pred_norm[0:1])], dim=0)
    flow_gt_3d = torch.cat([flow_gt_norm, torch.ones_like(flow_gt_norm[0:1])], dim=0)
    
    # Normalize vectors
    flow_pred_3d = F.normalize(flow_pred_3d, p=2, dim=0)
    flow_gt_3d = F.normalize(flow_gt_3d, p=2, dim=0)
    
    # Compute angular error
    cos_angle = (flow_pred_3d * flow_gt_3d).sum(dim=0).clamp(-1, 1)
    angular_error = torch.acos(cos_angle) * 180.0 / 3.14159265359
    angular_error = angular_error.mean()
    
    return {
        'epe': epe.item(),
        'l1': l1_error.item(),
        'angular': angular_error.item()
    }
