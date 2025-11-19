"""
Gaussian Flow Generator

This module contains the core function to calculate Gaussian flow from 
rendered Gaussian splatting outputs. The flow is computed based on the 
anisotropic transformation of Gaussians between two time steps.
"""

import torch


def calculate_gs_flow(gs_per_pixel, weight_per_gs_pixel, next_conic_2D, conic_2D_inv, 
                     proj_2D, next_proj_2D, x_mu):
    """
    Calculate Gaussian flow based on anisotropic Gaussian transformations.
    
    This function computes the optical flow induced by the motion of Gaussian splats
    between two consecutive frames. It accounts for both the positional change and
    the covariance transformation of each Gaussian.
    
    Args:
        gs_per_pixel (torch.Tensor): Gaussian indices per pixel, shape [K, H, W]
        weight_per_gs_pixel (torch.Tensor): Weight of each Gaussian per pixel, shape [K, H, W]
        next_conic_2D (torch.Tensor): 2D conic parameters at next frame, shape [N, 3]
        conic_2D_inv (torch.Tensor): Inverse 2D conic parameters at current frame, shape [N, 3]
        proj_2D (torch.Tensor): 2D projections at current frame, shape [N, 2]
        next_proj_2D (torch.Tensor): 2D projections at next frame, shape [N, 2]
        x_mu (torch.Tensor): Pixel displacement from Gaussian center, shape [K, 2, H, W]
        
    Returns:
        torch.Tensor: Gaussian flow, shape [2, H, W]
    """
    conic_2D_inv = conic_2D_inv.detach()  # K 3

    gs_per_pixel = gs_per_pixel.long()  # K H W
    
    # Compute covariance matrix multiplication: next_conic @ conic_inv
    # This represents the anisotropic transformation of the Gaussian
    conv_conv = torch.zeros([conic_2D_inv.shape[0], 2, 2], device=conic_2D_inv.device)  # K 2 2
    conv_conv[:, 0, 0] = next_conic_2D[:, 0] * conic_2D_inv[:, 0] + next_conic_2D[:, 1] * conic_2D_inv[:, 1]
    conv_conv[:, 0, 1] = next_conic_2D[:, 0] * conic_2D_inv[:, 1] + next_conic_2D[:, 1] * conic_2D_inv[:, 2]
    conv_conv[:, 1, 0] = next_conic_2D[:, 1] * conic_2D_inv[:, 0] + next_conic_2D[:, 2] * conic_2D_inv[:, 1]
    conv_conv[:, 1, 1] = next_conic_2D[:, 1] * conic_2D_inv[:, 1] + next_conic_2D[:, 2] * conic_2D_inv[:, 2]

    # Anisotropic Gaussian flow computation
    # Transform pixel displacement by covariance transformation
    conv_multi = (conv_conv[gs_per_pixel] @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze()  # K H W 2
    
    # Final flow combines the transformed displacement and positional change
    flow_per_pixel = (conv_multi + next_proj_2D[gs_per_pixel] - 
                     proj_2D[gs_per_pixel].detach() - x_mu.permute(0,2,3,1).detach())  # K H W 2

    # Normalize weights to sum to 1
    weight_per_gs_pixel = weight_per_gs_pixel / (weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7)  # K H W
    
    # Weighted sum of per-Gaussian flows
    flow_gs = torch.einsum("khw, khwa -> ahw", [weight_per_gs_pixel.detach(), flow_per_pixel])  # 2 H W
    
    return flow_gs


class GaussianFlowGenerator:
    """
    A class-based interface for generating Gaussian flows.
    
    This wrapper provides a cleaner API for computing Gaussian flows
    and can be extended with additional functionality in the future.
    """
    
    def __init__(self):
        """Initialize the Gaussian Flow Generator."""
        pass
    
    def compute_flow(self, gs_per_pixel, weight_per_gs_pixel, next_conic_2D, 
                    conic_2D_inv, proj_2D, next_proj_2D, x_mu):
        """
        Compute Gaussian flow using the underlying calculation function.
        
        Args:
            Same as calculate_gs_flow function
            
        Returns:
            torch.Tensor: Gaussian flow, shape [2, H, W]
        """
        return calculate_gs_flow(gs_per_pixel, weight_per_gs_pixel, next_conic_2D,
                                conic_2D_inv, proj_2D, next_proj_2D, x_mu)
    
    def __call__(self, *args, **kwargs):
        """Allow the instance to be called directly."""
        return self.compute_flow(*args, **kwargs)
