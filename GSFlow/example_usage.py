"""
Example Usage of GSFlow Module

This script demonstrates how to integrate GSFlow into a 4DGS training pipeline.
"""

import torch
from GSFlow import calculate_gs_flow, GaussianFlowGenerator, warping_gs_flow
from utils.flow_loss_utils import combined_flow_l1_loss, compute_flow_metrics


def example_training_step(viewpoint_cam1, viewpoint_cam2, gaussians, deform, 
                          pipe, background, iteration, opt):
    """
    Example training step showing how to use GSFlow for supervision.
    
    This is a simplified example showing the key integration points.
    """
    
    # 1. Get time inputs for deformation
    N = gaussians.get_xyz.shape[0]
    fid1 = viewpoint_cam1.fid
    fid2 = viewpoint_cam2.fid
    time_input_1 = fid1.unsqueeze(0).expand(N, -1)
    time_input_2 = fid2.unsqueeze(0).expand(N, -1)
    
    # 2. Compute deformations
    d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input_1)
    d_xyz_2, d_rotation_2, d_scaling_2 = deform.step(gaussians.get_xyz.detach(), time_input_2)
    
    # 3. Render both frames
    # Note: This assumes you have a render function compatible with GSFlow
    render_pkg_1 = render(viewpoint_cam1, gaussians, pipe, background, 
                         d_xyz, d_rotation, d_scaling)
    
    # Render frame 2 with frame 1's deformation (for flow computation)
    render_pkg_2_1 = render(viewpoint_cam2, gaussians, pipe, background, 
                           d_xyz, d_rotation, d_scaling)
    
    # Render frame 2 with frame 2's deformation  
    render_pkg_2 = render(viewpoint_cam2, gaussians, pipe, background, 
                         d_xyz_2, d_rotation_2, d_scaling_2)
    
    # 4. Extract rendered images and depth
    image = render_pkg_1["render"]
    depth = render_pkg_1["depth"].detach()
    
    # 5. Calculate Gaussian Flow using GSFlow module
    gs_flow = calculate_gs_flow(
        render_pkg_2_1["gs_per_pixel"],
        render_pkg_2_1["weight_per_gs_pixel"],
        render_pkg_2["conic_2D"],
        render_pkg_2_1["conic_2D_inv"],
        render_pkg_2_1["proj_2D"],
        render_pkg_2["proj_2D"],
        render_pkg_2_1["x_mu"]
    )
    
    # 6. Warp the flow to account for camera motion
    gs_flow = warping_gs_flow(depth, gs_flow, viewpoint_cam1, viewpoint_cam2)
    
    # 7. Get ground truth data
    gt_image = viewpoint_cam1.original_image.cuda()
    next_gt_image = viewpoint_cam2.original_image.cuda()
    
    # Assuming you have a method to compute or load ground truth optical flow
    flow_gt = compute_optical_flow_gt(gt_image, next_gt_image)  # Placeholder
    
    # 8. Compute combined loss with flow supervision
    H, W = image.shape[-2:]
    loss_dict = combined_flow_l1_loss(
        image, gt_image,
        gs_flow, flow_gt,
        H, W,
        flow_weight=opt.flow_loss_weight,
        lambda_dssim=opt.lambda_dssim
    )
    
    # 9. Extract losses
    total_loss = loss_dict['total']
    
    # Optional: Compute flow metrics for monitoring
    if iteration % 100 == 0:
        metrics = compute_flow_metrics(gs_flow, flow_gt, H, W)
        print(f"Iteration {iteration}: EPE={metrics['epe']:.4f}, "
              f"Angular Error={metrics['angular']:.2f}°")
    
    return total_loss, loss_dict


def example_with_class_interface():
    """
    Example using the class-based interface for GSFlow.
    """
    # Initialize the flow generator
    flow_generator = GaussianFlowGenerator()
    
    # Use it in your training loop
    # gs_flow = flow_generator.compute_flow(...)
    # or
    # gs_flow = flow_generator(...)  # Callable interface
    
    print("GaussianFlowGenerator initialized and ready to use")


def example_simple_flow_loss(image_pred, image_gt, flow_pred, flow_gt, H, W):
    """
    Example of using the simplified flow loss function.
    """
    from utils.flow_loss_utils import flow_supervised_loss
    
    # Simple interface - just L1 + Flow
    loss = flow_supervised_loss(
        image_pred, image_gt,
        flow_pred, flow_gt,
        H, W,
        flow_weight=1.0,
        use_l1_only=True  # Use only L1, no SSIM
    )
    
    return loss


def compute_optical_flow_gt(image1, image2):
    """
    Placeholder for optical flow ground truth computation.
    
    In practice, you would either:
    1. Use a pre-trained optical flow network (GMFlow, RAFT, etc.)
    2. Load pre-computed flow from disk
    """
    # This is just a placeholder - implement based on your needs
    H, W = image1.shape[-2:]
    return torch.zeros(2, H, W).cuda()


if __name__ == "__main__":
    print("=" * 60)
    print("GSFlow Module - Example Usage")
    print("=" * 60)
    print()
    print("This script demonstrates how to integrate GSFlow into 4DGS.")
    print()
    print("Key Integration Points:")
    print("1. Import GSFlow modules")
    print("2. Calculate Gaussian Flow during rendering")
    print("3. Warp flow to account for camera motion")
    print("4. Apply combined flow + L1 loss supervision")
    print("5. Monitor flow metrics during training")
    print()
    print("See the functions above for detailed implementation examples.")
    print()
    
    # Test class interface
    example_with_class_interface()
