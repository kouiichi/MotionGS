# GSFlow 集成指南

本指南详细说明如何将GSFlow模块集成到现有的4DGS (4D Gaussian Splatting) 项目中，以使用光流监督提升训练效果。

## 概述

本项目基于MotionGS，提取并模块化了Gaussian Flow（高斯光流）的计算方法，去除了camera flow和motion flow的部分，专注于纯粹的Gaussian Flow监督。

## 项目结构

```
MotionGS/
├── GSFlow/                          # 新增：GSFlow模块
│   ├── __init__.py                 # 模块初始化
│   ├── gs_flow_generator.py        # 高斯光流计算
│   ├── warp_functions.py           # 光流变换函数
│   ├── example_usage.py            # 使用示例
│   └── README.md                   # 模块文档
├── utils/
│   ├── flow_loss_utils.py          # 新增：光流损失函数
│   └── ...                         # 其他工具
├── check_requirements.py            # 新增：依赖检查脚本
└── ...
```

## 安装和依赖检查

### 1. 检查依赖

运行依赖检查脚本：

```bash
python check_requirements.py
```

该脚本会检查：
- 核心Python包（torch, numpy, opencv等）
- 可选包（wandb, tensorboard等）
- CUDA扩展
- GPU可用性

### 2. 安装缺失依赖

如果有缺失的依赖，运行：

```bash
# 安装Python包
pip install -r requirements.txt

# 编译CUDA扩展
cd submodules/diff-gaussian-rasterization-w-pose && pip install -e .
cd ../flow-diff-gaussian-rasterization && pip install -e .
```

## 使用GSFlow模块

### 方法1：函数式接口

```python
from GSFlow import calculate_gs_flow, warping_gs_flow

# 计算Gaussian Flow
gs_flow = calculate_gs_flow(
    gs_per_pixel,           # 每个像素的高斯索引 [K, H, W]
    weight_per_gs_pixel,    # 每个高斯的权重 [K, H, W]
    next_conic_2D,          # 下一帧的圆锥参数 [N, 3]
    conic_2D_inv,           # 当前帧的逆圆锥参数 [N, 3]
    proj_2D,                # 当前帧投影 [N, 2]
    next_proj_2D,           # 下一帧投影 [N, 2]
    x_mu                    # 像素位移 [K, 2, H, W]
)

# 变换光流以匹配相机运动
gs_flow_warped = warping_gs_flow(
    depth,              # 深度图
    gs_flow,           # 计算的高斯光流
    camera_pose,       # 当前相机位姿
    next_camera_pose   # 下一帧相机位姿
)
```

### 方法2：面向对象接口

```python
from GSFlow import GaussianFlowGenerator

# 初始化生成器
flow_generator = GaussianFlowGenerator()

# 计算光流
gs_flow = flow_generator.compute_flow(
    gs_per_pixel, weight_per_gs_pixel,
    next_conic_2D, conic_2D_inv,
    proj_2D, next_proj_2D, x_mu
)

# 或使用可调用接口
gs_flow = flow_generator(...)
```

## 集成光流监督

### 基本集成步骤

#### 1. 导入必要模块

```python
from GSFlow import calculate_gs_flow, warping_gs_flow
from utils.flow_loss_utils import combined_flow_l1_loss, compute_flow_metrics
```

#### 2. 在训练循环中计算光流

```python
# 渲染两帧
render_pkg_1 = render(viewpoint_cam1, gaussians, pipe, background, 
                      d_xyz, d_rotation, d_scaling)
render_pkg_2_1 = render(viewpoint_cam2, gaussians, pipe, background, 
                        d_xyz, d_rotation, d_scaling)  # 使用相同形变
render_pkg_2 = render(viewpoint_cam2, gaussians, pipe, background, 
                      d_xyz_2, d_rotation_2, d_scaling_2)  # 使用下一帧形变

# 提取必要参数
gs_per_pixel = render_pkg_2_1["gs_per_pixel"]
weight_per_gs_pixel = render_pkg_2_1["weight_per_gs_pixel"]
conic_2D_inv = render_pkg_2_1["conic_2D_inv"]
proj_2D = render_pkg_2_1["proj_2D"]
x_mu = render_pkg_2_1["x_mu"]
next_conic_2D = render_pkg_2["conic_2D"]
next_proj_2D = render_pkg_2["proj_2D"]

# 计算Gaussian Flow
gs_flow = calculate_gs_flow(
    gs_per_pixel, weight_per_gs_pixel,
    next_conic_2D, conic_2D_inv,
    proj_2D, next_proj_2D, x_mu
)

# 变换光流
depth = render_pkg_1["depth"].detach()
gs_flow = warping_gs_flow(depth, gs_flow, viewpoint_cam1, viewpoint_cam2)
```

#### 3. 计算光流监督损失

```python
# 获取ground truth光流（需要预先计算或使用光流网络）
flow_gt = get_optical_flow(gt_image, next_gt_image)

# 计算组合损失：图像L1 + SSIM + 光流L1
H, W = image.shape[-2:]
loss_dict = combined_flow_l1_loss(
    image, gt_image,          # 渲染图像和真实图像
    gs_flow, flow_gt,         # 预测光流和真实光流
    H, W,
    flow_weight=1.0,          # 光流损失权重
    lambda_dssim=0.2          # SSIM权重
)

total_loss = loss_dict['total']
```

#### 4. 监控训练指标

```python
# 定期计算光流评估指标
if iteration % 100 == 0:
    metrics = compute_flow_metrics(gs_flow, flow_gt, H, W)
    print(f"EPE: {metrics['epe']:.4f}, "
          f"L1: {metrics['l1']:.4f}, "
          f"Angular: {metrics['angular']:.2f}°")
```

## 损失函数选项

### 1. 完整损失（推荐）

```python
from utils.flow_loss_utils import combined_flow_l1_loss

loss_dict = combined_flow_l1_loss(
    image_pred, image_gt,
    flow_pred, flow_gt,
    H, W,
    flow_weight=1.0,
    lambda_dssim=0.2
)
```

包含：
- L1 图像损失
- SSIM 结构损失  
- 光流 L1 损失

### 2. 简化损失

```python
from utils.flow_loss_utils import flow_supervised_loss

loss = flow_supervised_loss(
    image_pred, image_gt,
    flow_pred, flow_gt,
    H, W,
    flow_weight=1.0,
    use_l1_only=True  # 只用L1，不用SSIM
)
```

### 3. 纯光流损失

```python
from utils.flow_loss_utils import flow_loss

Lflow = flow_loss(flow_pred, flow_gt, H, W)
```

## 替换原有训练代码

### 原有4DGS训练代码

```python
# 原来的损失计算
Ll1 = l1_loss(image, gt_image)
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
```

### 替换为光流监督

```python
# 新的损失计算（添加光流监督）
from utils.flow_loss_utils import combined_flow_l1_loss

# 计算Gaussian Flow
gs_flow = calculate_gs_flow(...)
gs_flow = warping_gs_flow(...)

# 组合损失
loss_dict = combined_flow_l1_loss(
    image, gt_image,
    gs_flow, flow_gt,
    H, W,
    flow_weight=opt.flow_loss_weight,
    lambda_dssim=opt.lambda_dssim
)
loss = loss_dict['total']
```

## 获取Ground Truth光流

有几种方法获取训练所需的ground truth光流：

### 方法1：使用预训练光流网络

```python
from gmflow.gmflow import build_gmflow
from gmflow.config import get_cfg as get_gmflow_cfg

# 加载GMFlow
cfg = get_gmflow_cfg()
flownet = build_gmflow(cfg)
flownet.load_state_dict(torch.load('path/to/checkpoint'))
flownet = flownet.cuda().eval()

# 计算光流
with torch.no_grad():
    flow_gt = flownet(image1 * 255, image2 * 255)[0].squeeze()
```

### 方法2：预计算并缓存

```python
# 预计算所有帧对的光流并保存
flow_cache = {}
for i in range(len(frames) - 1):
    flow = compute_flow(frames[i], frames[i+1])
    flow_cache[i] = flow
    torch.save(flow, f'flows/flow_{i}.pt')

# 训练时加载
flow_gt = torch.load(f'flows/flow_{frame_id}.pt')
```

## 超参数建议

基于MotionGS的经验，推荐以下超参数：

```python
# 训练参数
flow_loss_weight = 1.0      # 光流损失权重
lambda_dssim = 0.2          # SSIM权重
warm_up = 3000              # 预热迭代（之后才启用光流）

# 在训练循环中
if iteration >= warm_up:
    # 使用光流监督
    loss_dict = combined_flow_l1_loss(...)
    loss = loss_dict['total']
else:
    # 预热阶段只用图像损失
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(...))
```

## 完整示例

参考 `GSFlow/example_usage.py` 获取完整的集成示例代码。

## 常见问题

### Q1: 渲染器需要返回哪些参数？

A: 渲染器必须返回以下参数以支持GSFlow：
- `gs_per_pixel`: 每像素的高斯索引
- `weight_per_gs_pixel`: 每高斯的权重
- `proj_2D`: 2D投影
- `conic_2D`: 2D圆锥参数
- `conic_2D_inv`: 逆2D圆锥参数
- `x_mu`: 像素位移

### Q2: 如何确保渲染器兼容？

A: 使用 `flow-diff-gaussian-rasterization` 子模块中的渲染器，它已经支持返回所需参数。

### Q3: 光流计算对性能的影响？

A: Gaussian Flow计算是轻量级的，主要开销在于：
1. 额外的一次渲染（用于下一帧）
2. 光流变换（可忽略）
3. 光流损失计算（可忽略）

总体性能影响约10-15%。

### Q4: 可以只使用部分功能吗？

A: 可以。模块设计为解耦的：
- 只用 `calculate_gs_flow` 而不用 `warping_gs_flow`
- 只用光流损失而不用其他监督
- 自定义损失权重和组合方式

## 技术支持

如有问题，请参考：
1. `GSFlow/README.md` - GSFlow模块详细文档
2. `GSFlow/example_usage.py` - 使用示例
3. 原始MotionGS项目文档

## 总结

集成GSFlow到4DGS的核心步骤：

1. ✅ 安装依赖（运行 `check_requirements.py`）
2. ✅ 导入GSFlow模块
3. ✅ 在训练循环中计算Gaussian Flow
4. ✅ 应用光流监督损失
5. ✅ 监控训练指标

通过光流监督，您的4DGS模型将获得：
- 更好的时序一致性
- 更准确的运动建模
- 更高的渲染质量
