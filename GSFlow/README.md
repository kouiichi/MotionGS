# GSFlow Module

GSFlow模块提供了从4D高斯点云渲染中提取光流的核心功能，专注于Gaussian Flow的生成，不包含camera flow和motion flow的计算。

## 模块结构

```
GSFlow/
├── __init__.py              # 模块初始化和导出
├── gs_flow_generator.py     # 高斯光流计算核心函数
├── warp_functions.py        # 光流变换和投影函数
└── README.md               # 本文档
```

## 主要功能

### 1. Gaussian Flow 计算 (`gs_flow_generator.py`)

**核心函数**: `calculate_gs_flow()`

该函数基于各向异性的高斯变换计算光流。它考虑了：
- 高斯点的位置变化
- 高斯协方差的转换
- 像素级别的加权融合

**输入参数**:
- `gs_per_pixel`: 每个像素对应的高斯索引 [K, H, W]
- `weight_per_gs_pixel`: 每个高斯在像素上的权重 [K, H, W]
- `next_conic_2D`: 下一帧的2D圆锥参数 [N, 3]
- `conic_2D_inv`: 当前帧的逆2D圆锥参数 [N, 3]
- `proj_2D`: 当前帧的2D投影 [N, 2]
- `next_proj_2D`: 下一帧的2D投影 [N, 2]
- `x_mu`: 像素相对高斯中心的位移 [K, 2, H, W]

**输出**:
- 高斯光流张量 [2, H, W]

**使用示例**:
```python
from GSFlow import calculate_gs_flow

# 从渲染器获取必要参数
gs_flow = calculate_gs_flow(
    gs_per_pixel, 
    weight_per_gs_pixel, 
    next_conic_2D, 
    conic_2D_inv, 
    proj_2D, 
    next_proj_2D, 
    x_mu
)
```

### 2. 光流变换 (`warp_functions.py`)

**核心函数**: `warping_gs_flow()`

该函数使用深度信息将高斯光流从一个相机视角变换到另一个视角。

**输入参数**:
- `depth_gt`: 深度图 [(B), (1), H, W]
- `gs_flow`: 待变换的高斯光流 [2, H, W]
- `camera_pose`: 当前相机位姿（包含内参和外参）
- `next_camera_pose`: 下一帧相机位姿

**输出**:
- 变换后的高斯光流 [2, H, W]

**使用示例**:
```python
from GSFlow import warping_gs_flow

# 对光流进行相机变换
warped_flow = warping_gs_flow(
    depth_gt, 
    gs_flow, 
    camera_pose, 
    next_camera_pose
)
```

### 3. 面向对象接口

`GaussianFlowGenerator` 类提供了更清晰的API接口：

```python
from GSFlow import GaussianFlowGenerator

flow_generator = GaussianFlowGenerator()
gs_flow = flow_generator.compute_flow(
    gs_per_pixel, weight_per_gs_pixel, 
    next_conic_2D, conic_2D_inv, 
    proj_2D, next_proj_2D, x_mu
)
```

## 技术细节

### Gaussian Flow 计算原理

Gaussian Flow基于以下物理模型：

1. **协方差变换**: 计算高斯的协方差矩阵在两帧之间的变换
   ```
   conv_conv = next_conic @ conic_inv
   ```

2. **各向异性流**: 通过协方差变换调整像素位移
   ```
   conv_multi = conv_conv @ x_mu
   ```

3. **最终光流**: 结合变换后的位移和位置变化
   ```
   flow = conv_multi + (next_proj - proj) - x_mu
   ```

4. **加权融合**: 使用渲染权重对多个高斯的流进行加权平均

### 与MotionGS的区别

本模块专注于Gaussian Flow的计算，与原始MotionGS的区别在于：

- **剔除了**: camera flow和motion flow的计算
- **保留了**: 纯粹的Gaussian Flow生成逻辑
- **优化了**: 模块化设计，便于集成到4DGS训练流程

## 依赖要求

- PyTorch >= 1.10
- NumPy
- CUDA (用于GPU加速)

运行 `python check_requirements.py` 检查所有依赖是否已安装。

## 集成指南

要将GSFlow集成到4DGS训练流程中：

1. 导入模块:
```python
from GSFlow import calculate_gs_flow, warping_gs_flow
```

2. 在训练循环中计算光流:
```python
# 渲染当前帧和下一帧
render_pkg_1 = render(viewpoint_cam1, ...)
render_pkg_2 = render(viewpoint_cam2, ...)

# 计算Gaussian Flow
gs_flow = calculate_gs_flow(
    render_pkg_2["gs_per_pixel"],
    render_pkg_2["weight_per_gs_pixel"],
    render_pkg_2_next["conic_2D"],
    render_pkg_2["conic_2D_inv"],
    render_pkg_2["proj_2D"],
    render_pkg_2_next["proj_2D"],
    render_pkg_2["x_mu"]
)

# 变换光流以匹配相机运动
gs_flow = warping_gs_flow(depth, gs_flow, viewpoint_cam1, viewpoint_cam2)
```

3. 使用光流监督训练:
```python
from utils.flow_loss_utils import combined_flow_l1_loss

loss_dict = combined_flow_l1_loss(
    image_pred, image_gt,
    gs_flow, flow_gt,
    height, width,
    flow_weight=1.0
)
loss = loss_dict['total']
```

## 注意事项

1. 所有张量计算都在CUDA上进行，确保数据已移至GPU
2. 光流计算依赖于渲染器输出的特定参数，确保使用兼容的渲染器
3. 深度图和相机位姿信息对于光流变换至关重要

## 参考

基于MotionGS项目中的光流计算方法，专门提取和优化用于4DGS训练。
