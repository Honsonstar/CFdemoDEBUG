"""
================================================================================
kernel.py - 径向基函数 (RBF) 核计算
================================================================================

【文件作用】
实现 RBF (Radial Basis Function) 高斯核函数计算，用于在特征空间中度量数据点
之间的相似性。这是 KCIT 算法和 GP 回归的核心组件，支持自动带宽选择。

【使用方法】
```python
import numpy as np
from kernel import kernel, kernel_matrix

# 两组数据之间的核矩阵
x = np.array([[1, 2], [3, 4], [5, 6]])
x_kern = np.array([[2, 3], [4, 5]])
theta = [1.0, 1.0]  # [length_scale, variance]
K, bw = kernel(x, x_kern, theta)

# 单组数据的核矩阵
K = kernel_matrix(x, theta)
```

【参数说明】
- x: 输入点，shape = (n1, d)
- x_kern: 核计算点，shape = (n2, d)
- theta: 超参数 [length_scale, variance]
  - length_scale: 核长度尺度，控制函数的平滑程度
  - variance: 核方差，控制输出振幅

【返回值】
- kx: 核矩阵，shape = (n1, n2)
- bw_new: 1/length_scale^2

【算法原理】
- RBF 核函数: K(x, x') = variance * exp(-||x - x'||^2 / (2 * length_scale^2))
- 自动带宽选择: 使用中位数启发式方法 (median heuristic)
  - 计算成对距离的中位数
  - length_scale = 2 / median_dist^2

【应用场景】
- 高斯过程回归中的核函数
- KCIT 条件独立性检验
- 非线性相似性度量

================================================================================
"""

import numpy as np
from .dist2 import dist2


def kernel(x: np.ndarray, x_kern: np.ndarray, theta: np.ndarray) -> tuple:
    """
    Compute the RBF kernel matrix.

    Args:
        x: Input points (n1, d)
        x_kern: Input points (n2, d)
        theta: Hyperparameters [length_scale, variance]

    Returns:
        kx: Kernel matrix (n1, n2)
        bw_new: 1/length_scale^2
    """
    # ========== [KERNEL TRACE] 输入形状检查 ==========
    # print(f"[KERNEL_TRACE] kernel called | x.shape={x.shape} | x_kern.shape={x_kern.shape} | theta.shape={theta.shape}", flush=True)

    # 防御性检查：确保输入是2D
    if x.ndim != 2:
        print(f"[KERNEL_ERROR] x is not 2D! x.shape={x.shape}, reshaping...", flush=True)
        x = x.reshape(-1, 1)
    if x_kern.ndim != 2:
        print(f"[KERNEL_ERROR] x_kern is not 2D! x_kern.shape={x_kern.shape}, reshaping...", flush=True)
        x_kern = x_kern.reshape(-1, 1)

    n2 = dist2(x, x_kern)

    if theta[0] == 0:
        # Automatic bandwidth selection using median heuristic
        n2_valid = n2[np.tril_indices_from(n2, k=-1)]
        n2_valid = n2_valid[n2_valid > 0]
        if len(n2_valid) > 0:
            theta[0] = 2 / np.median(n2_valid)
        else:
            theta[0] = 1.0

    wi2 = theta[0] / 2
    kx = theta[1] * np.exp(-n2 * wi2)
    bw_new = 1 / theta[0]

    return kx, bw_new


def kernel_matrix(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute the kernel matrix for a single set of points.

    Args:
        x: Input points (n, d)
        theta: Hyperparameters [length_scale, variance]

    Returns:
        K: Kernel matrix (n, n)
    """
    n2 = dist2(x, x)
    wi2 = theta[0] / 2
    K = theta[1] * np.exp(-n2 * wi2)
    return K
