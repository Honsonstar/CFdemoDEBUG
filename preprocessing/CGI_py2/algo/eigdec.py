"""
================================================================================
eigdec.py - 排序特征分解
================================================================================

【文件作用】
计算矩阵的最大 N 个特征值（降序排列），并可选地返回对应的特征向量。
这是 HSIC 统计量计算和主成分分析 (PCA) 的核心组件。

【使用方法】
```python
import numpy as np
from eigdec import eigdec, eigdec_evals_only

# 创建对称正定矩阵
A = np.array([[3, 1, 0],
              [1, 2, 1],
              [0, 1, 1]])

# 计算最大的3个特征值和特征向量
evals, evec = eigdec(A, 3)
print(f"特征值: {evals}")
print(f"特征向量:\n{evec}")

# 仅计算特征值
evals_only = eigdec_evals_only(A, 2)
print(f"最大的2个特征值: {evals_only}")
```

【参数说明】
- x: 输入矩阵 (n, n)，应为对称矩阵
- N: 要计算的特征值数量

【返回值】
- evals: 最大的 N 个特征值（降序排列），shape = (N,)
- evec: 对应的特征向量，shape = (n, N)，可选

【算法原理】
- 使用 scipy.linalg.eigh 进行特征分解（针对对称矩阵优化）
- 结果按特征值降序排列
- eigdec_evals_only 只计算特征值，更节省内存

【应用场景】
- HSIC 统计量计算（核方法中的中心矩阵）
- 主成分分析 (PCA)
- 谱聚类

【依赖】
- numpy
- scipy.linalg

================================================================================
"""

import numpy as np
from scipy import linalg


def eigdec(x: np.ndarray, N: int) -> tuple:
    """
    Compute the largest N eigenvalues and optionally eigenvectors.

    Args:
        x: Input matrix
        N: Number of eigenvalues to compute

    Returns:
        evals: (N,) array of largest N eigenvalues in descending order
        evec: (n, N) array of corresponding eigenvectors (if requested)
    """
    n = x.shape[0]

    if N < 1 or N > n:
        raise ValueError('Number of PCs must be integer, >0, < dim')

    # Use eig function as it's generally more reliable
    temp_evals, temp_evec = linalg.eigh(x)
    # eigh returns in ascending order, so we need to reverse
    temp_evals = temp_evals[::-1]
    temp_evec = temp_evec[:, ::-1]

    evals = temp_evals[:N]

    if N == len(temp_evals):
        evec = temp_evec[:, :N]
    else:
        evec = temp_evec[:, :N]

    return evals, evec


def eigdec_evals_only(x: np.ndarray, N: int) -> np.ndarray:
    """
    Compute only the largest N eigenvalues.

    Args:
        x: Input matrix
        N: Number of eigenvalues to compute

    Returns:
        evals: (N,) array of largest N eigenvalues
    """
    temp_evals = np.linalg.eigvalsh(x)
    evals = np.sort(temp_evals)[::-1][:N]
    return evals
