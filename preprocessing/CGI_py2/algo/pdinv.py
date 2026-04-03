"""
================================================================================
pdinv.py - 正定矩阵求逆
================================================================================

【文件作用】
使用 Cholesky 分解计算正定矩阵的逆矩阵。相比直接求逆，Cholesky 分解
在数值上更稳定且效率更高。如果 Cholesky 分解失败，则回退到 SVD 方法。

【使用方法】
```python
import numpy as np
from pdinv import pdinv

# 创建正定矩阵
A = np.array([[4, 2, 0],
              [2, 5, 2],
              [0, 2, 4]], dtype=float)

# 求逆
A_inv = pdinv(A)
print("原矩阵:\n", A)
print("逆矩阵:\n", A_inv)
print("验证 A @ A_inv:\n", A @ A_inv)
```

【参数说明】
- A: 正定矩阵，shape = (n, n)

【返回值】
- Ainv: A 的逆矩阵，shape = (n, n)

【算法原理】
1. 优先尝试 Cholesky 分解: A = L @ L^T
2. 通过三角方程组求解逆矩阵
3. 若 Cholesky 失败（矩阵非正定），使用 SVD 回退

【应用场景】
- 高斯过程回归中的协方差矩阵求逆
- 线性系统求解
- 统计推断

【依赖】
- numpy
- scipy.linalg

================================================================================
"""

import numpy as np
from scipy import linalg


def pdinv(A: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a positive definite matrix.

    Uses Cholesky decomposition for efficiency, with SVD fallback
    for non-positive definite matrices.

    Args:
        A: Positive definite matrix

    Returns:
        Ainv: Inverse of A
    """
    n = A.shape[0]

    try:
        U = linalg.cholesky(A, lower=False)
        invU = linalg.solve_triangular(U, np.eye(n), lower=False)
        Ainv = invU.T @ invU
    except linalg.LinAlgError:
        # Fall back to SVD if Cholesky fails
        U, S, V = linalg.svd(A)
        Ainv = V.T @ np.diag(1.0 / S) @ U.T

    return Ainv
