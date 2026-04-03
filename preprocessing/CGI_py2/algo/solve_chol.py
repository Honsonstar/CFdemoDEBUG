"""
================================================================================
solve_chol.py - Cholesky 分解求解线性方程组
================================================================================

【文件作用】
利用 Cholesky 分解高效求解线性方程组 L @ L^T @ X = B。
相比直接求解，A @ X = B 的效率更高，数值稳定性更好。

【使用方法】
```python
import numpy as np
from solve_chol import solve_chol

# 创建正定矩阵 A 和右端项 B
A = np.array([[4, 2, 0],
              [2, 5, 2],
              [0, 2, 4]], dtype=float)

# 计算 Cholesky 分解
L = np.linalg.cholesky(A)
print("Cholesky 因子 L:\n", L)

# 求解 L @ L^T @ X = B
B = np.array([1, 2, 3], dtype=float)
X = solve_chol(L, B)
print("解 X:", X)
print("验证 A @ X:", A @ X)
```

【参数说明】
- L: 下三角 Cholesky 因子 (from chol(A) where A = L @ L.T)
- B: 右端矩阵/向量

【返回值】
- X: 方程组的解

【算法原理】
求解 L @ L^T @ X = B 分两步：
1. 求解 L @ Y = B (下三角方程)
2. 求解 L^T @ X = Y (上三角方程)

【应用场景】
- 高斯过程中的协方差矩阵相关计算
- 线性系统高效求解
- 统计推断

【依赖】
- numpy
- scipy.linalg

================================================================================
"""

import numpy as np
from scipy import linalg


def solve_chol(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve L @ L^T @ X = B for X, where L is lower triangular Cholesky factor.

    Args:
        L: Lower triangular Cholesky factor (from chol(A) where A = L @ L.T)
        B: Right-hand side matrix/vector

    Returns:
        X: Solution matrix/vector
    """
    return linalg.solve_triangular(L.T, linalg.solve_triangular(L, B, lower=True), lower=False)
