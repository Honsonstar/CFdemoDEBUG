"""
================================================================================
dist2.py - 欧氏距离计算
================================================================================

【文件作用】
计算两组数据点之间的平方欧氏距离矩阵。这是核函数计算的基础组件。

【使用方法】
```python
import numpy as np
from dist2 import dist2

# 两组数据
X = np.array([[1, 2], [3, 4], [5, 6]])  # 3个点，2维
C = np.array([[1, 2], [7, 8]])            # 2个中心，2维

# 计算距离矩阵
D = dist2(X, C)
# D[i,j] = ||X[i] - C[j]||^2
# 结果 shape = (3, 2)
```

【参数说明】
- x: 数据矩阵，shape = (ndata, dim)
- c: 中心矩阵，shape = (ncentres, dim)

【返回值】
- 距离矩阵 D，shape = (ndata, ncentres)
- D[i,j] = sum((x[i,:] - c[j,:])^2)

【注意事项】
- 输入的两矩阵必须具有相同的列维度
- 由于数值精度问题，可能产生微小的负值，会被自动置为0

【算法复杂度】
- 时间复杂度: O(ndata * ncentres * dim)

================================================================================
DIST2 - Calculates squared distance between two sets of points.

D = DIST2(X, C) takes two matrices of vectors and calculates the
squared Euclidean distance between them. Both matrices must be of the
same column dimension. If X has M rows and N columns, and C has L rows
and N columns, then the result has M rows and L columns.

Copyright (c) Ian T Nabney (1996-2001)
================================================================================
"""

import numpy as np


def dist2(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Calculate squared Euclidean distance between two sets of points.

    Args:
        x: (ndata, dimx) array
        c: (ncentres, dimc) array

    Returns:
        n2: (ndata, ncentres) array of squared distances
    """
    ndata, dimx = x.shape
    ncentres, dimc = c.shape

    if dimx != dimc:
        raise ValueError('Data dimension does not match dimension of centres')

    # Compute squared distances using vectorized operations
    n2 = (np.ones((ncentres, 1)) * np.sum(x ** 2, axis=1, keepdims=True).T).T + \
         np.ones((ndata, 1)) * np.sum(c ** 2, axis=1, keepdims=True).T - \
         2 * np.dot(x, c.T)

    # Rounding errors occasionally cause negative entries in n2
    n2[n2 < 0] = 0

    return n2
