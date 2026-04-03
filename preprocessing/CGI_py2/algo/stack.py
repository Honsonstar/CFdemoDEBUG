"""
================================================================================
stack.py - 矩阵堆叠为向量
================================================================================

【文件作用】
将矩阵按列堆叠成一个向量。这是 GPML 工具箱中的辅助函数，用于将
二维数据转换为一维向量，便于某些算法操作。

【使用方法】
```python
import numpy as np
from stack import stack

# 将矩阵按列堆叠
M = np.array([[1, 4, 7],
              [2, 5, 8],
              [3, 6, 9]])

v = stack(M)
print("原矩阵:\n", M)
print("堆叠向量:", v)
# 输出: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

【参数说明】
- M: 输入矩阵，shape = (n, t)

【返回值】
- v: 堆叠向量，shape = (n * t,)，按列顺序排列

【算法原理】
按列顺序依次堆叠矩阵元素：
v = [M[:,0], M[:,1], ..., M[:,t-1]]

【应用场景】
- GPML 工具箱中的辅助函数
- 数据格式转换
- 向量化操作

【依赖】
- numpy

================================================================================
"""

import numpy as np


def stack(M: np.ndarray) -> np.ndarray:
    """
    Stack the matrix M into a vector.

    Args:
        M: Input matrix (n, t)

    Returns:
        v: Stacked vector of length n*t
    """
    n, t = M.shape
    v = np.zeros(n * t)

    for i in range(t):
        v[i * n:(i + 1) * n] = M[:, i]

    return v
